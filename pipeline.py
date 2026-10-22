"""
Real Estate Due-Diligence Copilot — Core Pipeline
====================================================
Stages:
  1. OCR            -> extract raw text from scanned/PDF property documents
  2. RAG index       -> chunk + embed documents into a vector store for retrieval
  3. Multi-agent     -> 3 LangGraph agents collaborate on the due-diligence check
  4. Orchestration    -> a LangGraph StateGraph wires the agents into one flow

Run standalone:
    python pipeline.py --docs sample_docs/ --property "Plot 14, Balanagar, Hyderabad"
"""

import os
import json
import argparse
from typing import TypedDict, List, Optional

# ---------------------------------------------------------------------------
# 1. OCR
# ---------------------------------------------------------------------------
# Uses pytesseract for scanned images/PDF pages. For text-native PDFs, PyMuPDF
# is tried first (fast, no OCR needed); falls back to OCR only when a page
# has near-zero extractable text (i.e. it's a scanned image).

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


def ocr_pdf(path: str) -> str:
    """Extract text from a PDF, using native text first and OCR as fallback."""
    text_parts = []
    doc = fitz.open(path)
    for page in doc:
        native_text = page.get_text().strip()
        if len(native_text) > 30:
            text_parts.append(native_text)
        else:
            # Scanned page -> render to image -> OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            text_parts.append(ocr_text)
    doc.close()
    return "\n\n".join(text_parts)


def load_documents(doc_dir: str) -> dict:
    """OCR every file in doc_dir. Returns {filename: extracted_text}."""
    results = {}
    for fname in sorted(os.listdir(doc_dir)):
        fpath = os.path.join(doc_dir, fname)
        if fname.lower().endswith(".pdf"):
            results[fname] = ocr_pdf(fpath)
        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
            results[fname] = pytesseract.image_to_string(Image.open(fpath))
        elif fname.lower().endswith(".txt"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                results[fname] = f.read()
    return results


# ---------------------------------------------------------------------------
# 2. RAG index
# ---------------------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document


def build_vectorstore(doc_texts: dict, persist_dir: str = "faiss_index") -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks: List[Document] = []
    for fname, text in doc_texts.items():
        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append(Document(page_content=chunk, metadata={"source": fname, "chunk": i}))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(persist_dir)
    return vs


def load_vectorstore(persist_dir: str = "faiss_index") -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# 3. Multi-agent system (3 agents) + 4. LangGraph orchestration
# ---------------------------------------------------------------------------
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

REQUIRED_DOCS = [
    "sale deed", "encumbrance certificate", "property tax receipt",
    "khata certificate", "layout approval / DTCP-HMDA approval",
    "occupancy certificate", "link documents / chain of title", "survey sketch",
]


class DueDiligenceState(TypedDict):
    property_desc: str
    doc_texts: dict
    checklist_report: Optional[str]
    legal_findings: Optional[str]
    risk_report: Optional[str]


# --- Agent 1: Document Verification Agent -----------------------------------
doc_verification_prompt = ChatPromptTemplate.from_template(
    """You are a Document Verification Agent for Indian real estate due diligence.

Property: {property_desc}

Required document checklist:
{checklist}

Documents provided (filename -> extracted text, truncated):
{docs_summary}

For each required document, state: PRESENT / MISSING / UNCLEAR, and note any
obvious inconsistency (mismatched names, survey numbers, or dates across
documents). Be concise, output as a markdown table."""
)


def document_verification_node(state: DueDiligenceState) -> DueDiligenceState:
    docs_summary = "\n\n".join(
        f"### {fname}\n{text[:1500]}" for fname, text in state["doc_texts"].items()
    )
    chain = doc_verification_prompt | llm
    result = chain.invoke({
        "property_desc": state["property_desc"],
        "checklist": "\n".join(f"- {d}" for d in REQUIRED_DOCS),
        "docs_summary": docs_summary,
    })
    state["checklist_report"] = result.content
    return state


# --- Agent 2: Legal & Title Agent (RAG-grounded) -----------------------------
legal_prompt = ChatPromptTemplate.from_template(
    """You are a Legal & Title Verification Agent specializing in Indian property law
(Telangana RERA, Registration Act 1908, Transfer of Property Act).

Property: {property_desc}

Relevant retrieved excerpts from the property documents:
{context}

Document verification findings so far:
{checklist_report}

Check specifically for:
1. Clear, unbroken chain of title (at least 30 years, per standard practice)
2. Any pending litigation, court orders, or disputes mentioned
3. Encumbrances: mortgages, liens, or charges not yet cleared
4. Match between seller's name on the sale deed and the title chain
5. Zoning / land-use compliance (agricultural vs non-agricultural conversion, if applicable)

Output a concise legal findings report with a clear RED FLAG list (if any)."""
)


def legal_title_node(state: DueDiligenceState) -> DueDiligenceState:
    vs = load_vectorstore()
    query = f"title chain, encumbrance, litigation, mortgage for {state['property_desc']}"
    retrieved = vs.similarity_search(query, k=6)
    context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in retrieved)

    chain = legal_prompt | llm
    result = chain.invoke({
        "property_desc": state["property_desc"],
        "context": context,
        "checklist_report": state["checklist_report"],
    })
    state["legal_findings"] = result.content
    return state


# --- Agent 3: Risk Scoring Agent ---------------------------------------------
risk_prompt = ChatPromptTemplate.from_template(
    """You are a Risk Scoring Agent. Combine the two reports below into ONE final
due-diligence report for the buyer.

Document verification report:
{checklist_report}

Legal & title findings:
{legal_findings}

Produce:
1. Overall Risk Score: LOW / MEDIUM / HIGH (with one-line justification)
2. Top 5 red flags, ranked by severity
3. Recommended next actions before proceeding with purchase
4. A "Ready to Register?" YES/NO/CONDITIONAL verdict

Keep it tight and skimmable — this goes straight to a buyer."""
)


def risk_scoring_node(state: DueDiligenceState) -> DueDiligenceState:
    chain = risk_prompt | llm
    result = chain.invoke({
        "checklist_report": state["checklist_report"],
        "legal_findings": state["legal_findings"],
    })
    state["risk_report"] = result.content
    return state


# --- Orchestration graph ------------------------------------------------------
def build_graph():
    graph = StateGraph(DueDiligenceState)
    graph.add_node("document_verification", document_verification_node)
    graph.add_node("legal_title_check", legal_title_node)
    graph.add_node("risk_scoring", risk_scoring_node)

    graph.set_entry_point("document_verification")
    graph.add_edge("document_verification", "legal_title_check")
    graph.add_edge("legal_title_check", "risk_scoring")
    graph.add_edge("risk_scoring", END)

    return graph.compile()


def run_due_diligence(doc_dir: str, property_desc: str) -> dict:
    doc_texts = load_documents(doc_dir)
    build_vectorstore(doc_texts)  # (re)build index for this property's docs

    app_graph = build_graph()
    final_state = app_graph.invoke({
        "property_desc": property_desc,
        "doc_texts": doc_texts,
        "checklist_report": None,
        "legal_findings": None,
        "risk_report": None,
    })
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Folder of property documents")
    parser.add_argument("--property", required=True, help="Short property description")
    args = parser.parse_args()

    result = run_due_diligence(args.docs, args.property)
    print(json.dumps(result, indent=2, default=str))
