"""
FastAPI service for the Real Estate Due-Diligence Copilot.

Endpoints:
  POST /upload       -> upload property documents for a job_id
  POST /analyze/{id}  -> run the OCR -> RAG -> multi-agent pipeline
  GET  /report/{id}   -> fetch the final due-diligence report
  GET  /health        -> liveness check (for AWS ALB / ECS health checks)
"""

import os
import shutil
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pipeline import run_due_diligence

app = FastAPI(
    title="Real Estate Due-Diligence Copilot",
    description="OCR + RAG + multi-agent property document verification",
    version="1.0.0",
)

JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# in-memory job store (swap for DynamoDB/RDS in production — see README)
JOB_STATUS: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_docs(
    property_desc: str = Form(...),
    files: list[UploadFile] = File(...),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    for f in files:
        dest = os.path.join(job_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

    JOB_STATUS[job_id] = {"status": "uploaded", "property_desc": property_desc}
    return {"job_id": job_id, "files_received": len(files), "status": "uploaded"}


@app.post("/analyze/{job_id}")
def analyze(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(404, "job_id not found — upload documents first")

    job_dir = os.path.join(JOBS_DIR, job_id)
    property_desc = JOB_STATUS[job_id]["property_desc"]

    JOB_STATUS[job_id]["status"] = "processing"
    try:
        result = run_due_diligence(job_dir, property_desc)
        JOB_STATUS[job_id]["status"] = "done"
        JOB_STATUS[job_id]["result"] = {
            "checklist_report": result["checklist_report"],
            "legal_findings": result["legal_findings"],
            "risk_report": result["risk_report"],
        }
        return JOB_STATUS[job_id]["result"]
    except Exception as e:
        JOB_STATUS[job_id]["status"] = "failed"
        raise HTTPException(500, f"Pipeline failed: {e}")


@app.get("/report/{job_id}")
def get_report(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(404, "job_id not found")
    if JOB_STATUS[job_id]["status"] != "done":
        return {"status": JOB_STATUS[job_id]["status"]}
    return JSONResponse(JOB_STATUS[job_id]["result"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
