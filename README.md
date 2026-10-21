# Real Estate Due-Diligence Copilot

OCR → RAG → 3-agent LangGraph system → FastAPI → AWS. Buyer uploads property
documents (sale deed, EC, tax receipts, khata, approvals, sketch); the system
verifies completeness, checks legal/title risk against the docs via RAG, and
returns a scored due-diligence report.

## Architecture

```
Upload docs → OCR (PyMuPDF + Tesseract fallback)
            → Chunk + Embed → FAISS vector store
            → LangGraph pipeline:
                 1. Document Verification Agent  (checklist: PRESENT/MISSING/UNCLEAR)
                 2. Legal & Title Agent           (RAG-grounded: title chain, encumbrance, litigation)
                 3. Risk Scoring Agent            (LOW/MEDIUM/HIGH + red flags + verdict)
            → FastAPI serves /upload, /analyze, /report
```

## Local setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# system dependency for OCR
sudo apt-get install tesseract-ocr   # macOS: brew install tesseract

export OPENAI_API_KEY=sk-...

# CLI run
python pipeline.py --docs sample_docs/ --property "Plot 14, Balanagar, Hyderabad"

# API run
uvicorn app:app --reload
```

## API usage

```bash
curl -X POST localhost:8000/upload \
  -F "property_desc=Plot 14, Balanagar, Hyderabad" \
  -F "files=@sample_docs/sale_deed.pdf" \
  -F "files=@sample_docs/encumbrance_certificate.pdf"
# -> {"job_id": "a1b2c3d4", ...}

curl -X POST localhost:8000/analyze/a1b2c3d4
curl localhost:8000/report/a1b2c3d4
```

## AWS deployment

**Option A — ECS Fargate (recommended for this capstone, no server management)**

1. Build & push image to ECR:
   ```bash
   aws ecr create-repository --repository-name re-due-diligence
   aws ecr get-login-password | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
   docker build -t re-due-diligence .
   docker tag re-due-diligence:latest <account_id>.dkr.ecr.<region>.amazonaws.com/re-due-diligence:latest
   docker push <account_id>.dkr.ecr.<region>.amazonaws.com/re-due-diligence:latest
   ```
2. Store `OPENAI_API_KEY` in **AWS Secrets Manager**; reference it in the ECS
   task definition's `secrets` block (never bake keys into the image).
3. Create an ECS Fargate service (1 task, 1 vCPU / 2GB is enough for a demo)
   behind an **Application Load Balancer**, target group health check on
   `GET /health`.
4. Swap the in-memory `JOB_STATUS` dict in `app.py` for **DynamoDB** (one
   table, `job_id` partition key) so state survives task restarts/scaling.
5. Move the FAISS index to **S3** (persist per job_id under
   `s3://<bucket>/indexes/<job_id>/`) instead of local disk, since ECS Fargate
   storage is ephemeral.
6. Point a Route 53 record / API Gateway HTTP API at the ALB for a stable
   public endpoint.

**Option B — EC2 (simpler, cheaper for a single-instance demo)**

1. Launch a `t3.medium` (Ubuntu 22.04), open port 8000 in the security group.
2. `git clone` the repo, `docker build` + `docker run -p 8000:8000 --env-file .env`.
3. Put an Nginx reverse proxy + Let's Encrypt in front for HTTPS, or use an
   ALB in front of the single instance if you want managed TLS.

**Option C — Lambda (cheapest at low volume)**

Package with **Mangum** (`pip install mangum`, wrap `app` for the Lambda
handler) behind API Gateway. Note: Lambda's 15-min timeout and cold starts
make this a poor fit once documents get large or OCR runs long — Fargate is
the better default for this workload.

## Production hardening notes (for the demo/interview, mention these even if unbuilt)

- Swap in-memory job store → DynamoDB or RDS
- Swap local FAISS → S3-backed FAISS or a managed vector DB (Pinecone/OpenSearch)
  if concurrent multi-tenant use is expected
- Add auth (API key or Cognito) before exposing `/upload` and `/analyze` publicly
- Rate-limit and validate file types/sizes before OCR (cost + abuse control)
- Log agent reasoning traces (LangSmith) for audit trail — important for a
  due-diligence product where buyers need to trust the verdict
