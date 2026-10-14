# OmniBrain — Agentic Multi-Modal RAG Orchestrator

OmniBrain is a Python document-intelligence prototype with a Streamlit interface and a FastAPI backend. Upload PDF documents, ask questions grounded in extracted text, inspect page-level citations, and review basic retrieval/guardrail metadata.

> **Current scope:** the supplied implementation extracts PDF text and embedded images, performs text-chunk retrieval, and detects available images. It does **not** yet interpret chart pixels with a vision-language model, use a persistent vector database, or execute generated SQL against a stock database. The SQL agent currently performs limited calculations only when structured rows are supplied in code.

## Features

- PDF upload and page-aware text chunking
- Optional semantic embeddings with a keyword-vector fallback
- Query routing for retrieval, basic numeric analysis, and image checks
- Retrieval retry when evidence scores are weak
- Source citations with document name and PDF page
- FastAPI endpoints for health, uploads, queries, documents, history, and stats
- Streamlit UI for chat, document management, and statistics

## Architecture

```text
Browser
  └── Streamlit UI (omnibrain_frontend.py)
        └── HTTP requests
              └── FastAPI (omnibrain_backend.py)
                    └── OmniBrainOrchestrator (omnibrain_core.py)
                          ├── SupervisorAgent
                          ├── VectorSearchAgent + EmbeddingService
                          ├── SQLQueryAgent (limited calculations)
                          ├── VisionLanguageAgent (image detection only)
                          ├── SelfCorrectionLoop
                          ├── SynthesisAgent
                          └── GuardrailsEvaluator
```

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Optional internet access when installing packages or downloading the sentence-transformer model

## Quick start (Windows PowerShell)

```powershell
# 1. Open this project folder in VS Code, then create a virtual environment
py -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, use this for the current terminal session:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Terminal 1: start API
python omnibrain_backend.py

# 5. Terminal 2: start UI
streamlit run omnibrain_frontend.py
```

Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`). The API docs are at `http://127.0.0.1:8000/docs`, and health is at `http://127.0.0.1:8000/health`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python omnibrain_backend.py
# In a second terminal:
streamlit run omnibrain_frontend.py
```

## Environment configuration

Copy `.env.example` to `.env` if you later add API credentials or other configuration. **The current code does not require an OpenAI API key**; its current response generation is evidence-based text assembly, not an OpenAI chat completion. Never commit `.env`, API keys, passwords, or private PDFs.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health and capability status |
| POST | `/upload` | Upload and process one PDF |
| POST | `/query` | Query indexed document chunks |
| GET | `/documents` | List indexed documents |
| GET | `/documents/{document_id}` | Document metadata |
| GET | `/query-history` | Recent query history (in-memory) |
| GET | `/stats` | Document/chunk/image/query counts |

Example query body:

```json
{
  "query": "Summarize the main findings",
  "document_ids": [],
  "include_reasoning": false,
  "confidence_threshold": 0.5
}
```

`include_reasoning` exposes a short execution trace of observable agent actions; it is not a hidden chain-of-thought transcript.

## Notes and limitations

- The document store and query history are in memory. Restarting the backend clears them.
- Uploaded PDFs and extracted images are written to `data/uploads/`; these files are ignored by Git.
- Scanned PDFs may need OCR. The current text extraction can return few or no chunks if a PDF has only scanned pages.
- Image extraction is best-effort and does not mean chart values have been understood.
- Similarity/confidence values are heuristics, not calibrated probabilities or guarantees against hallucination.
- Do not use this prototype as the sole basis for financial, legal, medical, or other high-impact decisions.

## Troubleshooting

**Frontend says backend offline**
1. Start `python omnibrain_backend.py` in one terminal.
2. Confirm `http://127.0.0.1:8000/health` opens.
3. In Streamlit's sidebar, set Backend URL to `http://127.0.0.1:8000`.

**Missing module**
```bash
python -m pip install -r requirements.txt
```
Ensure VS Code is using the same Python interpreter where you installed packages.

**Port 8000 is busy**
Stop the process using that port, or change the port in the `uvicorn.run(...)` block and update the frontend Backend URL.

## Roadmap

- Persistent vector storage (FAISS or Qdrant)
- OCR support for scanned PDFs
- Real VLM-based chart/table interpretation
- Structured data ingestion and safe SQL execution over an explicit database
- Better tests, persistent metadata, and authentication for multi-user deployment
- Evaluation datasets and observability

## License

Choose a license before publishing publicly. If you select MIT, add the standard MIT `LICENSE` file and replace the copyright placeholder with your name or organization.
