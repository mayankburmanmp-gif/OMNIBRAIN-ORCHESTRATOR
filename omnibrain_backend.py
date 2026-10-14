"""
OmniBrain FastAPI backend.

Run with:
    python omnibrain_backend.py
"""

import hashlib
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from omnibrain_core import OmniBrainOrchestrator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnibrain")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="OmniBrain - Agentic Multi-Modal RAG API",
    description="Local document intelligence backend.",
    version="2.0.0",
)

orchestrator = OmniBrainOrchestrator()

# Simple in-memory storage for this local project.
DOCUMENT_STORE: Dict[str, Dict[str, Any]] = {}
QUERY_HISTORY: List[Dict[str, Any]] = []


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    document_ids: List[str] = Field(default_factory=list)
    include_reasoning: bool = False
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Citation(BaseModel):
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    similarity: float = 0.0
    content: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    guardrails: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float
    routes: List[str] = Field(default_factory=list)
    semantic_embeddings: bool = False


def make_document_id(filename: str, content: bytes) -> str:
    """Create a short ID using the filename and file content."""
    digest = hashlib.sha256(content).hexdigest()[:12]
    return f"{Path(filename).stem[:30]}-{digest}"


def extract_pdf(
    content: bytes,
    filename: str,
    document_id: str,
) -> tuple[list[dict], list[dict]]:
    """Extract page text and embedded images from a PDF."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="pypdf is not installed. Run: pip install pypdf",
        ) from exc

    reader = PdfReader(io.BytesIO(content))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        words = text.split()
        chunk_size = 180
        overlap = 30
        start = 0
        chunk_number = 0

        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append({
                    "document_id": document_id,
                    "document_name": filename,
                    "page": page_number,
                    "chunk_id": (
                        f"{document_id}-p{page_number}-c{chunk_number}"
                    ),
                    "text": chunk_text,
                })

            if end >= len(words):
                break

            start = max(start + 1, end - overlap)
            chunk_number += 1

    images = []

    try:
        import fitz

        pdf = fitz.open(stream=content, filetype="pdf")
        image_folder = UPLOAD_DIR / document_id / "images"
        image_folder.mkdir(parents=True, exist_ok=True)

        for page_index in range(len(pdf)):
            page = pdf[page_index]

            for image_index, image in enumerate(
                page.get_images(full=True)
            ):
                xref = image[0]
                image_data = pdf.extract_image(xref)

                extension = image_data.get("ext", "png")
                image_path = (
                    image_folder
                    / f"page_{page_index + 1}_image_{image_index + 1}.{extension}"
                )

                image_path.write_bytes(image_data["image"])

                images.append({
                    "document_id": document_id,
                    "page": page_index + 1,
                    "image_index": image_index + 1,
                    "image_path": str(image_path),
                })

        pdf.close()

    except Exception as exc:
        logger.warning("Image extraction skipped: %s", exc)

    return chunks, images


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
        "agents": [
            "SupervisorAgent",
            "VectorSearchAgent",
            "SQLQueryAgent",
            "VisionLanguageAgent",
            "SynthesisAgent",
            "GuardrailsEvaluator",
        ],
        "semantic_embeddings": (
            orchestrator.embedding_service.semantic_available
        ),
        "documents_loaded": len(DOCUMENT_STORE),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF is larger than the 100 MB limit.",
        )

    document_id = make_document_id(file.filename, content)
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    pdf_path.write_bytes(content)

    start_time = time.perf_counter()
    chunks, images = extract_pdf(
        content,
        file.filename,
        document_id,
    )

    page_numbers = {chunk["page"] for chunk in chunks}

    DOCUMENT_STORE[document_id] = {
        "document_id": document_id,
        "filename": file.filename,
        "path": str(pdf_path),
        "size_bytes": len(content),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(page_numbers),
        "chunk_count": len(chunks),
        "image_count": len(images),
        "chunks": chunks,
        "images": images,
        "status": "ready",
        "processing_time": round(time.perf_counter() - start_time, 3),
    }

    logger.info(
        "Processed %s: %s chunks, %s images",
        file.filename,
        len(chunks),
        len(images),
    )

    return {
        "success": True,
        "document_id": document_id,
        "filename": file.filename,
        "status": "ready",
        "pages": len(page_numbers),
        "chunks": len(chunks),
        "images": len(images),
        "processing_time": DOCUMENT_STORE[document_id]["processing_time"],
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    if not DOCUMENT_STORE:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before querying.",
        )

    documents = list(DOCUMENT_STORE.values())

    if request.document_ids:
        unknown_ids = [
            document_id
            for document_id in request.document_ids
            if document_id not in DOCUMENT_STORE
        ]

        if unknown_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown document IDs: {unknown_ids}",
            )

    result = orchestrator.process_query(
        request.query,
        documents=documents,
        selected_document_ids=request.document_ids,
    )

    if result["confidence"] < request.confidence_threshold:
        result["guardrails"]["below_requested_confidence"] = True
        result["answer"] += (
            "\n\nConfidence warning: the retrieved evidence is below "
            "your selected threshold. Please verify the cited PDF pages."
        )
    else:
        result["guardrails"]["below_requested_confidence"] = False

    if not request.include_reasoning:
        # Only return simple observable actions.
        result["execution_trace"] = [
            {
                "step": item.get("step"),
                "action": item.get("action"),
            }
            for item in result["execution_trace"]
        ]

    QUERY_HISTORY.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "document_ids": request.document_ids,
        "confidence": result["confidence"],
        "processing_time": result["processing_time"],
        "routes": result["routes"],
    })

    return QueryResponse(**result)


@app.get("/documents")
def get_documents() -> List[Dict[str, Any]]:
    fields = (
        "document_id",
        "filename",
        "size_bytes",
        "uploaded_at",
        "page_count",
        "chunk_count",
        "image_count",
        "status",
        "processing_time",
    )

    return [
        {field: document.get(field) for field in fields}
        for document in DOCUMENT_STORE.values()
    ]


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> Dict[str, Any]:
    document = DOCUMENT_STORE.get(document_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    fields = (
        "document_id",
        "filename",
        "size_bytes",
        "uploaded_at",
        "page_count",
        "chunk_count",
        "image_count",
        "status",
        "processing_time",
    )

    return {
        field: document.get(field)
        for field in fields
    }


@app.get("/query-history")
def query_history() -> List[Dict[str, Any]]:
    return QUERY_HISTORY[-100:]


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return {
        "documents": len(DOCUMENT_STORE),
        "chunks": sum(
            document.get("chunk_count", 0)
            for document in DOCUMENT_STORE.values()
        ),
        "images": sum(
            document.get("image_count", 0)
            for document in DOCUMENT_STORE.values()
        ),
        "queries": len(QUERY_HISTORY),
        "semantic_embeddings": (
            orchestrator.embedding_service.semantic_available
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
