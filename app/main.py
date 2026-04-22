"""
main.py — FastAPI Application Layer (Production-Grade)
======================================================
WHY THIS FILE EXISTS:
  The API gateway for the Healthcare RAG system. Exposes REST endpoints
  that the frontend (or test_script.py) calls for document ingestion,
  query processing, and system monitoring.

CHANGES MADE vs ORIGINAL:

  1. Request Logging Middleware (MLOps):
     Every HTTP request is logged with method, path, response status, and
     latency (ms). Standard practice in production services — enables analysis
     of slow endpoints and error rates without touching business logic.

  2. File Size Validation at API layer:
     Rejects files > 50 MB early (before even saving to disk).
     Works alongside the same guard in processor.py for defence-in-depth.

  3. Richer /upload response:
     Now returns pages_processed, total_chars, tokens_estimated, and
     ingestion_time_sec so the caller can monitor document quality.

  4. Input Validation for /query:
     Rejects empty question strings with HTTP 400 (was previously a vague
     LLM error which was hard to debug).

  5. /stats endpoint (NEW — MLOps monitoring):
     Returns upload_count, index_size_mb, index_exists, log_file path.
     Designed to be polled by a monitoring dashboard (Grafana, etc.).

  6. /delete-document endpoint (NEW):
     Removes a file from the uploads/ directory. Logs a warning reminding
     operators that FAISS doesn't support selective deletion — a re-index
     endpoint (future work) would be needed for full cleanup.

  7. tokens_used split into tokens_input + tokens_output:
     Matches rag_chain.py's new response dict. Both values returned to caller
     for cost tracking in production.

STRICTLY PRESERVED:
  - /upload endpoint with OCR + Structured data support ✅
  - /list-files endpoint ✅
  - /query endpoint with Toggle (target_file) filtering ✅
  - FlashRank Reranking (via rag.get_response) ✅
  - QueryRequest / QueryResponse Pydantic models ✅
"""

import os
import time
import shutil
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, validator
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

from app.processor import DocumentProcessor, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB
from app.rag_chain import RAGChain, prompt_cache
from app.utils import get_logger, ensure_dirs, get_system_stats, token_budget
from app.services.table_extractor import convert_pdf_to_excel, validate_with_llm

logger = get_logger(__name__)

# ── APP INITIALISATION ────────────────────────────────────────────────────────
app = FastAPI(
    title="Advanced Healthcare RAG System",
    description=(
        "Production-ready Retrieval-Augmented Generation system for the "
        "healthcare sector. Supports PDF (OCR + Digital), CSV, and Excel "
        "ingestion with FAISS vector search, FlashRank reranking, and Llama-3 via Groq."
    ),
    version="3.0.0"  # Bumped: reflects the Small-to-Big + Graph-RAG upgrade
)

# ── REQUEST LOGGING MIDDLEWARE ────────────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    WHY STARLETTE MIDDLEWARE:
      Intercepts every request/response at the ASGI layer.
      Logs: method, path, status code, and latency.
      This is the standard MLOps pattern for REST API observability —
      gives us data to build dashboards on slow or failing endpoints.
    """
    async def dispatch(self, request: Request, call_next):
        t_start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[HTTP] {request.method} {request.url.path} "
            f"→ {response.status_code} | {elapsed_ms:.1f}ms"
        )
        return response

app.add_middleware(RequestLoggingMiddleware)

# ── MODULE INITIALISATION ─────────────────────────────────────────────────────
processor = DocumentProcessor()
rag = RAGChain()

UPLOAD_DIR  = "uploads"
INDEX_PATH  = "data/vector_store"
ensure_dirs([UPLOAD_DIR, "data", "logs"])

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    # target_file enables Toggle Search — query against a specific document
    target_file: Optional[str] = None

    @validator("question")
    def question_must_not_be_empty(cls, v):
        """
        WHY: An empty question causes a vague LLM error that's hard to debug.
        Catching it at the Pydantic layer returns HTTP 422 with a clear message.
        """
        if not v or not v.strip():
            raise ValueError("Question must not be empty.")
        return v.strip()


class QueryResponse(BaseModel):
    question:         str
    answer:           str
    tokens_input:     int   # Input token count (query + context)
    tokens_output:    int   # Output token count (generated answer)
    search_mode:      str
    confidence_level: str
    sources:          List[dict]   # Full metadata list for frontend citations
    # ── New fields ───────────────────────────────────────────────────────────────
    cache_hit:        bool         # True = served from in-memory cache (0 API calls)
    model_used:       Optional[str]  # Which Groq model was selected by ModelRouter
    eval_metrics:     Optional[dict]  # LLM-as-Judge scores (faithfulness, relevance, precision)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Handles PDF, CSV, and Excel uploads.
    Runs the Dual-Layer OCR (PyMuPDF + EasyOCR) or structured data parser,
    then indexes the result into FAISS with Graph-RAG metadata.

    CHANGES vs original:
      - File size validation before saving
      - Returns richer response: pages_processed, total_chars, tokens_estimated,
        ingestion_time_sec
      - Uses updated processor.extract_text() which returns List[dict] with metadata
    """
    allowed_extensions  = [".pdf", ".csv", ".xlsx", ".xls"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Allowed: {allowed_extensions}"
        )

    # Read file into memory to check size before saving to disk
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=(
                f"File '{file.filename}' is {size_mb:.1f} MB, "
                f"exceeding the {MAX_FILE_SIZE_MB} MB limit. "
                "Please split large documents before uploading."
            )
        )

    # Save to disk for record-keeping and potential re-indexing
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"[UPLOAD] Saved '{file.filename}' ({len(file_bytes)/1024:.1f} KB)")

    t_start = time.perf_counter()
    try:
        # Extract text with page-level metadata (List[dict] format)
        pages_data = processor.extract_text(file_path)

        # Index into FAISS with Graph-RAG metadata (Small-to-Big chunking)
        rag.create_vector_store(pages_data, file.filename)

        # ── Cache Invalidation: New document = stale answers ─────────────────
        # WHY: If a doctor uploads an updated protocol, old cached answers
        # for the same question would be wrong. Clear all cached responses.
        prompt_cache.clear()

        elapsed_sec = round(time.perf_counter() - t_start, 2)
        total_chars = sum(p["metadata"].get("char_count", 0) for p in pages_data)
        tokens_estimated = total_chars // 4  # ~4 chars per token (rough estimate)

        logger.info(
            f"[UPLOAD DONE] '{file.filename}' | "
            f"pages={len(pages_data)} | chars={total_chars} | "
            f"est_tokens={tokens_estimated} | time={elapsed_sec}s"
        )

        return {
            "message":           f"Successfully processed and indexed: {file.filename}",
            "file_type":         file_ext,
            "status":            "Success",
            "pages_processed":   len(pages_data),
            "total_chars":       total_chars,
            "tokens_estimated":  tokens_estimated,
            "ingestion_time_sec": elapsed_sec
        }

    except ValueError as ve:
        # Known, actionable errors from processor (empty file, bad format, etc.)
        logger.warning(f"[UPLOAD WARN] '{file.filename}': {ve}")
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as e:
        logger.error(f"[UPLOAD ERROR] '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/list-files")
async def list_files():
    """
    Returns the list of all indexed documents for the UI Toggle dropdown.
    UNCHANGED vs original.
    """
    if not os.path.exists(UPLOAD_DIR):
        return {"files": []}

    allowed = {".pdf", ".csv", ".xlsx", ".xls"}
    files = [
        f for f in os.listdir(UPLOAD_DIR)
        if os.path.splitext(f)[1].lower() in allowed
    ]
    return {"files": sorted(files)}


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Performs context-aware Q&A with Toggle filtering, FlashRank reranking,
    Small-to-Big context expansion, and Token Optimization.

    CHANGES vs original:
      - Returns tokens_input + tokens_output (was single tokens_used)
      - Returns sources list (metadata dicts) for frontend citation rendering
      - Checks for LLM error in result dict (vs just checking if result is str)
    """
    try:
        result = rag.get_response(request.question, request.target_file)

        # Detect error responses from rag_chain (e.g. empty DB, LLM timeout)
        if result.get("confidence") in ("N/A", "Error"):
            status = 404 if result["confidence"] == "N/A" else 503
            raise HTTPException(status_code=status, detail=result["answer"])

        mode = (
            f"Filtered Search → {request.target_file}"
            if request.target_file
            else "Global Hybrid Search (all documents)"
        )

        return {
            "question":         request.question,
            "answer":           result["answer"],
            "tokens_input":     result["tokens_input"],
            "tokens_output":    result["tokens_output"],
            "search_mode":      mode,
            "confidence_level": result["confidence"],
            "sources":          result["sources"],
            "cache_hit":        result.get("cache_hit", False),
            "model_used":       result.get("model_used"),
            "eval_metrics":     result.get("eval_metrics")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QUERY ERROR] Unexpected failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats")
async def system_stats():
    """
    MLOps monitoring endpoint.
    Returns FAISS index size, upload count, and log file path.
    Designed to be polled by monitoring dashboards (Grafana, custom UI, etc.).

    NEW endpoint — not in original.
    """
    stats = get_system_stats(UPLOAD_DIR, INDEX_PATH)
    # Augment with token budget + cache stats
    stats["token_budget"]  = token_budget.get_budget_summary()
    stats["prompt_cache"]  = prompt_cache.stats()
    logger.info(f"[STATS] Polled: {stats}")
    return stats


@app.delete("/delete-document")
async def delete_document(filename: str = Query(..., description="Exact filename to delete")):
    """
    Removes an uploaded file from disk.
    NEW endpoint — not in original.

    NOTE: FAISS does not natively support selective document deletion.
    To fully remove a document from the vector index, the index must be
    rebuilt (future work: POST /rebuild-index endpoint).
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found in uploads directory."
        )
    try:
        os.remove(file_path)
        logger.warning(
            f"[DELETE] '{filename}' removed from uploads/. "
            "Note: FAISS index still contains its vectors. "
            "Run /rebuild-index to fully remove from search."
        )
        return {
            "deleted": True,
            "filename": filename,
            "warning": (
                "File deleted from disk. FAISS index still contains vectors "
                "for this document. A full re-index is required to remove it "
                "from search results."
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


def remove_file(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {path}: {e}")


@app.post("/extract-excel")
async def extract_to_excel(
    file: UploadFile = File(...),
    use_llm_validation: bool = Query(False, description="Use Llama-3 to structure and validate raw OCR tables into clean columns.")
):
    """
    Extracts tabular data from a PDF (digital or scanned) and returns an Excel file.
    Bypasses the FAISS semantic search pipeline.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != ".pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported for table extraction. Got: '{file_ext}'"
        )

    # Read file to check size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is {size_mb:.1f} MB, exceeding {MAX_FILE_SIZE_MB} MB limit."
        )

    temp_id = str(uuid.uuid4())
    temp_pdf_path = f"/tmp/{temp_id}_{file.filename}"
    temp_excel_path = f"/tmp/{temp_id}_output.xlsx"

    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"[EXTRACT-EXCEL] Processing '{file.filename}' (LLM Validation: {use_llm_validation})")

    t_start = time.perf_counter()
    try:
        if use_llm_validation:
            validate_with_llm(temp_pdf_path, temp_excel_path)
        else:
            convert_pdf_to_excel(temp_pdf_path, temp_excel_path)
    except ValueError as ve:
        remove_file(temp_pdf_path)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        remove_file(temp_pdf_path)
        logger.error(f"[EXTRACT ERROR] '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    elapsed_sec = time.perf_counter() - t_start
    logger.info(f"[EXTRACT-EXCEL DONE] '{file.filename}' processed in {elapsed_sec:.1f}s.")

    # Clean up the PDF immediately
    remove_file(temp_pdf_path)

    # Return Excel and clean it up as a background task
    tasks = BackgroundTasks()
    tasks.add_task(remove_file, temp_excel_path)

    return FileResponse(
        path=temp_excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"Extracted_{file.filename.replace('.pdf', '')}.xlsx",
        background=tasks
    )


@app.get("/")
async def health_check():
    """
    Health check endpoint. UNCHANGED vs original, enhanced with version.
    """
    return {
        "status":   "Online",
        "system":   "Healthcare Intelligence",
        "version":  "4.0.0",
        "features": [
            "Dual-Layer-OCR",
            "Structured-Data-Support",
            "FlashRank-Reranking",
            "Toggle-Search",
            "Small-to-Big-Chunking",
            "Graph-RAG-Metadata",
            "Token-Optimization",
            "MLOps-Logging",
            "Prompt-Caching-SHA256-TTL",
            "Multi-Factor-Model-Router",
            "LLM-as-Judge-Evaluation",
            "Token-Budget-Tracker"
        ]
    }