"""
utils.py — Logging Foundation & Shared Utilities
=================================================
WHY THIS FILE EXISTS:
  This module is the single source of truth for logging configuration across the
  entire Healthcare RAG system. Centralising it here means every other module
  (processor.py, rag_chain.py, main.py) gets the same log format, same file
  handler, and same rotation policy — no duplicated basicConfig calls.

CHANGES MADE vs ORIGINAL:
  1. RotatingFileHandler  — Prevents log files from growing unbounded in production.
     Max 5 MB per file, 3 backup files kept (logs/*.1, *.2, *.3).
  2. @log_performance decorator — Wraps any function to emit execution time + pass/fail
     status automatically. Used in processor.py and rag_chain.py for MLOps latency
     tracking without modifying business logic.
  3. get_system_stats() — Provides a JSON-ready dict of FAISS index size on disk +
     upload directory document count. Powers the new /stats endpoint in main.py.
  4. format_citations() — Enhanced to include page numbers if available in metadata.
  5. TokenBudgetTracker (NEW) — Session-level token usage and cost tracking.
     Records per-query token counts, model used, and cache hits. Computes
     estimated USD cost using Groq API pricing for 8B and 70B models.
     Powers the token_budget section of the /stats endpoint.

STRICTLY PRESERVED:
  - ensure_dirs()  ✅
  - clear_uploads() ✅
  - format_citations() ✅ (signature unchanged, output enriched)
"""

import os
import shutil
import logging
import time
import functools
from logging.handlers import RotatingFileHandler

# ── DIRECTORY SETUP ──────────────────────────────────────────────────────────
# Ensure the logs/ directory exists BEFORE we configure the handler.
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# ── LOGGING CONFIGURATION ─────────────────────────────────────────────────────
# WHY RotatingFileHandler:
#   In production (servers), a plain FileHandler would fill the disk over weeks.
#   Rotating at 5 MB with 3 backups keeps total log footprint under 20 MB.
def get_logger(name: str) -> logging.Logger:
    """
    Factory that returns a named logger already wired to:
      - StreamHandler (console) for dev visibility
      - RotatingFileHandler (logs/healthcare_rag.log) for production persistence
    
    Call this at the top of each module:
        from app.utils import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger was already configured
    # (happens when FastAPI reloads on --reload mode)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Shared formatter: timestamp | level | module | message
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console handler — visible in terminal / docker logs
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 2. Rotating file handler — persisted to disk
    #    maxBytes=5 MB, backupCount=3 → max 20 MB total log footprint
    fh = RotatingFileHandler(
        os.path.join(LOGS_DIR, "healthcare_rag.log"),
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# Module-level logger for utils itself
logger = get_logger(__name__)


# ── PERFORMANCE DECORATOR ─────────────────────────────────────────────────────
def log_performance(func):
    """
    Decorator — automatically logs:
      - Function entry with arguments (file/query name)
      - Execution time in milliseconds on success
      - Exception type and message on failure (then re-raises)

    WHY A DECORATOR:
      Keeps business logic in processor.py / rag_chain.py clean.
      MLOps best practice: latency data in logs makes it easy to build dashboards
      (e.g. Grafana + Loki or ELK stack) without changing application code.

    USAGE:
        @log_performance
        def extract_text(self, file_path: str) -> str:
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fn_name = func.__qualname__   # e.g. "DocumentProcessor.extract_text"
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(f"[PERF] {fn_name} completed in {elapsed_ms:.1f} ms")
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"[PERF] {fn_name} FAILED after {elapsed_ms:.1f} ms — "
                f"{type(exc).__name__}: {exc}"
            )
            raise   # Re-raise so callers still get the exception
    return wrapper


# ── DIRECTORY HELPERS (UNCHANGED) ─────────────────────────────────────────────
def ensure_dirs(dirs: list):
    """
    Ensures that required directories exist.
    Called at startup in main.py for 'uploads/' and 'data/vector_store/'.
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")


def clear_uploads(upload_dir: str):
    """
    Optional: Deletes all files in the upload directory.
    Useful for testing or a hard-reset endpoint.
    """
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")


# ── CITATION HELPER (ENHANCED) ────────────────────────────────────────────────
def format_citations(metadata: list) -> str:
    """
    Helper to format source citations for the UI.

    ENHANCED vs original:
      Now includes page number if available in chunk metadata so the doctor/user
      can jump directly to the source page in the original document.
      Fallback to original behaviour if 'page' key is absent.
    """
    citations = []
    seen = set()

    for m in metadata:
        source = m.get("source", "Unknown Document")
        page   = m.get("page", None)
        doc_type = m.get("doc_type", "")

        # Build citation string
        if page is not None:
            citation = f"Source: {source} (Page {page})"
        else:
            citation = f"Source: {source}"

        if doc_type:
            citation += f" [{doc_type.upper()}]"

        # Deduplicate identical citations
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)

    return "\n".join(citations) if citations else "No source available."


# ── SYSTEM STATS HELPER ───────────────────────────────────────────────────────
def get_system_stats(upload_dir: str, index_path: str) -> dict:
    """
    Returns a JSON-ready dict for the /stats monitoring endpoint.

    WHY:
      MLOps dashboards need a lightweight way to poll system health without
      querying the vector store or LLM. This reads only the filesystem.

    Returns:
        {
            "upload_count": int,           -- files in uploads/
            "index_size_mb": float,        -- FAISS index folder size on disk
            "index_exists": bool,          -- False if no documents indexed yet
            "log_file": str                -- path to current log file
        }
    """
    # Count uploaded files
    upload_count = 0
    if os.path.exists(upload_dir):
        upload_count = len([
            f for f in os.listdir(upload_dir)
            if os.path.isfile(os.path.join(upload_dir, f))
        ])

    # FAISS index size on disk
    index_size_mb = 0.0
    index_exists = os.path.exists(index_path)
    if index_exists:
        for dirpath, _, filenames in os.walk(index_path):
            for fname in filenames:
                fsize = os.path.getsize(os.path.join(dirpath, fname))
                index_size_mb += fsize / (1024 * 1024)
        index_size_mb = round(index_size_mb, 3)

    return {
        "upload_count": upload_count,
        "index_size_mb": index_size_mb,
        "index_exists": index_exists,
        "log_file": os.path.join(LOGS_DIR, "healthcare_rag.log")
    }


# ── TOKEN BUDGET TRACKER ────────────────────────────────────────────────────────────
# Groq API pricing (as of mid-2025, per 1M tokens, input+output combined)
# Source: console.groq.com/docs/openai
# WHY track costs: JD explicitly mentions "cost & latency optimisation".
# Having per-session USD estimates is a concrete, measurable outcome.
GROQ_PRICE_PER_1M = {
    "llama-3-8b-8192":         0.05,   # $0.05 per 1M tokens
    "llama-3.3-70b-versatile": 0.59,   # $0.59 per 1M tokens
}


class TokenBudgetTracker:
    """
    Session-level token usage and cost tracker.

    WHY session-level (not persistent):
      Persisting to DB/file adds complexity not justified for a demo system.
      Session stats reset on server restart but clearly show the concept:
      tracking + optimising token spend — which is what the JD asks for.

    WHY estimated cost:
      Groq's API is billed per token. Knowing which queries are expensive helps
      teams decide when to upgrade models or optimise prompts. This data feeds
      the /stats endpoint for dashboard display.
    """

    def __init__(self):
        self._lock = __import__("threading").Lock()
        self._reset()

    def _reset(self):
        """Internal state initialiser — called at startup."""
        self.total_queries      = 0
        self.total_input_tokens = 0
        self.total_output_tokens= 0
        self.cache_hits         = 0
        self.model_8b_calls     = 0
        self.model_70b_calls    = 0
        self.estimated_cost_usd = 0.0

    def record_query(
        self,
        tokens_input:  int,
        tokens_output: int,
        model_used:    str,
        cache_hit:     bool = False
    ):
        """
        Records stats for a single query execution.

        PARAMETERS:
          tokens_input  : Input token count (query + context)
          tokens_output : Output token count (generated answer)
          model_used    : Model name string (MODEL_LIGHT or MODEL_HEAVY)
          cache_hit     : True if served from cache (no API call made)
        """
        with self._lock:
            self.total_queries += 1

            if cache_hit:
                # Cache hits don't consume tokens or incur API cost
                self.cache_hits += 1
                return

            # Track token consumption
            self.total_input_tokens  += tokens_input
            self.total_output_tokens += tokens_output

            # Track model usage
            if "8b" in model_used.lower():
                self.model_8b_calls += 1
            else:
                self.model_70b_calls += 1

            # Estimate cost in USD
            price_per_1m = GROQ_PRICE_PER_1M.get(model_used, 0.59)  # Default to 70B price
            total_tokens = tokens_input + tokens_output
            self.estimated_cost_usd += round(
                (total_tokens / 1_000_000) * price_per_1m, 6
            )

    def get_budget_summary(self) -> dict:
        """
        Returns a JSON-ready summary of session token usage and estimated cost.
        Powers the token_budget section of the /stats endpoint.
        """
        with self._lock:
            cache_rate = (
                round(self.cache_hits / self.total_queries * 100, 1)
                if self.total_queries > 0 else 0.0
            )
            api_calls = self.model_8b_calls + self.model_70b_calls
            return {
                "total_queries":       self.total_queries,
                "api_calls_made":      api_calls,             # Excludes cache hits
                "cache_hits":          self.cache_hits,
                "cache_hit_rate_pct":  cache_rate,
                "total_input_tokens":  self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "model_8b_calls":      self.model_8b_calls,
                "model_70b_calls":     self.model_70b_calls,
                "estimated_cost_usd":  round(self.estimated_cost_usd, 6)
            }


# ── MODULE-LEVEL SINGLETON ────────────────────────────────────────────────────
# WHY module-level: utils.py is imported by both rag_chain.py and main.py.
# Module-level singleton ensures they all share the SAME tracker instance,
# giving a unified view of all session stats without any coordination code.
token_budget = TokenBudgetTracker()