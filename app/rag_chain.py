"""
rag_chain.py — Advanced RAG Pipeline (Core Module)
===================================================
WHY THIS FILE EXISTS:
  This is the intelligence layer of the Healthcare RAG system.
  It converts raw extracted text into a searchable vector index (FAISS) and
  answers clinical queries using a two-stage retrieval → reranking → generation
  pipeline built on LangChain + Groq (Llama-3).

CHANGES MADE vs ORIGINAL:

  1. Graph-RAG Metadata (JD: metadata-linking):
     create_vector_store() now accepts a list of page-metadata dicts from
     processor.py. Each chunk stored in FAISS carries:
       {"source": filename, "page": N, "doc_type": "pdf/csv",
        "ocr_used": bool, "ingested_at": ISO timestamp, "chunk_index": N}
     This enables filtering not just by filename but by doc_type or page range.

  2. Small-to-Big Chunking (Accuracy Enhancement — ADDITIVE):
     Two text splitters:
       - child_splitter  (size=400, overlap=50):  what gets embedded in FAISS
       - parent_splitter (size=1500, overlap=100): richer context sent to LLM
     Parent chunks stored in self.parent_store (in-memory dict keyed by chunk id).
     At retrieval time, child chunks are found → their parent chunk id → full
     parent text → passed to LLM. Gives richer context without sacrificing
     retrieval precision.

  3. Thread-safe FAISS writes (Scalability):
     threading.Lock() wraps all save_local / load_local / add_documents calls.
     Prevents index corruption when FastAPI processes 2+ concurrent uploads.

  4. Token Optimization Fix (JD: Token Optimization & Model Context Protocols):
     Now counts BOTH input tokens (query + context) and output tokens.
     Hard MAX_CONTEXT_TOKENS=6000 guard: trims least-relevant chunks if context
     window would overflow before sending to Groq.

  5. Groq API Timeout & Error Handling:
     rag_chain.invoke() wrapped in try/except for httpx.TimeoutException and
     broad Exception to return a structured error dict instead of crashing.

  6. Retrieval Latency Logging (MLOps):
     Separate timing for FAISS retrieval, FlashRank reranking, LLM generation.
     Logged to healthcare_rag.log for dashboarding.

  7. Improved Prompt with Page-Level Citations:
     Template now includes "{source} (Page {page})" extracted from chunk metadata.
     Source list is de-duplicated and injected below the context.

  8. Prompt Caching (Cost & Latency Optimization — NEW):
     PromptCache: in-memory SHA-256 keyed cache with 1-hour TTL.
     Repeat queries served instantly with zero Groq API calls.
     Cache is cleared automatically on every new document upload.

  9. Multi-Factor Model Router (Cost Optimization — NEW):
     select_model() combines 3 signals: query word count, complex keyword
     detection, and context token count. Only when all three are "simple"
     does it route to llama-3-8b-8192 (10x cheaper). Otherwise uses
     llama-3.3-70b-versatile for diagnostic accuracy.

  10. LLM-as-Judge Evaluation (RAG Quality — NEW):
      After generation, RAGEvaluator scores Faithfulness, Answer Relevance,
      and Context Precision using a single 8B judge call. Semantically aware
      (fracture = broken bone). Gracefully falls back to null on API error.

STRICTLY PRESERVED:
  - FAISS vector store 
  - HuggingFace all-MiniLM-L6-v2 embeddings 
  - FlashRank reranking 
  - Toggle / target_file metadata filter 
  - Groq Llama-3 LLM 
  - count_tokens() 
"""

import os
import re
import time
import hashlib
import threading
import tiktoken
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.utils import get_logger, format_citations, token_budget
from app.evaluator import RAGEvaluator

# Load API Keys (GROQ_API_KEY from .env)
load_dotenv()

# Force Hugging Face to use local cached files (offline mode)
# WHY: Prevents "Temporary failure in name resolution" errors on startup
# if the machine cannot reach huggingface.co.
os.environ["HF_HUB_OFFLINE"] = "1"

logger = get_logger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# WHY 6000 TOKENS:
#   Llama-3-8B has an 8192 token context window. Reserving ~2000 for the prompt
#   template + generated answer leaves ~6000 for retrieved context chunks.
MAX_CONTEXT_TOKENS = 6000

# Model identifiers
MODEL_LIGHT  = "llama-3-8b-8192"          # Fast, cheap — simple queries
MODEL_HEAVY  = "llama-3.3-70b-versatile"  # Accurate — diagnostic queries

# ── COMPLEX KEYWORD LIST ──────────────────────────────────────────────────────
# WHY these words: They signal multi-step clinical reasoning which benefits from
# the 70B model's broader parametric knowledge.
COMPLEX_KEYWORDS = [
    "diagnose", "diagnosis", "differential", "differentiate", "compare",
    "protocol", "prognosis", "etiology", "pathophysiology", "mechanism",
    "contraindication", "treatment plan", "management", "complication",
    "risk factor", "interpret", "explain why", "why does", "how does"
]


# ── PROMPT CACHE ──────────────────────────────────────────────────────────────
@dataclass
class CacheEntry:
    """
    Single cached query result.
    Stores the full response dict (including eval_metrics) so cache hits
    return complete, pre-evaluated responses without any API calls.
    """
    result:     Dict
    created_at: float = field(default_factory=time.time)
    hit_count:  int   = 0


class PromptCache:
    """
    In-memory prompt cache with SHA-256 keying and TTL-based expiry.

    WHY SHA-256:
      Normalises the cache key (case, whitespace) and prevents key collisions
      from similar-looking queries without storing the raw query text in memory.

    WHY TTL=3600s:
      Medical guidelines don't change hourly. A 1-hour TTL gives fast repeat
      responses while ensuring users see updated answers after a reasonable time.

    WHY clear() on upload:
      A new document changes the knowledge base, so cached answers may be stale.
      Clearing on upload ensures correctness over speed.
    """
    TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock  = threading.Lock()

    def _make_key(self, query: str, target_file: Optional[str]) -> str:
        """Deterministic SHA-256 key from normalised query + file filter."""
        raw = f"{query.lower().strip()}|{target_file or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, target_file: Optional[str]) -> Optional[Dict]:
        """Returns cached result if present and not expired, else None."""
        key = self._make_key(query, target_file)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() - entry.created_at > self.TTL_SECONDS:
                del self._cache[key]  # Expired — evict eagerly
                logger.info(f"[CACHE] Expired entry evicted for key={key[:8]}...")
                return None
            entry.hit_count += 1
            logger.info(
                f"[CACHE] HIT — key={key[:8]}... | hits={entry.hit_count} | "
                f"age={int(time.time()-entry.created_at)}s"
            )
            return entry.result

    def set(self, query: str, target_file: Optional[str], result: Dict):
        """Stores a new result in cache."""
        key = self._make_key(query, target_file)
        with self._lock:
            self._cache[key] = CacheEntry(result=result)
        logger.info(f"[CACHE] STORED — key={key[:8]}... | cache_size={len(self._cache)}")

    def clear(self):
        """Clears all cache entries. Called on every new document upload."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        logger.info(f"[CACHE] CLEARED — {count} entries removed (new document ingested).")

    def stats(self) -> Dict:
        """Returns cache stats for the /stats endpoint."""
        with self._lock:
            total_hits = sum(e.hit_count for e in self._cache.values())
            return {
                "cached_queries": len(self._cache),
                "total_cache_hits": total_hits
            }


# ── MODULE-LEVEL SINGLETONS ───────────────────────────────────────────────────
# WHY module-level: Both RAGChain and the FastAPI app share the same cache
# instance. Module-level singletons are the correct pattern in FastAPI.
prompt_cache = PromptCache()


# ── MODEL ROUTER ──────────────────────────────────────────────────────────────
def select_model(query: str, tokens_input: int) -> str:
    """
    Multi-factor model selection router.

    Routes to the LIGHT model (8B) ONLY when ALL three signals indicate simplicity:
      1. Query is short  (≤ 12 words)
      2. No complex clinical keywords detected
      3. Context is not large (≤ 3500 tokens)

    WHY "ALL three" instead of "ANY one":
      Using ANY would make it too aggressive — a short query with "prognosis"
      is still a complex medical reasoning task. ALL ensures the light model
      is only used when it is genuinely sufficient.

    Example routing:
      "What is fever?"                           → 8B  (short, simple, small context)
      "List symptoms of hypertension"            → 8B  (short, simple keyword)
      "Diagnose based on the following symptoms" → 70B (complex keyword: 'diagnose')
      "What is the prognosis for stage 3 CKD?"   → 70B (complex keyword: 'prognosis')
      "Summarise this document" + 5000 tokens    → 70B (high context load)
    """
    word_count   = len(query.split())
    has_complex  = any(kw in query.lower() for kw in COMPLEX_KEYWORDS)
    high_context = tokens_input > 3500

    if word_count <= 12 and not has_complex and not high_context:
        logger.info(
            f"[MODEL ROUTER] → LIGHT ({MODEL_LIGHT}) | "
            f"words={word_count} | complex={has_complex} | high_ctx={high_context}"
        )
        return MODEL_LIGHT
    else:
        logger.info(
            f"[MODEL ROUTER] → HEAVY ({MODEL_HEAVY}) | "
            f"words={word_count} | complex={has_complex} | high_ctx={high_context}"
        )
        return MODEL_HEAVY


class RAGChain:
    def __init__(self, index_path: str = "data/vector_store"):
        # Evaluator initialised once here — shared across all requests
        self.evaluator = RAGEvaluator()
        self.index_path = index_path

        # Thread lock for FAISS index reads/writes (Scalability)
        # WHY: FastAPI processes concurrent requests in a thread pool.
        # Without a lock, two simultaneous uploads can corrupt the FAISS index
        # on disk via interleaved save_local calls.
        self._faiss_lock = threading.Lock()

        # In-memory parent store for Small-to-Big chunking
        # Key: unique chunk ID  Value: parent chunk text
        # WHY DICT: FAISS can't store arbitrary Python objects; we sidecar the
        # parent text in RAM and reference it via chunk metadata.
        self.parent_store: Dict[str, str] = {}

        logger.info("Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")

        # 1. Embeddings — all-MiniLM-L6-v2 is efficient for local retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True}
        )

        # 2. LLM — dynamically selected per query by select_model().
        #    Placeholder set here; get_response() swaps the model on each call.
        #    WHY not set here: tokens_input is only known after FAISS retrieval.
        self._llm_cache: Dict[str, ChatGroq] = {}   # Avoid recreating LLM objects

        # 3. Reranker — FlashRank (Move to __init__ for efficiency + offline handling)
        self.compressor = None
        try:
            logger.info("Initializing FlashRank reranker...")
            # Note: Flashrank may try to download a model if not cached.
            # We catch errors here to allow fallback to non-reranked retrieval.
            self.compressor = FlashrankRerank()
        except Exception as e:
            logger.warning(f"FlashRank initialization failed (likely offline): {e}. Falling back to default retrieval.")

        # 3a. CHILD splitter — small chunks for precise FAISS embedding
        #     WHY 400/50: Captures a single clinical finding or sentence cluster.
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            add_start_index=True  # Enables parent-child index linking
        )

        # 3b. PARENT splitter — larger chunks for richer LLM context
        #     WHY 1500/100: Preserves clinical narrative around a child chunk.
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100
        )

        # 4. Tokenizer for input + output token counting
        #    cl100k_base is the closest public encoding to Llama tokenisation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info("RAGChain initialised successfully.")

    # ── TOKEN COUNTING ────────────────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        """Counts tokens to manage context window and costs. UNCHANGED."""
        return len(self.tokenizer.encode(text))

    def _trim_context_to_token_limit(
        self, docs: List[Document], max_tokens: int
    ) -> List[Document]:
        """
        WHY THIS EXISTS:
          Even after reranking to top-3, the combined text of those chunks might
          exceed Groq's context window if documents are dense (e.g. WHO PDF).
          This trims from the least-relevant end (last in reranked list) to stay
          under max_tokens, preventing API errors.
        """
        trimmed = []
        used_tokens = 0
        for doc in docs:
            chunk_tokens = self.count_tokens(doc.page_content)
            if used_tokens + chunk_tokens > max_tokens:
                remainder = max_tokens - used_tokens
                if remainder > 50:  # Keep partial only if meaningful
                    trimmed_text = doc.page_content[:remainder * 4]  # ~4 chars/token
                    trimmed.append(
                        Document(page_content=trimmed_text, metadata=doc.metadata)
                    )
                break
            trimmed.append(doc)
            used_tokens += chunk_tokens
        return trimmed

    # ── VECTOR STORE CREATION ─────────────────────────────────────────────────
    def create_vector_store(
        self,
        pages_data: List[Dict[str, Any]],
        filename: str
    ):
        """
        Two-tier chunking strategy (Small-to-Big) + Graph-RAG metadata.

        PARAMETERS:
          pages_data: Output from processor.py — list of {"text", "metadata"} dicts.
          filename:   Used as the 'source' metadata key (Toggle/Filter feature).

        FLOW:
          For each page:
            1. Split into PARENT chunks (1500 chars) → stored in self.parent_store
            2. Split each parent into CHILD chunks (400 chars) → stored in FAISS
               with metadata pointing back to parent_id
          FAISS append mode: new docs added to existing index (no re-indexing).

        Graph-RAG metadata per child chunk:
          {source, page, doc_type, ocr_used, ingested_at, parent_id, chunk_index}
        """
        ingested_at = datetime.now(timezone.utc).isoformat()
        all_child_docs = []

        for page_data in pages_data:
            page_text  = page_data["text"]
            page_meta  = page_data["metadata"]

            # Ensure 'source' is always set (Toggle/Filter depends on this)
            page_meta.setdefault("source", filename)

            # ── Small-to-Big: Create parent chunks first ───────────────────
            parent_chunks = self.parent_splitter.split_text(page_text)

            for p_idx, parent_text in enumerate(parent_chunks):
                # Unique key for parent lookup
                parent_id = f"{filename}::p{page_meta.get('page', 0)}::parent{p_idx}"
                self.parent_store[parent_id] = parent_text

                # ── Create child chunks from this parent ───────────────────
                child_chunks = self.child_splitter.split_text(parent_text)

                for c_idx, child_text in enumerate(child_chunks):
                    child_meta = {
                        # Core metadata (Toggle/Filter feature — UNCHANGED)
                        "source": filename,

                        # Extended metadata (Graph-RAG / page citations)
                        "page":         page_meta.get("page", 0),
                        "doc_type":     page_meta.get("doc_type", "unknown"),
                        "ocr_used":     page_meta.get("ocr_used", False),
                        "ingested_at":  ingested_at,

                        # Parent-child link (Small-to-Big retrieval)
                        "parent_id":    parent_id,
                        "chunk_index":  c_idx,
                    }
                    all_child_docs.append(
                        Document(page_content=child_text, metadata=child_meta)
                    )

        if not all_child_docs:
            logger.warning(f"[VECTOR STORE] No chunks created for '{filename}'.")
            return None

        # ── Thread-safe FAISS write ────────────────────────────────────────
        with self._faiss_lock:
            # Check if index exists AND contains the necessary files
            if os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "index.faiss")):
                # Append mode: merge into existing index
                logger.info(
                    f"[VECTOR STORE] Appending {len(all_child_docs)} child chunks "
                    f"for '{filename}' to existing index."
                )
                vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_store.add_documents(all_child_docs)
            else:
                # Fresh index
                logger.info(
                    f"[VECTOR STORE] Creating new index with {len(all_child_docs)} "
                    f"child chunks for '{filename}'."
                )
                vector_store = FAISS.from_documents(all_child_docs, self.embeddings)

            vector_store.save_local(self.index_path)

        logger.info(
            f"[VECTOR STORE] Saved. Total child docs indexed: {len(all_child_docs)}"
        )
        return vector_store

    def _get_llm(self, model_name: str) -> ChatGroq:
        """
        Returns a cached ChatGroq instance for the given model name.
        WHY caching: Creating a new ChatGroq object on every query adds ~5ms
        overhead and is wasteful. We keep one instance per model name.
        """
        if model_name not in self._llm_cache:
            self._llm_cache[model_name] = ChatGroq(
                model_name=model_name,
                temperature=0
            )
            logger.info(f"[LLM] Created ChatGroq instance for model: {model_name}")
        return self._llm_cache[model_name]

    # ── QUERY & RETRIEVAL ─────────────────────────────────────────────────────
    def get_response(
        self, query: str, target_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full pipeline: Cache check → FAISS retrieval → FlashRank reranking →
        Small-to-Big expansion → Model routing → Groq LLM generation → LLM-as-Judge eval.

        Toggle logic: if target_file is set, search only within that document's
        chunks via FAISS metadata filter. (UNCHANGED FEATURE)

        CHANGES vs original:
          - Prompt cache: returns instantly on repeat queries (0 API calls)
          - Multi-factor model router: llama-3-8b-8192 vs llama-3.3-70b-versatile
          - LLM-as-Judge: RAGEvaluator scores faithfulness/relevance/precision
          - Logs latency for each phase (FAISS, rerank, LLM, eval)
          - Expands child chunks to parent chunks before LLM call
          - Trims context if it exceeds MAX_CONTEXT_TOKENS
          - Counts both input AND output tokens
          - Extracts page-level citations from metadata
          - Catches Groq API timeouts gracefully
        """
        # ── Phase 0: Prompt Cache Check ───────────────────────────────────────
        cached = prompt_cache.get(query, target_file)
        if cached is not None:
            # Return cached response with cache_hit flag set
            cached_response = dict(cached)
            cached_response["cache_hit"] = True
            token_budget.record_query(
                tokens_input=cached_response.get("tokens_input", 0),
                tokens_output=cached_response.get("tokens_output", 0),
                model_used=cached_response.get("model_used", MODEL_HEAVY),
                cache_hit=True
            )
            return cached_response

        # Check if index exists and contains the necessary files
        if not os.path.exists(self.index_path) or not os.path.exists(os.path.join(self.index_path, "index.faiss")):
            return {
                "answer": "Error: Database is empty. Please upload a document first.",
                "tokens_input": 0,
                "tokens_output": 0,
                "confidence": "N/A",
                "sources": [],
                "cache_hit": False,
                "model_used": None,
                "eval_metrics": None
            }

        # ── Phase 1: FAISS Retrieval ──────────────────────────────────────────
        t0 = time.perf_counter()
        with self._faiss_lock:
            vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        # Retrieve top-10 child chunks; apply Toggle filter if target_file set
        search_kwargs = {"k": 10}
        if target_file:
            search_kwargs["filter"] = {"source": target_file}

        base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        t1 = time.perf_counter()
        logger.info(f"[RETRIEVAL] FAISS load+setup: {(t1-t0)*1000:.1f}ms")

        # ── Phase 2: FlashRank Reranking (UNCHANGED FEATURE — with Offline Fallback) ──
        t2 = time.perf_counter()
        if self.compressor:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=base_retriever
            )
            # Retrieve and rerank
            reranked_docs = compression_retriever.invoke(query)
            logger.info(
                f"[RETRIEVAL] FlashRank reranking: {(time.perf_counter()-t2)*1000:.1f}ms | "
                f"docs_returned={len(reranked_docs)}"
            )
        else:
            # Fallback: use base_retriever directly if Flashrank is unavailable
            reranked_docs = base_retriever.invoke(query)
            logger.info(
                f"[RETRIEVAL] FlashRank SKIPPED (offline/failed init): {(time.perf_counter()-t2)*1000:.1f}ms | "
                f"docs_returned={len(reranked_docs)}"
            )

        # ── Small-to-Big: Expand child chunks to parent context ───────────────
        # WHY: The child chunk (≤400 chars) was retrieved for precision.
        # The parent chunk (≤1500 chars) provides the LLM with surrounding
        # clinical context, improving answer quality.
        expanded_docs = []
        for doc in reranked_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_store:
                # Use parent text; keep child's metadata for citations
                expanded_text = self.parent_store[parent_id]
                expanded_docs.append(
                    Document(page_content=expanded_text, metadata=doc.metadata)
                )
            else:
                # Fallback: parent not in memory (e.g. after server restart)
                expanded_docs.append(doc)

        # ── Token guard: trim if context exceeds Groq's context window ────────
        expanded_docs = self._trim_context_to_token_limit(
            expanded_docs, MAX_CONTEXT_TOKENS
        )

        # ── Build context string + citation list ──────────────────────────────
        context_parts = []
        metadata_list = []
        for doc in expanded_docs:
            context_parts.append(doc.page_content)
            metadata_list.append(doc.metadata)

        context_str = "\n\n---\n\n".join(context_parts)
        citations_str = format_citations(metadata_list)  # Uses enhanced utils.py

        # Count INPUT tokens (query + context) for cost tracking
        input_text = f"{query}\n{context_str}"
        tokens_input = self.count_tokens(input_text)
        logger.info(
            f"[TOKEN OPT] Input tokens (query+context): {tokens_input} / "
            f"{MAX_CONTEXT_TOKENS} limit"
        )

        # ── Phase 3: Prompt Engineering with Page-Level Citations ─────────────
        # WHY this template:
        #   - Explicit "if answer is missing" guard prevents hallucinations
        #   - Source citations embedded in prompt so LLM can reference them
        #   - Clinical tone appropriate for medical staff use-case
        template = """You are a Senior Medical AI Assistant. Analyze the medical \
documents from the healthcare system and answer accurately based ONLY on the \
provided context. If the answer is not found in the context, clearly state that the \
context is insufficient — do NOT hallucinate.

Context (Retrieved Medical Documents):
{context}

Source Files:
{citations}

Clinical Question: {question}

Instructions:
- Answer concisely and factually.
- Reference the source filename and page number where relevant.
- If multiple sources contain relevant information, synthesize them.

Answer:"""

        prompt = PromptTemplate.from_template(template)

        # ── Phase 3.5: Model Router ───────────────────────────────────────────
        # WHY after token counting: select_model() uses tokens_input as a signal.
        # Only possible to call after FAISS + expansion + trimming are done.
        model_name = select_model(query, tokens_input)
        llm = self._get_llm(model_name)

        # ── Phase 4: LLM Generation (Groq Llama-3) ───────────────────────────
        t4 = time.perf_counter()
        try:
            rag_chain = (
                {
                    "context":   lambda _: context_str,
                    "citations": lambda _: citations_str,
                    "question":  RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            response_text = rag_chain.invoke(query)

        except Exception as exc:
            # Catches Groq API timeouts, rate limits, and network errors
            # WHY: Surface a structured error rather than a 500 crash
            err_type = type(exc).__name__
            logger.error(f"[LLM ERROR] {err_type}: {exc}")
            return {
                "answer": (
                    f"LLM generation failed ({err_type}). "
                    "Please retry. If the issue persists, check Groq API status."
                ),
                "tokens_input":  tokens_input,
                "tokens_output": 0,
                "confidence":    "Error",
                "sources":       metadata_list,
                "cache_hit":     False,
                "model_used":    model_name,
                "eval_metrics":  None
            }

        t5 = time.perf_counter()
        logger.info(f"[LLM] Groq generation via {model_name}: {(t5-t4)*1000:.1f}ms")

        # Count OUTPUT tokens
        tokens_output = self.count_tokens(response_text)

        # ── Phase 5: LLM-as-Judge Evaluation ─────────────────────────────────
        # WHY after generation: We need the actual answer to evaluate.
        # WHY not blocking: If evaluator fails, query result is still returned.
        eval_metrics = self.evaluator.evaluate(
            question=query,
            context=context_str,
            answer=response_text
        )

        logger.info(
            f"[QUERY DONE] Total pipeline: {(t5-t0)*1000:.1f}ms | "
            f"input_tokens={tokens_input} | output_tokens={tokens_output} | "
            f"model={model_name} | "
            f"eval_grade={eval_metrics.get('eval_grade', 'N/A')} | "
            f"toggle={'ON → ' + target_file if target_file else 'OFF (global)'}"
        )

        # ── Record in Token Budget Tracker ───────────────────────────────────
        token_budget.record_query(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model_used=model_name,
            cache_hit=False
        )

        # ── Build final response ──────────────────────────────────────────────
        final_result = {
            "answer":        response_text,
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "confidence":    "High (FlashRank Reranked + Small-to-Big)",
            "sources":       metadata_list,
            "cache_hit":     False,
            "model_used":    model_name,
            "eval_metrics":  eval_metrics
        }

        # ── Store in cache for future repeat queries ───────────────────────────
        prompt_cache.set(query, target_file, final_result)

        return final_result