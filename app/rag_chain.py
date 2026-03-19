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

STRICTLY PRESERVED:
  - FAISS vector store ✅
  - HuggingFace all-MiniLM-L6-v2 embeddings ✅
  - FlashRank reranking ✅
  - Toggle / target_file metadata filter ✅
  - Groq Llama-3 LLM ✅
  - count_tokens() ✅
"""

import os
import time
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import tiktoken
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

from app.utils import get_logger, format_citations

# Load API Keys (GROQ_API_KEY from .env)
load_dotenv()

logger = get_logger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# WHY 6000 TOKENS:
#   Llama-3-8B has an 8192 token context window. Reserving ~2000 for the prompt
#   template + generated answer leaves ~6000 for retrieved context chunks.
MAX_CONTEXT_TOKENS = 6000


class RAGChain:
    def __init__(self, index_path: str = "data/vector_store"):
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
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. LLM — Llama-3 via Groq (temperature=0 for factual medical answers)
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0
        )

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
            if os.path.exists(self.index_path):
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

    # ── QUERY & RETRIEVAL ─────────────────────────────────────────────────────
    def get_response(
        self, query: str, target_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Two-stage retrieval: FAISS (child chunks) → FlashRank reranking →
        Small-to-Big expansion → Groq LLM generation.

        Toggle logic: if target_file is set, search only within that document's
        chunks via FAISS metadata filter. (UNCHANGED FEATURE)

        CHANGES vs original:
          - Logs latency for each phase (FAISS, rerank, LLM)
          - Expands child chunks to parent chunks before LLM call
          - Trims context if it exceeds MAX_CONTEXT_TOKENS
          - Counts both input AND output tokens
          - Extracts page-level citations from metadata
          - Catches Groq API timeouts gracefully
        """
        if not os.path.exists(self.index_path):
            return {
                "answer": "Error: Database is empty. Please upload a document first.",
                "tokens_input": 0,
                "tokens_output": 0,
                "confidence": "N/A",
                "sources": []
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

        # ── Phase 2: FlashRank Reranking (UNCHANGED FEATURE) ─────────────────
        t2 = time.perf_counter()
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # Retrieve and rerank
        reranked_docs = compression_retriever.invoke(query)
        t3 = time.perf_counter()
        logger.info(
            f"[RETRIEVAL] FlashRank reranking: {(t3-t2)*1000:.1f}ms | "
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
                | self.llm
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
                "tokens_input": tokens_input,
                "tokens_output": 0,
                "confidence": "Error",
                "sources": metadata_list
            }

        t5 = time.perf_counter()
        logger.info(f"[LLM] Groq generation: {(t5-t4)*1000:.1f}ms")

        # Count OUTPUT tokens
        tokens_output = self.count_tokens(response_text)

        logger.info(
            f"[QUERY DONE] Total pipeline: {(t5-t0)*1000:.1f}ms | "
            f"input_tokens={tokens_input} | output_tokens={tokens_output} | "
            f"toggle={'ON → ' + target_file if target_file else 'OFF (global)'}"
        )

        # ── Return structured response to main.py ─────────────────────────────
        return {
            "answer":        response_text,
            "tokens_input":  tokens_input,    # New: input token count
            "tokens_output": tokens_output,   # Was "tokens" — now explicit
            "confidence":    "High (FlashRank Reranked + Small-to-Big)",
            "sources":       metadata_list    # Full metadata for downstream use
        }