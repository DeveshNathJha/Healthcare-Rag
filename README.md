# Healthcare RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built for healthcare applications. This system provides context-aware Q&A against healthcare documents (PDF, CSV, Excel), employing advanced ML techniques including Dual-Layer OCR, FlashRank Reranking, Graph-RAG metadata-linking, Small-to-Big Chunking, LLM-as-Judge Evaluation, Prompt Caching, and Multi-Factor Model Routing.

---

## Key Features

### Phase 1 - Core RAG Pipeline
1. **Dual-Layer Ingestion (OCR + Digital)**: Uses PyMuPDF for native text extraction, falls back to EasyOCR (2x/3x DPI adaptive retry) for scanned documents.
2. **Structured Data Support**: Processes CSV and Excel files via pandas, treating each row as a sentence.
3. **Small-to-Big Chunking**: Child chunks (400 chars) embedded in FAISS for precision; parent chunks (1500 chars) fed to LLM for rich context.
4. **Graph-RAG Metadata Linking**: Every chunk carries `source`, `page`, `doc_type`, `ocr_used`, `ingested_at`, `parent_id`, `chunk_index` - enables citations and filtered search.
5. **FlashRank Reranking**: Cross-encoder reranks FAISS top-10 results by logical relevance before LLM generation.
6. **Token Optimization**: `tiktoken` counts input + output tokens; `_trim_context_to_token_limit()` prevents Groq context window overflow.
7. **Toggle / Global Search**: `target_file` parameter restricts FAISS search to a single document (Toggle mode) or all documents (Global mode).

### Phase 2 - Cost & Quality Optimization (NEW)
8. **LLM-as-Judge Evaluation** (`app/evaluator.py`): After every query, `llama-3-8b-8192` scores the answer on 3 semantic dimensions - no keyword matching, no extra dependencies.
   - **Faithfulness** - Hallucination detection (does answer use only retrieved context?)
   - **Answer Relevance** - Is the answer on-topic?
   - **Context Precision** - Was FAISS retrieval actually useful?
   - **Composite Grade** - A (≥0.85) / B (≥0.70) / C (≥0.50) / F (<0.50)
9. **Prompt Cache** (`app/rag_chain.py` - `PromptCache`): SHA-256 keyed in-memory cache with 1-hour TTL. Repeat queries served instantly (0 API calls). Auto-cleared on new document upload.
10. **Multi-Factor Model Router** (`app/rag_chain.py` - `select_model()`): Routes to `llama-3-8b-8192` (cheap, fast) only when ALL three signals are simple: query <=12 words, no complex clinical keywords, context <=3500 tokens. Otherwise routes to `llama-3.3-70b-versatile`.
11. **Token Budget Tracker** (`app/utils.py` - `TokenBudgetTracker`): Session-level tracking of token usage, model calls, cache hits, and estimated USD cost - visible on `/stats`.

### MLOps
12. **Structured Logging**: `RotatingFileHandler` (5 MB × 3 backups), `@log_performance` decorator, per-phase latency logging (FAISS / Rerank / LLM / Eval).
13. **Thread Safety**: `threading.Lock()` on all FAISS read/write operations.
14. **File Validation**: 50 MB upload limit enforced at API layer.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  FastAPI (main.py)               │
│  /upload  /query  /list-files  /stats  /delete  │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  processor.py  │ <- PyMuPDF + EasyOCR + pandas
       │  (Librarian)   │
       └───────┬────────┘
               │ pages_data (text + metadata)
       ┌───────▼────────┐
       │  rag_chain.py  │ <- FAISS + FlashRank + PromptCache + ModelRouter
       │  (Researcher)  │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  evaluator.py  │ <- LLM-as-Judge (llama-3-8b-8192, temp=0)
       │  (Evaluator)   │
       └───────┬────────┘
               │ eval_metrics attached to response
       ┌───────▼────────┐
       │   utils.py     │ <- Logging + TokenBudgetTracker + format_citations
       └────────────────┘

Stack:
  Embedding:   sentence-transformers/all-MiniLM-L6-v2 (384-dim, offline)
  Vector DB:   FAISS (local, CPU)
  Reranker:    FlashRank ms-marco-MiniLM-L-12-v2 (offline)
  LLM:         Groq API -> llama-3-8b-8192 / llama-3.3-70b-versatile
  Framework:   FastAPI + LangChain
```

---

## Installation

1. Clone the repository and navigate to the project root:
   ```bash
   cd healthcare_rag
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment variables - create a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Pre-cache models (required for offline/restricted environments):
   ```bash
   # HuggingFace embedding model
   python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"
   # FlashRank reranker
   python -c "from langchain_community.document_compressors import FlashrankRerank; FlashrankRerank()"
   ```

---

## Usage

### Starting the Server

```bash
uvicorn app.main:app --reload
```

Server starts on `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload PDF, CSV, or Excel. Returns chunk stats. Cache cleared automatically. |
| `POST` | `/query` | Ask a question. Returns answer + eval_metrics + cache_hit + model_used. |
| `GET`  | `/list-files` | List all indexed documents. |
| `GET`  | `/stats` | System health + token budget dashboard. |
| `DELETE` | `/delete-document/{filename}` | Remove a document from disk. |

### /query Request & Response

**Request:**
```json
{
  "question": "What are the symptoms of Type 2 Diabetes?",
  "target_file": null
}
```

**Response (Phase 2 format):**
```json
{
  "answer": "Type 2 Diabetes symptoms include increased thirst...",
  "tokens_input": 1240,
  "tokens_output": 187,
  "confidence": "High (FlashRank Reranked + Small-to-Big)",
  "sources": [{"source": "diabetes_guide.pdf", "page": 3, "doc_type": "pdf"}],
  "cache_hit": false,
  "model_used": "llama-3-8b-8192",
  "eval_metrics": {
    "faithfulness": 0.91,
    "answer_relevance": 0.88,
    "context_precision": 0.80,
    "eval_grade": "A",
    "judge_model": "llama-3-8b-8192",
    "eval_latency_ms": 340
  }
}
```

### /stats Response (Token Budget)

```json
{
  "upload_count": 3,
  "index_size_mb": 2.4,
  "index_exists": true,
  "log_file": "logs/healthcare_rag.log",
  "token_budget": {
    "total_queries": 47,
    "api_calls_made": 35,
    "cache_hits": 12,
    "cache_hit_rate_pct": 25.5,
    "total_input_tokens": 58420,
    "total_output_tokens": 9110,
    "model_8b_calls": 30,
    "model_70b_calls": 5,
    "estimated_cost_usd": 0.0038
  }
}
```

### Running Tests

```bash
python test_script.py
```

---

## Logs & Monitoring

Logs stored in `logs/healthcare_rag.log` with automatic 5 MB rotation (3 backups kept).

Per-request log includes:
- Phase-wise latency: FAISS / FlashRank / LLM / Evaluator
- Token counts (input + output)
- Model selected by router
- Cache hit/miss
- Eval grade

---

## Model Router Reference

| Signal | Threshold | Light Model (8B) | Heavy Model (70B) |
|--------|-----------|-----------------|-------------------|
| Query word count | <= 12 words | Yes | - |
| Complex keywords | "diagnose", "prognosis", "etiology", "contraindication"... | - | Yes |
| Context token count | > 3500 tokens | - | Yes |
| **Rule** | ALL three simple | **Use 8B** | **ANY complex -> Use 70B** |

---

## Cost Reference (Groq API, ~2025)

| Model | Per 1M Tokens | Typical Query Cost |
|-------|--------------|-------------------|
| `llama-3-8b-8192` | $0.05 | ~$0.00006 |
| `llama-3.3-70b-versatile` | $0.59 | ~$0.00073 |
| Judge eval call (8B) | $0.05 | ~$0.000025 |

---

## Cloud Scalability & Azure Deployment

While the current system is optimized for **on-premise inference** to ensure medical data privacy (FAISS runs locally, no PHI leaves the server), the architecture is **fully Dockerized and Azure-ready** for enterprise-scale deployment.

### Current State (On-Premise / Local)

| Component | Current Implementation | Rationale |
|-----------|----------------------|-----------|
| Vector DB | FAISS (local, CPU) | Zero cost, complete data privacy, offline-capable |
| File Storage | Local `uploads/` directory | No cloud dependency for dev/test |
| Compute | Local Python process | Development and testing environment |
| LLM Inference | Groq API (external) | 800+ tokens/sec, low latency |
| Embeddings | `all-MiniLM-L6-v2` (local) | Fully offline after first download |

### Azure Cloud Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                                │
│              (Web UI / REST Clients / Swagger)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼──────────────────────────────────────┐
│              Azure App Service / Azure Container Instances       │
│                   FastAPI Backend (Dockerized)                  │
│         /upload  /query  /list-files  /stats  /delete           │
└────┬──────────────────┬───────────────────┬─────────────────────┘
     │                  │                   │
┌────▼────┐    ┌────────▼────────┐   ┌──────▼──────────────────┐
│  Azure  │    │  Azure AI Search │   │  Azure Key Vault        │
│  Blob   │    │  (Vector Search) │   │  (GROQ_API_KEY, secrets)│
│ Storage │    │  Replaces FAISS  │   └─────────────────────────┘
│ (PDFs)  │    │  at million-doc  │
└─────────┘    │  scale           │
               └──────────────────┘
                        │
               ┌────────▼────────┐
               │   Groq LPU API  │
               │  (LLM Inference)│
               │  llama-3 8B/70B │
               └─────────────────┘
```

**Migration path — 3 steps:**

**Step 1 — Data Layer: Azure Blob Storage**
```bash
# Replace local uploads/ with Azure Blob
pip install azure-storage-blob
# AZURE_STORAGE_CONNECTION_STRING set in Azure Key Vault
```
- PDFs uploaded to Blob container `healthcare-docs`
- FAISS index persisted to Blob (`data/vector_store/` → Blob)
- Enables multi-instance file access (horizontal scaling)

**Step 2 — Compute: Docker + Azure Container Instances**
```dockerfile
# Dockerfile (already structured for containerization)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group healthcare-rg \
  --name healthcare-rag-api \
  --image devesh/healthcare-rag:latest \
  --ports 8000 \
  --environment-variables GROQ_API_KEY=@keyvault
```

**Step 3 — Vector Search: Azure AI Search (at million-doc scale)**

| Scale | Solution | Reason |
|-------|----------|--------|
| < 100K docs | FAISS (local) | Zero cost, fast, sufficient |
| 100K – 1M docs | FAISS on persistent Azure Disk | Scale without architecture change |
| > 1M docs | Azure AI Search | Managed distributed vector search, native filtering, RBAC |

```python
# Swap in rag_chain.py — single interface change
# Current:
self.vector_store = FAISS.load_local(index_path, self.embeddings)

# Azure AI Search migration:
from langchain_community.vectorstores import AzureSearch
self.vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name="healthcare-vectors",
    embedding_function=self.embeddings.embed_query,
)
# All downstream code (similarity_search, filter) remains unchanged
# LangChain abstracts the vector store interface
```

### Security in Azure Deployment

| Concern | Azure Solution |
|---------|---------------|
| API Key management | Azure Key Vault — no hardcoded secrets |
| Network security | Azure Virtual Network + Private Endpoints |
| Data encryption | Azure Blob Storage encryption at rest (AES-256) |
| Access control | Azure RBAC + Managed Identity |
| HIPAA compliance | Azure HIPAA BAA available for healthcare workloads |

### Docker Quickstart (Current — Ready to Deploy)

```bash
# Build image
docker build -t healthcare-rag .

# Run locally
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  healthcare-rag

# Push to Azure Container Registry
az acr build --registry healthcareragacr \
  --image healthcare-rag:v1 .
```

> **Design principle:** All cloud components (Blob Storage, Azure AI Search, Key Vault) are accessed through environment variables and LangChain's abstract vector store interface. This means the same codebase runs locally for development and on Azure for production — **zero code change required for deployment**.
