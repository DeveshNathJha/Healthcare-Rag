# Healthcare RAG - Azure Cloud Scalability & Architecture

> **Current Approach:** Optimized for on-premise inference to ensure medical data privacy.
> **Cloud Roadmap:** Fully Dockerized and designed for seamless Azure migration at enterprise scale.

---

## 1. Current State - On-Premise Architecture

The system is intentionally built local-first for two reasons:
- **Healthcare data privacy** - No PHI (Protected Health Information) leaves the server
- **Cost efficiency** - FAISS + local embeddings = zero vector DB cost during development

| Component | Current Implementation | Rationale |
|-----------|----------------------|-----------|
| **Vector DB** | FAISS (`faiss-cpu`, local) | Zero cost, offline-capable, billion-scale similarity search |
| **File Storage** | Local `uploads/` directory | No cloud dependency; HIPAA-friendly by default |
| **Compute** | FastAPI on local server | Development & testing; Dockerized for cloud lift-and-shift |
| **LLM Inference** | Groq API → Llama-3 (8B / 70B) | 800+ tok/sec LPU inference; 10x cheaper than GPU hosting |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace, local) | 384-dim, fully offline after first download |
| **Prompt Cache** | In-memory SHA-256 cache, 1-hr TTL | RAM-based; Azure Redis Cache upgrade path for multi-instance |

---

## 2. Azure Cloud Architecture - Target State

```
┌──────────────────────────────────────────────────────────────┐
│                       CLIENT LAYER                           │
│              Web UI  /  REST Clients  /  Swagger UI          │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTPS / TLS
┌──────────────────────────▼───────────────────────────────────┐
│         Azure App Service  /  Azure Container Instances      │
│              FastAPI Backend  (Dockerized Image)             │
│       /upload   /query   /list-files   /stats   /delete      │
└───────┬──────────────────┬──────────────────┬────────────────┘
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼───────────────┐
│ Azure Blob   │  │ Azure AI Search  │  │  Azure Key Vault      │
│ Storage      │  │ (Vector Search)  │  │  GROQ_API_KEY         │
│              │  │ Replaces FAISS   │  │  Managed Identity     │
│ · PDFs       │  │ at 1M+ doc scale │  └──────────────────────┘
│ · FAISS idx  │  │ · Native RBAC    │
│ · AES-256    │  │ · Distributed    │
└──────────────┘  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   Groq LPU API  │
                  │ llama-3.1-8b    │
                  │ llama-3.3-70b   │
                  └─────────────────┘
```

**Stack summary:**

| Layer | Local (Dev) | Azure (Prod) |
|-------|------------|-------------|
| Compute | `uvicorn` local | Azure App Service / ACI |
| File Storage | `uploads/` folder | Azure Blob Storage |
| Vector DB | FAISS (local disk) | Azure AI Search |
| Secrets | `.env` file | Azure Key Vault |
| Cache | In-memory dict | Azure Cache for Redis |
| Embeddings | Local HuggingFace | Local HuggingFace (unchanged) |
| LLM | Groq API | Groq API (unchanged) |

---

## 3. Migration Roadmap - 3 Steps

### Step 1 - Data Layer: Azure Blob Storage

Replace local `uploads/` and `data/vector_store/` with Azure Blob containers.

```bash
pip install azure-storage-blob
```

```python
# uploads/ → Azure Blob container: "healthcare-docs"
# data/vector_store/ → Azure Blob container: "faiss-index"
# Driven entirely by environment variables - zero code change in business logic
```

**Benefits:**
- Multi-instance access (horizontal scaling enabled)
- AES-256 encryption at rest
- Geo-redundant storage option

---

### Step 2 - Compute: Docker → Azure Container Instances

The system is already structured for containerization:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and push to Azure Container Registry
az acr build --registry healthcareragacr --image healthcare-rag:v1 .

# Deploy to Azure Container Instances
az container create \
  --resource-group healthcare-rg \
  --name healthcare-rag-api \
  --image healthcareragacr.azurecr.io/healthcare-rag:v1 \
  --ports 8000 \
  --secure-environment-variables GROQ_API_KEY=$(az keyvault secret show ...)
```

---

### Step 3 - Vector Search: Azure AI Search (at million-doc scale)

LangChain's abstract vector store interface means FAISS → Azure AI Search is a **single-line swap** in `rag_chain.py`. All downstream retrieval, reranking, and generation logic remains unchanged.

```python
# Current - local FAISS
self.vector_store = FAISS.load_local(index_path, self.embeddings)

# Azure migration - single swap, everything else unchanged
from langchain_community.vectorstores import AzureSearch

self.vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name="healthcare-vectors",
    embedding_function=self.embeddings.embed_query,
)
# similarity_search(), metadata filters, toggle search - all work identically
```

**Scale decision matrix:**

| Document Scale | Recommended Solution | Reason |
|---------------|---------------------|--------|
| < 100K docs | FAISS (local) | Free, fast, no infrastructure |
| 100K – 1M docs | FAISS on Azure Persistent Disk | No code change, low cost |
| > 1M docs | Azure AI Search | Managed, distributed, RBAC, SLA |

---

## 4. Security & Compliance in Azure

| Concern | Azure Solution |
|---------|---------------|
| Secret management | Azure Key Vault + Managed Identity - no hardcoded keys |
| Network isolation | Azure Virtual Network + Private Endpoints |
| Data encryption | Azure Blob: AES-256 at rest; HTTPS in transit |
| Access control | Azure RBAC per user/role/service |
| HIPAA compliance | Azure HIPAA BAA available; all services within compliance boundary |
| Audit logging | Azure Monitor + Application Insights for full request tracing |

---

## 5. Why Local-First Was the Right Choice

| Decision | Rationale |
|----------|-----------|
| FAISS over Pinecone/Weaviate | Healthcare PHI cannot go to external cloud vector DBs without HIPAA BAA. Local FAISS = zero data exposure. |
| Groq API over self-hosted GPU | 800+ tok/sec on Groq LPU vs 2-3 min/query on CPU. $0/month on free tier vs $300+/month GPU instance. |
| Docker from day one | Ensures dev-to-prod parity. Same image runs locally and on ACI/App Service - no "works on my machine" issues. |
| LangChain vector store abstraction | FAISS, Azure AI Search, Pinecone, Qdrant - all share the same `similarity_search()` interface. Swap in one line. |

---

## 6. Docker Quickstart (Ready Today)

```bash
# Run locally with Docker
docker build -t healthcare-rag .

docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_key_here \
  -v $(pwd)/data:/app/data \
  healthcare-rag

# API available at: http://localhost:8000
# Swagger UI at:   http://localhost:8000/docs
```

---

> **Design principle:** All Azure components (Blob Storage, AI Search, Key Vault, Redis) are accessed through environment variables and LangChain's abstract interfaces.
> The same codebase runs locally for development and on Azure for production - **no code changes required for deployment, only configuration.**
