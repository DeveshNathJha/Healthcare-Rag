# Healthcare RAG System 🏥

A production-ready Retrieval-Augmented Generation (RAG) system built for healthcare applications. This system provides context-aware Q&A against healthcare documents (PDF, CSV, Excel), employing advanced ML techniques including Dual-Layer OCR, FlashRank Reranking, Graph-RAG metadata-linking, and Small-to-Big Chunking for high-accuracy semantic search.

## Key Features

1. **Dual-Layer Ingestion (OCR + Digital)**: Uses PyMuPDF for native text and gracefully falls back to EasyOCR for scanned documents or images.
2. **Robust Structured Data Support**: Processes CSV and Excel files by chunking meaningful rows to extract tabular insights.
3. **Advanced RAG Pipeline**:
   - **FlashRank Reranking**: Boosts hit relevance using cross-encoder reranking.
   - **Small-to-Big Chunking**: Employs an intelligent parent-child chunking strategy (indexing fine-grained chunks but providing broader parent context to the LLM) for optimal semantic retrieval.
   - **Graph-RAG Metadata Linking**: Attaches rich metadata (page numbers, token stats, timing) to track data provenance across ingestion.
4. **Token Optimization & Cost Control**: Monitors API tokens for LLM interactions and gracefully trims oversized contexts.
5. **Interactive Search Modes**:
   - **Global Search**: Queries the entire document FAISS index.
   - **Toggle Search**: Filters the FAISS index to target queries to specific documents.
6. **MLOps Best Practices**: Comprehensive logging via `RotatingFileHandler`, execution metrics, API tracing, error handling overrides, and a dedicated `/stats` endpoint for system monitoring.

## Architecture

* **Framework**: FastAPI
* **Embedding Model**: FastEmbed / BGE
* **Vector Store**: FAISS
* **LLM**: Llama-3 (via Groq API)
* **Metadata Splitting**: LangChain text splitters

## Installation

1. Clone the repository and navigate to the project root:
   ```bash
   cd healthcare_rag
   ```

2. Create a virtual environment (recommended) and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Starting the Server

Launch the FastAPI application using Uvicorn:

```bash
uvicorn app.main:app --reload
```
The server will start on `http://localhost:8000`.

### Interacting with the API

You can use the built-in Swagger UI at `http://localhost:8000/docs` to test endpoints, or verify functionality using the existing `test_script.py`.

* **`POST /upload`**: Upload PDF, CSV, or Excel files. Returns parsing statistics.
* **`POST /query`**: Pass `{"question": "..."}` or `{"question": "...", "target_file": "..."}` to run context-aware Q&A.
* **`GET /list-files`**: Retrieve a list of all currently indexed documents.
* **`GET /stats`**: View MLOps metrics (upload counts, FAISS index size, system health).
* **`DELETE /delete-document`**: Safely remove documents from local disk tracking.

### Running Tests

Execute the comprehensive test script to ensure systems are running smoothly:

```bash
python test_script.py
```

## Logs & Monitoring

Logs are generated with automatic rolling over at 5MB limits and are stored in:
`logs/healthcare_rag.log`

This tracking includes individual function latency tracking, LLM input/output token counts, and FAISS insertion speeds.
