
# Agentic RAG (Free & Local) — LangChain + Ollama + FAISS

A lightweight, agentic Retrieval-Augmented Generation (RAG) system that runs **fully free** on your machine using:
- **Ollama** for the LLM (e.g., `llama3.1` or `mistral`)
- **Sentence-Transformers** for **free local embeddings** (`intfloat/e5-small-v2` by default)
- **FAISS** for an in-memory **vector store**
- **Cross-Encoder** reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for **quality**
- **DuckDuckGo** search tool for optional **live web** lookups (no API key)
- **LangChain Agent** (ReAct) to decide when to use tools, retrieval, and reasoning

## Features
- Ingest your docs (PDF/text/Markdown) → embed → index with FAISS.
- Agent chooses among tools: `retrieve`, `web_search`, `calculator`, and uses the LLM to synthesize answers.
- Reranking for better snippet quality.
- CLI and minimal FastAPI server.

## Quickstart

### 1) Install Ollama (one-time)
- Download from https://ollama.com (Windows, macOS, Linux).
- Pull a free model (choose one):
```bash
ollama pull llama3.1
# or
ollama pull mistral
```
> Works on CPU or GPU. For older/low-RAM machines, try `phi3` or `llama3.2`.

### 2) Create & activate a Python env (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

> Tip: First run may download model weights for sentence-transformers and cross-encoder.

### 4) Add some sample docs
Place files in `data/`. A few sample notes are already included. You can drop PDFs, `.txt`, `.md` etc.

### 5) Build the index
```bash
python ingest.py
```

### 6) Run the CLI
```bash
python rag_agent.py
```
Then ask questions like:
```
> What are the core ideas in the sample docs? cite sources.
```

### 7) (Optional) Run the API
```bash
uvicorn server:app --reload --port 8000
```
POST JSON:
```json
{"query": "summarize the docs and add citations"}
```

---

## Project Structure
```
agentic_rag_free/
├─ data/                    # put your docs here
├─ ingest.py                # builds FAISS index
├─ retriever.py             # embeddings, vector store, reranker
├─ tools.py                 # retrieve, web_search, calculator tools
├─ rag_agent.py             # CLI agent loop
├─ server.py                # FastAPI server (optional)
├─ prompts.py               # system & answer templates
├─ requirements.txt
└─ README.md
```

## Environment variables
Create `.env` if you want to tweak defaults; all have safe fallbacks.
```
OLLAMA_MODEL=llama3.1
EMBEDDING_MODEL=intfloat/e5-small-v2
TOP_K=6
RERANK_K=8
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

## Notes
- This is **free** to run locally. Web search uses DuckDuckGo (no key).
- For bigger corpora, consider persistent FAISS (save/load) or a DB-backed store.
- For best results, keep docs clean and use the reranker (on by default).

Enjoy! 🚀
