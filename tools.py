
import os
from typing import List, Dict, Any

from duckduckgo_search import DDGS
from langchain.tools import tool

from retriever import build_or_load_faiss, retrieve

TOP_K = int(os.getenv("TOP_K", "6"))
RERANK_K = int(os.getenv("RERANK_K", "8"))

# Lazy init bundle
_bundle = None

def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = build_or_load_faiss()
    return _bundle

@tool
def retrieve_tool(query: str) -> str:
    """Search your local knowledge base and return the most relevant chunks with sources."""
    bundle = _get_bundle()
    hits = retrieve(bundle, query, top_k=TOP_K, rerank_k=RERANK_K)
    out = []
    for text, src in hits:
        out.append(f"[{src}]\n{text}")
    return "\n\n---\n\n".join(out) if out else "[no local matches]"

@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo (no API key). Returns a compact list of title, url, snippet."""
    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5, safesearch='moderate'):
            title = r.get('title', '')
            href = r.get('href', '')
            body = r.get('body', '')
            out.append(f"- {title}\n  {href}\n  {body}")
    return "\n".join(out) if out else "[no web results]"

@tool
def calculator(expression: str) -> str:
    """Evaluate a Python arithmetic expression safely (numbers, + - * / ** () )."""
    allowed = set("0123456789+-*/(). %**")  # simple safety net
    if any(ch not in allowed for ch in expression):
        return "[unsupported expression]"
    try:
        value = eval(expression, {"__builtins__": {}}, {})
        return str(value)
    except Exception as e:
        return f"[error: {e}]"
