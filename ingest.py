
# For this simple starter, ingest happens on-demand inside retriever.build_or_load_faiss().
# This script keeps a similar API so you can call it explicitly to pre-warm models and index.
from retriever import build_or_load_faiss

if __name__ == "__main__":
    print("Building local FAISS index (and caching models if needed)...")
    _ = build_or_load_faiss()
    print("✅ Done.")
