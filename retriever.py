
import os
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Tuple
# from sentence_transformers import SentenceTransformer
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
from langchain_community.embeddings import HuggingFaceEmbeddings

# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from sentence_transformers import  CrossEncoder
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

def _hash_path(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]

def load_docs(data_dir: str = "data") -> List[Document]:
    docs = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        for path in glob.glob(os.path.join(data_dir, ext)):
            try:
                if path.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    pages = loader.load()
                    for p in pages:
                        p.metadata.setdefault("source", f"Doc:{os.path.basename(path)} p.{p.metadata.get('page', 'NA')}")
                    docs.extend(pages)
                elif path.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(path)
                    docs.extend(loader.load())
                    for d in docs:
                        d.metadata.setdefault("source", f"Doc:{os.path.basename(path)}")
                else:
                    loader = TextLoader(path, encoding="utf-8")
                    dd = loader.load()
                    for d in dd:
                        d.metadata.setdefault("source", f"Doc:{os.path.basename(path)}")
                    docs.extend(dd)
            except Exception as e:
                print(f"[warn] failed to load {path}: {e}")
    return docs

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    return splitter.split_documents(docs)

@dataclass
class RetrieverBundle:
    vs: FAISS
    # embedder: SentenceTransformer
    reranker: CrossEncoder

def build_or_load_faiss(index_path: str = None, data_dir: str = "data") -> RetrieverBundle:
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    docs = load_docs(data_dir=data_dir)
    if not docs:
        raise RuntimeError("No documents found in ./data. Please add PDFs, .txt, or .md files.")
    chunks = split_docs(docs)
    texts = [c.page_content for c in chunks]


    metadatas = [c.metadata for c in chunks]
    vs = FAISS.from_texts(texts, embedding=embedder, metadatas=metadatas)
    return RetrieverBundle(vs=vs, embedder=embedder, reranker=reranker)

def retrieve(bundle: RetrieverBundle, query: str, top_k: int = 6, rerank_k: int = 8) -> List[Tuple[str, str]]:
    # Initial dense retrieval
    docs = bundle.vs.similarity_search(query, k=rerank_k)
    pairs = [(d.page_content, d.metadata.get("source", "Doc:unknown")) for d in docs]
    # Rerank with CrossEncoder
    if pairs:
        scored = bundle.reranker.predict([(query, p[0]) for p in pairs])
        ranked = sorted(zip(pairs, scored), key=lambda x: x[1], reverse=True)[:top_k]
        return [(p[0], p[1]) for (p, s) in ranked]
    return []
