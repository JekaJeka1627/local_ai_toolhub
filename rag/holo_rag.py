# Holo RAG Implementation
# RAG (Retrieval-Augmented Generation) functionality
# rag/holo_rag.py
from pathlib import Path
import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

BASE_DIR = Path(__file__).resolve().parents[1]

BOOKS_DIR = Path(os.environ.get("BOOKS_DIR", BASE_DIR / "Books"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", BASE_DIR / "rag" / "chroma_store"))
PERSIST_DIR = BASE_DIR / "rag" / "storage"

# Ensure the books directory exists before attempting to load files
if not BOOKS_DIR.exists():
    raise FileNotFoundError(
        f"Books directory '{BOOKS_DIR}' not found. Set the BOOKS_DIR environment variable to override."
    )

# Setup once
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_or_create_collection("books")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load or build index
if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
    storage_context = StorageContext.from_defaults(
        persist_dir=str(PERSIST_DIR), vector_store=vector_store
    )
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader(str(BOOKS_DIR)).load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults(
        persist_dir=str(PERSIST_DIR), vector_store=vector_store
    )
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist()

query_engine = index.as_query_engine(similarity_top_k=5)

# Callable function for Streamlit
def holo_query_books(prompt: str) -> str:
    response = query_engine.query(prompt)
    return str(response)
