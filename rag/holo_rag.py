# Holo RAG Implementation
# RAG (Retrieval-Augmented Generation) functionality
# rag/holo_rag.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
BOOKS_DIR = Path(os.getenv("BOOKS_DIR", BASE_DIR / "Books"))
CHROMA_DIR = BASE_DIR / "rag" / "chroma_store"

# Setup once
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_or_create_collection("books")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load documents and create index
documents = SimpleDirectoryReader(str(BOOKS_DIR)).load_data()
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes, storage_context=None)
query_engine = index.as_query_engine(similarity_top_k=5)

# Callable function for Streamlit

def holo_query_books(prompt):
    response = query_engine.query(prompt)
    return str(response)
