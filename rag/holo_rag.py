# Holo RAG Implementation
# RAG (Retrieval-Augmented Generation) functionality
# rag/holo_rag.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import os


# Determine repository paths and allow overriding via environment variables
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BOOKS_DIR = os.path.join(REPO_ROOT, "Books")
DEFAULT_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

BOOKS_DIR = os.environ.get("BOOKS_DIR", DEFAULT_BOOKS_DIR)
CHROMA_DIR = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)

# Ensure the books directory exists before attempting to load files
if not os.path.exists(BOOKS_DIR):
    raise FileNotFoundError(
        f"Books directory '{BOOKS_DIR}' not found. Set the BOOKS_DIR environment variable to override."
    )

# Setup once
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("books")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load documents and create index
documents = SimpleDirectoryReader(BOOKS_DIR).load_data()
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes, storage_context=None)
query_engine = index.as_query_engine(similarity_top_k=5)

# Callable function for Streamlit

def holo_query_books(prompt):
    response = query_engine.query(prompt)
    return str(response)
