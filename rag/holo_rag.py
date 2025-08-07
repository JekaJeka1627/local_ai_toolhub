# Holo RAG Implementation
# RAG (Retrieval-Augmented Generation) functionality
# rag/holo_rag.py
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
import chromadb
import os

BOOKS_DIR = os.getenv("BOOKS_DIR", r"C:\Users\jesse\Documents\Books\Books")
CHROMA_DIR = "rag/chroma_store"
PERSIST_DIR = "rag/storage"

# Setup once
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("books")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load or build index
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR, vector_store=vector_store
    )
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader(BOOKS_DIR).load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR, vector_store=vector_store
    )
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist()
query_engine = index.as_query_engine(similarity_top_k=5)

# Callable function for Streamlit

def holo_query_books(prompt):
    response = query_engine.query(prompt)
    return str(response)
