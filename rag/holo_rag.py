# Holo RAG Implementation
# RAG (Retrieval-Augmented Generation) functionality
# rag/holo_rag.py
from pathlib import Path
import os
import shutil
import subprocess
from typing import List
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
CONVERT_DIR = BASE_DIR / "rag" / "converted"
CALIBRE_BIN = os.environ.get("CALIBRE_BIN")  # optional full path to ebook-convert

# Perf knobs
RAG_MAX_FILES = int(os.environ.get("RAG_MAX_FILES", "500"))
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# Lazy singletons
_initialized = False
_query_engine = None

def _persist_dir_valid() -> bool:
    """Basic check for valid llama-index storage."""
    if not PERSIST_DIR.exists():
        return False
    # Expect at least a docstore.json for a valid persisted store
    docstore = PERSIST_DIR / "docstore.json"
    return docstore.exists()

def _clear_persist_dir():
    try:
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
    except Exception:
        pass

def _list_supported_files() -> List[Path]:
    # Support .txt, .md always; .pdf if PyMuPDF available; epub via Calibre conversion if available
    exts = {".txt", ".md"}
    pdf_supported = False
    try:
        import fitz  # PyMuPDF
        pdf_supported = True
    except Exception:
        pdf_supported = False
    if pdf_supported:
        exts.add(".pdf")

    files: List[Path] = []
    if BOOKS_DIR.exists():
        for p in BOOKS_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    # Include previously converted EPUB->TXT files
    if CONVERT_DIR.exists():
        for p in CONVERT_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
                files.append(p)
    # Cap total files for performance
    if len(files) > RAG_MAX_FILES:
        files = files[:RAG_MAX_FILES]
    return files

def _which(cmd: str) -> bool:
    try:
        from shutil import which
        return which(cmd) is not None
    except Exception:
        return False

def _convert_epubs_if_possible() -> str | None:
    """Convert .epub files in BOOKS_DIR to .txt into CONVERT_DIR using Calibre's ebook-convert if available.
    Returns an info message if conversion ran, else None.
    """
    if not BOOKS_DIR.exists():
        return None
    # Find epubs
    epubs = [p for p in BOOKS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".epub"]
    if not epubs:
        return None
    # Prefer explicit CALIBRE_BIN if provided
    calibre_cmd = CALIBRE_BIN if CALIBRE_BIN else "ebook-convert"
    if CALIBRE_BIN and not Path(CALIBRE_BIN).exists():
        calibre_cmd = "ebook-convert"
    if not (Path(calibre_cmd).exists() or _which(calibre_cmd)):
        return None
    CONVERT_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    for epub in epubs:
        out_txt = CONVERT_DIR / (epub.stem + ".txt")
        # Skip if already converted and newer than source
        if out_txt.exists() and out_txt.stat().st_mtime >= epub.stat().st_mtime:
            continue
        try:
            subprocess.run([calibre_cmd, str(epub), str(out_txt)], check=True, capture_output=True)
            converted += 1
        except Exception:
            continue
    if converted:
        return f"Converted {converted} EPUB file(s) via Calibre."
    return None

def _ensure_initialized():
    """Initialize RAG components once, if possible.

    Returns:
        tuple[bool, str | None]: (ok, error_message)
    """
    global _initialized, _query_engine
    if _initialized:
        return True, None

    # Verify books directory
    if not BOOKS_DIR.exists():
        return False, (
            f"Books directory '{BOOKS_DIR}' not found. "
            "Create it and add book files, or set BOOKS_DIR env var to an existing folder."
        )

    # Attempt EPUB conversion if possible, then gather files
    _convert_epubs_if_possible()
    # Check for supported files
    files = _list_supported_files()
    if not files:
        # Provide targeted guidance if folder only has unsupported types
        has_any = any(BOOKS_DIR.iterdir())
        msg = "No supported files found. Add .txt or .md"
        # Check for PDFs without PyMuPDF
        try:
            pdfs_present = any(p.suffix.lower() == ".pdf" for p in BOOKS_DIR.iterdir())
        except Exception:
            pdfs_present = False
        if pdfs_present:
            msg += ", or install 'pymupdf' to enable PDF support (pip install pymupdf)"
        # Check for epubs
        try:
            epubs_present = any(p.suffix.lower() == ".epub" for p in BOOKS_DIR.iterdir())
        except Exception:
            epubs_present = False
        if epubs_present:
            if _which("ebook-convert"):
                msg += ". EPUB will be auto-converted via Calibre on next run."
            else:
                msg += ". EPUB not supported by default. Install Calibre and ensure 'ebook-convert' is on PATH to auto-convert."
        return False, f"Books folder: {BOOKS_DIR}. {msg}."

    # Configure embeddings and vector store
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.chunk_size = RAG_CHUNK_SIZE

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("books")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load or build index
    if _persist_dir_valid():
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(PERSIST_DIR), vector_store=vector_store
            )
            index = load_index_from_storage(storage_context)
        except Exception:
            # Storage appears corrupted/partial; rebuild
            _clear_persist_dir()
            documents = SimpleDirectoryReader(
                input_files=[str(p) for p in files]
            ).load_data()
            parser = SentenceSplitter()
            nodes = parser.get_nodes_from_documents(documents)
            # Fresh storage context without persist_dir to avoid reading non-existent files
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    else:
        # Fresh build; attempt PDF parsing if available
        # Fresh build; build from gathered files
        try:
            documents = SimpleDirectoryReader(input_files=[str(p) for p in files]).load_data()
        except Exception:
            # If some inputs cause issues, filter by extension to txt/md
            txt_md = [str(p) for p in files if Path(p).suffix.lower() in {".txt", ".md"}]
            documents = SimpleDirectoryReader(input_files=txt_md).load_data()
        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)
        # Fresh storage context without persist_dir to avoid reading non-existent files
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(PERSIST_DIR))

    _query_engine = index.as_query_engine(similarity_top_k=RAG_TOP_K)
    _initialized = True
    return True, None

# Callable function for Streamlit
def holo_query_books(prompt: str) -> str:
    ok, err = _ensure_initialized()
    if not ok:
        return err
    response = _query_engine.query(prompt)
    return str(response)
