# Local AI Toolhub

A local AI assistant with RAG capabilities and tool integration.

## Features

- 🤖 Multiple AI model support (Qwen3, Holo1)
- 📚 RAG (Retrieval-Augmented Generation) for book queries
- 🔧 MCP (Model Context Protocol) tool integration
- 💬 Streamlit web interface
- ⚡ Streaming responses for Qwen3
- 🔎 Built-in tools: Web Search, Fetch URL, Shell (guarded), Spellchecker (demo)

## Setup

1. **Install Dependencies**
   ```bash
   # core
   pip install streamlit llama-index chromadb huggingface-hub requests

   # tools
   pip install duckduckgo-search

   # optional: for semantic search in chat history
   pip install sentence-transformers

   # optional: PDF ingestion for RAG
   pip install pymupdf
   ```

2. **Start the App**
   ```bash
   streamlit run local_chat.py
   ```

3. **Select Model in the Sidebar**
   - Choose "Qwen3 (General) ✨" for general chat with tools and streaming.
   - Choose "Holo1 (Book RAG) 📚" to query your local `Books/` folder (txt/md by default; pdf via PyMuPDF; epub via Calibre).

---

## Models and Backends

### Qwen3 (General)
Qwen3 is accessed through an OpenAI-compatible Chat Completions API (e.g., LM Studio).

- Environment variables (can also be set in the app under "LM Studio Settings"):
  - `LM_STUDIO_URL` (default: `http://localhost:1234/v1/chat`)
  - `LM_STUDIO_MODEL` (default: `Qwen/Qwen1.5-7B-Chat-GGUF`)

- The app normalizes common URL forms to `/v1/chat/completions` as needed.
- Streaming is supported if the server provides OpenAI-style SSE streaming.

### Holo1 (Book RAG)
RAG over your local books using `llama-index` + ChromaDB + `BAAI/bge-small-en-v1.5` embeddings.

- Default directories:
  - `BOOKS_DIR` (default: `<repo>/local_ai_toolhub/Books/`)
  - `CHROMA_DIR` (default: `<repo>/local_ai_toolhub/rag/chroma_store`)
  - Storage persists under `<repo>/local_ai_toolhub/rag/storage`

- Supported formats:
  - `.txt`, `.md` always
  - `.pdf` if `pymupdf` is installed
  - `.epub` auto-conversion to `.txt` if Calibre's `ebook-convert` is available (`CALIBRE_BIN` can point to it)

- RAG tuning (env vars):
  - `RAG_MAX_FILES` (default `500`)
  - `RAG_CHUNK_SIZE` (default `512`)
  - `RAG_TOP_K` (default `5`)

---

## Tools (MCP-like)
Accessible when using Qwen3. Select a tool in the sidebar or let the app auto-run a Web Search for web-like queries.

- **Web Search**: DuckDuckGo top 5 results (requires `duckduckgo-search`).
- **Fetch URL**: Fetch raw HTML/text of a URL using `requests`.
- **Shell Executor**: Run shell commands (with a guard that detects natural-language queries and suggests Web Search).
- **Spellchecker**: Simple demo fixer for common typos.

Tool handlers are defined in `tools.py` within `mcp_tools`.

---

## Streaming, Regenerate, and UX

- **Streaming**: Qwen3 responses stream token-by-token with a Stop button.
- **Auto Web Search**: If no tool is selected and your prompt looks like a search query, the app auto-runs Web Search and passes a concise snippet to Qwen3.
- Upcoming (planned): Regenerate response, edit last message, copy buttons, and collapsible tool outputs.

---

## Usage

1. Launch the app: `streamlit run local_chat.py`.
2. Choose a model in the sidebar.
3. For Qwen3:
   - Configure LM Studio under "LM Studio Settings" and click "Test LM Studio".
   - Optionally select a tool or rely on auto Web Search hints.
4. For Holo1 (RAG): Ensure `Books/` contains `.txt`/`.md` (or enable PDFs/EPUBs as above). The sidebar provides helpers to create sample files.

---

## Troubleshooting

- **LM Studio timeout / connection errors**:
  - Ensure LM Studio is serving a Chat Completions API on the configured port.
  - Set `LM_STUDIO_URL` and `LM_STUDIO_MODEL` in the app and press "Test LM Studio".

- **Web Search says dependency missing**:
  - Install: `pip install duckduckgo-search`

- **RAG shows missing storage/docstore**:
  - The app auto-rebuilds storage. If it fails, delete `rag/storage/` and `rag/chroma_store/` and retry.

- **PDF/EPUB ingestion**:
  - PDFs require `pymupdf`.
  - EPUBs require Calibre's `ebook-convert` on PATH (or set `CALIBRE_BIN`).

---

## Project Structure

```bash
local_ai_toolhub/
├─ local_chat.py           # Streamlit UI (chat, tools, settings, streaming)
├─ model_clients.py        # Backends: Qwen3 via LM Studio, Holo1 RAG
├─ tools.py                # MCP-like tools: web search, fetch URL, shell, spellchecker
├─ chat_store.py           # SQLite-based conversation history + embeddings
├─ rag/
│  ├─ holo_rag.py          # LlamaIndex + Chroma RAG over Books/
│  ├─ storage/             # LlamaIndex persisted storage (auto-created)
│  └─ chroma_store/        # ChromaDB persistent store (auto-created)
├─ Books/                  # Your local text files for RAG (create or use UI helpers)
└─ .streamlit/config.toml  # Theme
```

---

## License

MIT (add your preferred license if different)
