# Local AI Toolhub

A local AI assistant with RAG capabilities and tool integration.

## Features

- 🤖 Multiple AI model support (Qwen3, Holo1)
- 📚 RAG (Retrieval-Augmented Generation) for book queries
- 🔧 MCP (Model Context Protocol) tool integration
- 💬 Streamlit web interface

## Setup

1. **Install Dependencies**

   ```bash
   pip install streamlit llama-index chromadb huggingface-hub
   ```

2. **Configure Hugging Face (Optional)**

   ```bash
   huggingface-cli login
   ```

3. **Set Book Directory (Optional)**

   ```bash
   export BOOKS_DIR=/path/to/your/books
   ```

4. **Run the Application**

   ```bash
   streamlit run local_chat.py
   ```

## Project Structure

```text
local_ai_toolhub/
├── local_chat.py          # Main Streamlit interface
├── model_clients.py       # AI model client interfaces
├── tools.py              # MCP tools and utilities
├── rag/
│   └── holo_rag.py       # RAG implementation
├── rag/chroma_store/     # Vector database (auto-created)
└── Books/                # Document collection
```

## Configuration

### Books Directory

By default, the app reads documents from the `Books/` directory in the project root.
To use a different location, set the `BOOKS_DIR` environment variable:

```bash
export BOOKS_DIR=/path/to/your/books
```

### LM Studio

The Qwen3 model is served through LM Studio.
The default configuration expects LM Studio at `http://localhost:1234`
running the `Qwen/Qwen1.5-7B-Chat-GGUF` model.
To change the port or model, set the following environment variables:

```bash
export LM_STUDIO_URL=http://localhost:1234/v1/chat
export LM_STUDIO_MODEL=Qwen/Qwen1.5-7B-Chat-GGUF
```

## Usage

1. Open <http://localhost:8501> in your browser
2. Select a model (Qwen3 for general queries, Holo1 for book RAG)
3. Enter your prompt and optionally select tools
4. Click "Run" to get responses

## Requirements

- Book collection in the `Books/` directory or path specified by `$BOOKS_DIR`
