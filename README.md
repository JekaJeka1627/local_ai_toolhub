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

3. **Run the Application**
   ```bash
   streamlit run local_chat.py
   ```

## Project Structure

```
local_ai_toolhub/
├── local_chat.py          # Main Streamlit interface
├── model_clients.py       # AI model client interfaces
├── tools.py              # MCP tools and utilities
├── rag/
│   └── holo_rag.py       # RAG implementation
├── rag/chroma_store/     # Vector database (auto-created)
└── Books/                # Document collection
```

## Usage

1. Open http://localhost:8501 in your browser
2. Select a model (Qwen3 for general queries, Holo1 for book RAG)
3. Enter your prompt and optionally select tools
4. Click "Run" to get responses

## Requirements

- Python 3.8+
- LM Studio running locally (for Qwen3 model)
- Book collection in the specified directory