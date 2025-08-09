# local_chat.py
import streamlit as st
from model_clients import query_model, stream_qwen3
from tools import mcp_tools, run_tool
from chat_store import (
    init_db,
    list_conversations,
    create_conversation,
    add_message,
    get_messages,
    search_messages,
    update_conversation_title,
    delete_conversation,
    count_messages,
    upsert_message_embedding,
    get_messages_with_embeddings,
)
import os
import json
import math

st.set_page_config(page_title="Local AI ToolHub", layout="wide", initial_sidebar_state="expanded")
st.title("📚 Local AI ToolHub – Chat + Tools")

# ---------- Init DB and Session ----------
init_db()
if "conversation_id" not in st.session_state:
    # Start with a fresh conversation
    st.session_state.conversation_id = create_conversation("New Chat")
if "model" not in st.session_state:
    st.session_state.model = "Qwen3 (General) ✨"
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = "None"
if "embedder_loaded" not in st.session_state:
    st.session_state.embedder_loaded = False

# ---------- Sidebar: Conversations & Search ----------
with st.sidebar:
    st.header("💬 Conversations")
    convs = list_conversations(limit=50)
    conv_labels = [f"{c['title']} (#{c['id']})" for c in convs]
    conv_ids = [c["id"] for c in convs]
    if conv_ids:
        current_index = max(0, next((i for i, cid in enumerate(conv_ids) if cid == st.session_state.conversation_id), 0))
        selected = st.selectbox("Select a conversation", options=list(range(len(conv_ids))), format_func=lambda i: conv_labels[i], index=current_index)
        st.session_state.conversation_id = conv_ids[selected]

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.conversation_id = create_conversation("New Chat")
            st.rerun()
    with col_b:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Rename/Delete controls
    if conv_ids:
        st.write("")
        with st.expander("Manage conversation"):
            current_title = convs[selected]["title"]
            new_title = st.text_input("Title", value=current_title)
            c1, c2 = st.columns([2,1])
            with c1:
                if st.button("Rename", use_container_width=True):
                    update_conversation_title(st.session_state.conversation_id, new_title.strip() or current_title)
                    st.rerun()
            with c2:
                if st.button("Delete", type="secondary", use_container_width=True):
                    delete_conversation(st.session_state.conversation_id)
                    # Switch to a new chat
                    st.session_state.conversation_id = create_conversation("New Chat")
                    st.rerun()

    st.divider()
    st.subheader("🔎 Search")
    tabs = st.tabs(["Keyword", "Semantic"])
    with tabs[0]:
        q = st.text_input("Keyword search", placeholder="keywords…", key="kw_search")
        if q:
            results = search_messages(q)
            for r in results:
                if st.button(f"➡️ {r['title']} (#{r['conversation_id']})\n{r['snippet']}…", key=f"sr_{r['conversation_id']}"):
                    st.session_state.conversation_id = r["conversation_id"]
                    st.rerun()
    with tabs[1]:
        def _load_embedder():
            if not st.session_state.embedder_loaded:
                try:
                    from sentence_transformers import SentenceTransformer
                    st.session_state.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.embedder_loaded = True
                except Exception as e:
                    st.warning("Install sentence-transformers to enable semantic search: pip install sentence-transformers")
                    st.session_state.embedder = None
                    st.session_state.embedder_loaded = False

        sem_q = st.text_input("Semantic search", placeholder="natural language query…", key="sem_search")
        if st.button("Run semantic search"):
            _load_embedder()
            if st.session_state.embedder_loaded:
                with st.spinner("Embedding and searching…"):
                    emb = st.session_state.embedder.encode([sem_q])[0]
                    # Retrieve all embeddings
                    rows = get_messages_with_embeddings()
                    candidates = []
                    for r in rows:
                        vec = None
                        if r["vector_json"]:
                            try:
                                vec = json.loads(r["vector_json"])
                            except Exception:
                                vec = None
                        if vec is None:
                            continue
                        # cosine similarity
                        dot = sum(a*b for a, b in zip(emb, vec))
                        norm_a = math.sqrt(sum(a*a for a in emb))
                        norm_b = math.sqrt(sum(b*b for b in vec))
                        sim = dot / (norm_a * norm_b + 1e-9)
                        candidates.append((sim, r))
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    top = candidates[:10]
                    for score, r in top:
                        label = f"➡️ {r['title']} (#{r['conversation_id']})\n{r['content'][:120]}…\n(sim {score:.3f})"
                        if st.button(label, key=f"sem_{r['message_id']}"):
                            st.session_state.conversation_id = r["conversation_id"]
                            st.rerun()
            else:
                st.info("Semantic search unavailable. See warning above.")

    st.divider()
    st.subheader("⚙️ Model & Tools")
    st.session_state.model = st.selectbox(
        "Select Model",
        ["Qwen3 (General) ✨", "Holo1 (Book RAG) 📚"],
        index=0 if st.session_state.model.startswith("Qwen3") else 1,
    )
    if st.session_state.model.startswith("Qwen3"):
        st.markdown("### 🔧 MCP Tool (Optional)")
        st.session_state.selected_tool = st.selectbox(
            "Choose a tool (or none)", ["None"] + list(mcp_tools.keys()),
            index=(0 if st.session_state.get("selected_tool", "None") == "None" else 1),
        )
        # Show tool description
        if st.session_state.selected_tool != "None":
            try:
                desc = mcp_tools.get(st.session_state.selected_tool, {}).get("description", "")
                if desc:
                    st.caption(desc)
            except Exception:
                pass

        st.markdown("### ⚙️ LM Studio Settings")
        current_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat")
        current_model = os.environ.get("LM_STUDIO_MODEL", "Qwen/Qwen1.5-7B-Chat-GGUF")
        lm_url = st.text_input("LM_STUDIO_URL", value=current_url, placeholder="http://localhost:1234/v1/chat")
        lm_model = st.text_input("LM_STUDIO_MODEL", value=current_model, placeholder="Your model name as served by LM Studio")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Use these settings", use_container_width=True):
                os.environ["LM_STUDIO_URL"] = lm_url.strip() or current_url
                os.environ["LM_STUDIO_MODEL"] = lm_model.strip() or current_model
                st.success("Saved. These will apply to the next request.")
        with c2:
            if st.button("Test LM Studio", use_container_width=True):
                try:
                    import requests  # local import to avoid hard dep elsewhere
                    test_url = (lm_url or current_url).rstrip("/")
                    if test_url.endswith("/chat"):
                        test_url = f"{test_url}/completions"
                    elif test_url.endswith("/v1"):
                        test_url = f"{test_url}/chat/completions"
                    payload = {
                        "model": lm_model or current_model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "ping"},
                        ],
                        "stream": False,
                    }
                    r = requests.post(test_url, json=payload, timeout=10)
                    if r.status_code == 200:
                        st.success("LM Studio OK ✅")
                    else:
                        st.warning(f"HTTP {r.status_code}: {r.text[:300]}")
                except Exception as e:
                    st.error(f"Test failed: {e}")
    else:
        # Helper to create the Books directory
        books_dir = os.environ.get("BOOKS_DIR", os.path.join(os.path.dirname(__file__), "Books"))
        st.caption(f"Holo1 uses Books folder: {books_dir}")
        # Show count of .txt titles if available
        try:
            if os.path.isdir(books_dir):
                txts = [f for f in os.listdir(books_dir) if f.lower().endswith('.txt')]
                st.write(f"📚 Titles detected: {len(txts)}")
            else:
                st.write("📚 Titles detected: 0 (folder missing)")
        except Exception:
            pass
        custom_dir = st.text_input("Set a custom Books folder (BOOKS_DIR)", value=books_dir, placeholder="C:/path/to/Books")
        if st.button("Use this folder"):
            try:
                if custom_dir:
                    os.environ["BOOKS_DIR"] = custom_dir
                    st.success(f"BOOKS_DIR set to: {custom_dir}")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to set BOOKS_DIR: {e}")
        if st.button("📁 Create Books folder (with sample)"):
            try:
                os.makedirs(books_dir, exist_ok=True)
                sample_path = os.path.join(books_dir, "sample.txt")
                if not os.path.exists(sample_path):
                    with open(sample_path, "w", encoding="utf-8") as f:
                        f.write("This is a sample book file. Add more .txt files to improve RAG results.")
                st.success(f"Books folder ready at: {books_dir}")
            except Exception as e:
                st.error(f"Failed to create Books folder: {e}")
        if os.path.isdir(books_dir) and st.button("➕ Add two sample titles"):
            try:
                s1 = os.path.join(books_dir, "Pride_and_Prejudice_Chapter1.txt")
                s2 = os.path.join(books_dir, "Sherlock_A_Scandal_in_Bohemia.txt")
                if not os.path.exists(s1):
                    with open(s1, "w", encoding="utf-8") as f:
                        f.write("It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.")
                if not os.path.exists(s2):
                    with open(s2, "w", encoding="utf-8") as f:
                        f.write("To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name.")
                st.success("Added sample titles. Click Refresh and try Holo1 again.")
            except Exception as e:
                st.error(f"Failed to add samples: {e}")

# ---------- Load & Render Chat History ----------
messages = get_messages(st.session_state.conversation_id)
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    if role == "tool":
        with st.chat_message("assistant"):
            st.markdown("Tool output:")
            st.code(content)
    else:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

# ---------- Chat Input ----------
user_prompt = st.chat_input("Type your message…")
if user_prompt:
    # Save user message
    user_msg_id = add_message(st.session_state.conversation_id, "user", user_prompt)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Auto-title conversation on first user message
    try:
        if count_messages(st.session_state.conversation_id) == 1:
            update_conversation_title(st.session_state.conversation_id, user_prompt.strip()[:60])
    except Exception:
        pass

    # Optional tool run for Qwen3
    tool_output = None
    selected_tool = st.session_state.get("selected_tool", "None")
    model_display = st.session_state.model
    # Auto Web Search suggestion if no tool selected and prompt looks like a web query
    def _looks_like_web_query(text: str) -> bool:
        t = (text or "").strip().lower()
        keywords = ["who ", "what ", "when ", "where ", "why ", "how ", "latest", "news", "update", "meaning", "definition"]
        return any(k in t for k in keywords) and len(t) >= 8

    if model_display.startswith("Qwen3") and selected_tool == "None" and _looks_like_web_query(user_prompt):
        try:
            tool_func = mcp_tools["Web Search"]["handler"]
            tool_output = run_tool(tool_func, user_prompt)
            with st.chat_message("assistant"):
                st.markdown("🔎 Web Search (auto)")
                st.code(tool_output)
            add_message(st.session_state.conversation_id, "tool", tool_output)
        except Exception:
            pass

    if model_display.startswith("Qwen3") and selected_tool != "None":
        tool_func = mcp_tools[selected_tool]["handler"]
        tool_output = run_tool(tool_func, user_prompt)
        # Show tool output inline and store it
        with st.chat_message("assistant"):
            st.markdown(f"🔧 {selected_tool} Output:")
            st.code(tool_output)
        tool_msg_id = add_message(st.session_state.conversation_id, "tool", tool_output)

    # Query model
    if model_display.startswith("Qwen3"):
        with st.chat_message("assistant"):
            placeholder = st.empty()
            accumulated = ""
            stopped = False
            cols = st.columns([1,6])
            with cols[0]:
                if st.button("⏹ Stop", key=f"stop_{st.session_state.conversation_id}"):
                    st.session_state["stop_stream"] = True
            try:
                for chunk in stream_qwen3(user_prompt, tool_result=tool_output):
                    if st.session_state.get("stop_stream"):
                        stopped = True
                        break
                    if not chunk:
                        continue
                    accumulated += chunk
                    placeholder.markdown(accumulated)
            finally:
                st.session_state["stop_stream"] = False
            response = accumulated if accumulated else "(no content)"
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = query_model(model_display.split(" ")[0], user_prompt, tool_result=tool_output)
                st.markdown(response)

    asst_msg_id = add_message(st.session_state.conversation_id, "assistant", response)

    # Try to embed messages for semantic search
    try:
        if st.session_state.get("embedder_loaded"):
            emb = st.session_state.embedder.encode([user_prompt, response])
            if len(emb) == 2:
                upsert_message_embedding(user_msg_id, emb[0].tolist() if hasattr(emb[0], 'tolist') else list(emb[0]))
                upsert_message_embedding(asst_msg_id, emb[1].tolist() if hasattr(emb[1], 'tolist') else list(emb[1]))
    except Exception:
        # silently ignore embedding issues
        pass

    # Ensure latest render includes new messages
    st.rerun()
