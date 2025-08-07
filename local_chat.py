# local_chat.py
import streamlit as st
from model_clients import query_model
from tools import mcp_tools, run_tool

st.set_page_config(page_title="Local AI ToolHub", layout="wide")
st.title("📚 Local AI ToolHub – Query + Tools")

# Model and tool selection in main layout
st.header("🤖 Model & Tools")
model_col, tool_col = st.columns(2)

with model_col:
    model = st.selectbox("Select Model", ["Qwen3 (General) ✨", "Holo1 (Book RAG) 📚"])

with tool_col:
    selected_tool = None
    if model.startswith("Qwen3"):
        st.markdown("### 🔧 MCP Tool (Optional)")
        selected_tool = st.selectbox("Choose a tool (or none)", ["None"] + list(mcp_tools.keys()))

prompt = st.text_area("Your Prompt", height=200)
submit = st.button("Run", type="primary")

# Main output area
if submit and prompt:
    with st.spinner("Thinking..."):
        tool_output = None

        if model.startswith("Qwen3") and selected_tool != "None":
            tool_func = mcp_tools[selected_tool]['handler']
            tool_output = run_tool(tool_func, prompt)

        response = query_model(model.split(" ")[0], prompt, tool_result=tool_output)

    st.subheader("📝 Response")
    st.markdown(response)

    if tool_output:
        st.markdown("---")
        st.subheader(f"🔧 {selected_tool} Tool Output")
        st.code(tool_output)

else:
    st.info("Enter a prompt and click 'Run' to begin.")
