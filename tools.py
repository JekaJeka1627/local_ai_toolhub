# Tools Module
# Various tools and utilities for the AI toolhub
# tools.py
import subprocess

# Example MCP tool: Shell command executor
def shell_tool(prompt):
    try:
        result = subprocess.run(prompt, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Shell error: {e.stderr.strip()}"

# Example MCP tool: Text spellchecker (dummy)
def spellcheck_tool(prompt):
    return prompt.replace("teh", "the")  # You can upgrade this with actual NLP tools

mcp_tools = {
    "Shell Executor": {
        "description": "Run shell commands on your server.",
        "handler": shell_tool
    },
    "Spellchecker": {
        "description": "Fix simple typos in your input.",
        "handler": spellcheck_tool
    }
}

def run_tool(func, prompt):
    return func(prompt)
