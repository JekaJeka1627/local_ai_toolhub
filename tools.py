# Tools Module
# Various tools and utilities for the AI toolhub
# tools.py
import subprocess
from typing import List
import textwrap

# Example MCP tool: Shell command executor
def shell_tool(prompt):
    q = (prompt or "").strip()
    # Gentle guard: if the input looks like a plain natural-language question, suggest Web Search instead
    words = q.split()
    likely_natural = (
        len(words) >= 3
        and not any(tok in q for tok in ["&", "|", ">", "<", ";", "&&", "||"])
        and not any(q.lower().startswith(cmd) for cmd in ["dir", "ls", "echo", "type", "pip", "python", "git", "powershell", "cmd"])
    )
    if likely_natural:
        return "This looks like a question, not a shell command. Try the 'Web Search' tool."
    try:
        result = subprocess.run(q, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Shell error: {e.stderr.strip()}"

# Example MCP tool: Text spellchecker (dummy)
def spellcheck_tool(prompt):
    return prompt.replace("teh", "the")  # You can upgrade this with actual NLP tools

# Web search via DuckDuckGo (no API key required)
def web_search_tool(prompt: str) -> str:
    query = prompt.strip()
    if not query:
        return "Provide a search query."
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        return (
            "duckduckgo-search not installed. Install with: pip install duckduckgo-search"
        )
    try:
        results: List[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5, region="us-en", safesearch="moderate"):
                results.append(r)
        if not results:
            return "No results."
        lines: List[str] = []
        for r in results:
            title = r.get("title", "(no title)")
            href = r.get("href", "")
            body = r.get("body", "")
            snippet = textwrap.shorten(body, width=220, placeholder="…") if body else ""
            lines.append(f"- {title}\n  {href}\n  {snippet}")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"

# Simple URL fetcher (HTML text)
def fetch_url_tool(prompt: str) -> str:
    url = prompt.strip()
    if not url:
        return "Provide a URL to fetch."
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    try:
        import requests
        resp = requests.get(url, timeout=20)
    except Exception as e:
        return f"Request error: {e}"
    if resp.status_code != 200:
        body = resp.text[:500] if resp.text else ""
        return f"HTTP {resp.status_code}: {body}"
    text = resp.text
    if len(text) > 4000:
        text = text[:4000] + "\n… [truncated]"
    return text

mcp_tools = {
    "Shell Executor": {
        "description": "Run shell commands on your server.",
        "handler": shell_tool
    },
    "Spellchecker": {
        "description": "Fix simple typos in your input.",
        "handler": spellcheck_tool
    },
    "Web Search": {
        "description": "DuckDuckGo web search (top 5 results).",
        "handler": web_search_tool
    },
    "Fetch URL": {
        "description": "Fetch raw HTML/text from a URL.",
        "handler": fetch_url_tool
    }
}

def run_tool(func, prompt):
    return func(prompt)
