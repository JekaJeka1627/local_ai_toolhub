# model_clients.py
import requests
import os
from rag.holo_rag import holo_query_books
import json

# Core interface for querying models
def query_model(model_name, prompt, tool_result=None):
    if model_name == "Qwen3":
        return query_qwen3(prompt, tool_result)
    elif model_name == "Holo1":
        return query_holo1(prompt)
    else:
        return "Unknown model."

def query_qwen3(prompt, tool_result=None):
    """Query an OpenAI-compatible chat endpoint (e.g., LM Studio).

    Handles common response variants to avoid returning a confusing 'No content.'
    """
    # Read environment on each call to allow live updates from the UI
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat")
    lm_studio_model = os.getenv("LM_STUDIO_MODEL", "Qwen/Qwen1.5-7B-Chat-GGUF")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if tool_result:
        messages.append({"role": "system", "content": f"Tool result: {tool_result}"})

    # Build a robust chat completions URL
    url = lm_studio_url.rstrip("/")
    if url.endswith("/chat"):
        url = f"{url}/completions"
    elif url.endswith("/v1"):
        url = f"{url}/chat/completions"
    # If already endswith /chat/completions, keep as-is

    try:
        resp = requests.post(
            url,
            json={
                "model": lm_studio_model,
                "messages": messages,
                "stream": False,
            },
            timeout=30,
        )
    except requests.RequestException as e:
        return f"Request error: {e} (URL={url})"

    if resp.status_code != 200:
        # Try to surface server error body for easier debugging
        try:
            return f"Error {resp.status_code}: {resp.text}"
        except Exception:
            return f"Error {resp.status_code} with no body"

    # Try multiple extraction patterns
    data = {}
    try:
        data = resp.json()
    except ValueError:
        return f"Invalid JSON response: {resp.text[:500]}"

    # 1) Standard OpenAI Chat Completions
    content = (
        data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
    )
    if content:
        return content

    # 2) Some servers use choices[0]["text"]
    content = data.get("choices", [{}])[0].get("text")
    if content:
        return content

    # 3) OpenAI-like content as list of items with {type: "text", text: "..."}
    msg = data.get("choices", [{}])[0].get("message", {})
    if isinstance(msg.get("content"), list):
        texts = [c.get("text", "") for c in msg.get("content", []) if isinstance(c, dict)]
        content = "".join(texts).strip()
        if content:
            return content

    # 4) Fallback: return a concise dump for visibility
    return f"Unexpected response format: {str(data)[:800]}"

def query_holo1(prompt):
    return holo_query_books(prompt)

# Streaming generator for Qwen3 (OpenAI-compatible streaming)
def stream_qwen3(prompt, tool_result=None):
    """Yield text chunks from an OpenAI-compatible streaming endpoint.

    This attempts to parse Server-Sent Events in the typical OpenAI format.
    Falls back gracefully if the server doesn't support streaming.
    """
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat")
    lm_studio_model = os.getenv("LM_STUDIO_MODEL", "Qwen/Qwen1.5-7B-Chat-GGUF")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if tool_result:
        messages.append({"role": "system", "content": f"Tool result: {tool_result}"})

    url = lm_studio_url.rstrip("/")
    if url.endswith("/chat"):
        url = f"{url}/completions"
    elif url.endswith("/v1"):
        url = f"{url}/chat/completions"

    try:
        resp = requests.post(
            url,
            json={
                "model": lm_studio_model,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=60,
        )
    except requests.RequestException as e:
        yield f"[stream error] {e} (URL={url})"
        return

    if resp.status_code != 200:
        try:
            body = resp.text
        except Exception:
            body = ""
        yield f"[stream http {resp.status_code}] {body[:500]}"
        return

    # Parse event stream lines
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        choice = (obj.get("choices") or [{}])[0]
        # OpenAI style
        delta = (choice.get("delta") or {}).get("content")
        if delta:
            yield delta
            continue
        # Some servers send full message blocks repeatedly
        msg = (choice.get("message") or {}).get("content")
        if msg:
            yield msg
