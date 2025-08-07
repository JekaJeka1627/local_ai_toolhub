# model_clients.py
import requests
from rag.holo_rag import holo_query_books

LM_STUDIO_URL = "http://localhost:1234/v1/chat"

# Core interface for querying models
def query_model(model_name, prompt, tool_result=None):
    if model_name == "Qwen3":
        return query_qwen3(prompt, tool_result)
    elif model_name == "Holo1":
        return query_holo1(prompt)
    else:
        return "Unknown model."

def query_qwen3(prompt, tool_result=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if tool_result:
        messages.append({"role": "system", "content": f"Tool result: {tool_result}"})

    response = requests.post(LM_STUDIO_URL, json={
        "model": "Qwen/Qwen1.5-7B-Chat-GGUF",
        "messages": messages
    })

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No content.")
    else:
        return f"Error: {response.status_code}"

def query_holo1(prompt):
    return holo_query_books(prompt)
