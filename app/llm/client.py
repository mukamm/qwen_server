import requests
from app.core.config import QWEN_HOST, QWEN_PORT

BASE_URL = f"http://{QWEN_HOST}:{QWEN_PORT}/v1/chat/completions"

def send_to_ai(prompt: str) -> str:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "qwen-7b-instruct"
    }
    resp = requests.post(BASE_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
