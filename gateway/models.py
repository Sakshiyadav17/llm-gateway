"""Wrappers for the small (local) and big (API) models."""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_small(question: str, temperature: float = 0.0, timeout: int = 120) -> dict:
    """Ask the local Ollama model."""
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": question}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        if not r.ok:
            return {"answer": None, "latency": time.time() - start,
                    "error": f"{r.status_code}: {r.text[:300]}"}
        answer = r.json()["message"]["content"]
        return {"answer": answer, "latency": time.time() - start, "error": None}
    except Exception as e:
        return {"answer": None, "latency": time.time() - start, "error": str(e)}


def ask_big(question: str, temperature: float = 0.0, timeout: int = 60) -> dict:
    """Ask the big model through Groq's OpenAI-compatible API."""
    start = time.time()
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": question}],
                "temperature": temperature,
                "max_completion_tokens": 2048,
            },
            timeout=timeout,
        )
        if not r.ok:
            return {"answer": None, "latency": time.time() - start,
                    "tokens": None, "error": f"{r.status_code}: {r.text[:300]}"}
        data = r.json()
        answer = data["choices"][0]["message"].get("content") or ""
        tokens = data.get("usage", {}).get("total_tokens")
        return {"answer": answer, "latency": time.time() - start,
                "tokens": tokens, "error": None}
    except Exception as e:
        return {"answer": None, "latency": time.time() - start,
                "tokens": None, "error": str(e)}