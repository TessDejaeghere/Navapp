"""Minimal OpenRouter client with retries (copied from VaidyaBot).

The key is read from chatbot/openrouter_key.txt, or else from the Navez project's
config/key_openrouter.txt. It is re-read on every call, so fixing it needs no restart.
"""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_FILES = [ROOT / "openrouter_key.txt", ROOT.parent / "config" / "key_openrouter.txt"]
BASE = "https://openrouter.ai/api/v1"


def api_key() -> str:
    for f in KEY_FILES:
        if f.exists():
            m = re.search(r"sk-or-[A-Za-z0-9\-_]+", f.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return m.group(0)
    raise RuntimeError("No OpenRouter key found (chatbot/openrouter_key.txt or config/key_openrouter.txt).")


def post(path: str, payload: dict, retries: int = 5, timeout: int = 180) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE}/{path}", data=body,
            headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                     "X-Title": "Navapp chatbot"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code is not None and code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{path}: HTTP {code}: {e.read().decode()[:300]}") from e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{path}: still failing after {retries} retries")


def embed(texts: list[str], model: str = "openai/text-embedding-3-large") -> tuple[list[list[float]], float]:
    r = post("embeddings", {"model": model, "input": texts})
    vecs = [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    return vecs, r.get("usage", {}).get("cost", 0.0)


def chat_with_tools(messages: list[dict], tools: list[dict], model: str = "anthropic/claude-sonnet-4.5",
                    max_tokens: int = 4096) -> tuple[dict, float]:
    """One assistant turn that may request tool calls (OpenAI-style function calling)."""
    r = post("chat/completions", {"model": model, "messages": messages, "tools": tools,
                                  "max_tokens": max_tokens, "temperature": 0.0})
    return r["choices"][0]["message"], r.get("usage", {}).get("cost", 0.0)


def chat(messages: list[dict], model: str = "anthropic/claude-sonnet-4.5",
         max_tokens: int = 4096, temperature: float = 0.0) -> tuple[str, float]:
    r = post("chat/completions", {"model": model, "messages": messages,
                                  "max_tokens": max_tokens, "temperature": temperature})
    return r["choices"][0]["message"]["content"], r.get("usage", {}).get("cost", 0.0)
