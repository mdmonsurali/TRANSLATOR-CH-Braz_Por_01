"""Thin async client for the llama.cpp server's OpenAI-compatible endpoint.

Only `chat_completion()` is used by the translator. The llama.cpp server
exposes /v1/chat/completions on the same host:port we control."""

from __future__ import annotations

import os
from typing import Optional

import httpx

LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = os.getenv("LLAMA_PORT", "8090")
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

TRANSLATE_MAX_TOKENS = int(os.getenv("TRANSLATE_MAX_TOKENS", "4096"))
# PER-REQUEST ceiling on one llama.cpp call (a chunk of ~20 strings) — the
# translator's equivalent of OCR_REQUEST_TIMEOUT_SEC, and the timeout that
# actually protects the pipeline. On timeout the chunk keeps its source text and
# is counted as `failed`; the rest of the document still translates.
#
# Deliberately NOT named TRANSLATE_TIMEOUT_SEC: that name is already the
# orchestrator's whole-document safety net (24h). Sharing it would silently
# stretch this per-chunk bound to 24h and re-introduce the hang it prevents.
TRANSLATE_REQUEST_TIMEOUT_SEC = float(
    os.getenv("TRANSLATE_REQUEST_TIMEOUT_SEC", "600"))


async def chat_completion(
    system: str,
    user: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    response_format_json: bool = False,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Call llama.cpp /v1/chat/completions and return the assistant content
    string. Raises httpx.HTTPError on transport / status failure."""
    payload: dict = {
        "model": "translator",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens or TRANSLATE_MAX_TOKENS,
        "stream": False,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=TRANSLATE_REQUEST_TIMEOUT_SEC)
    try:
        resp = await client.post(f"{LLAMA_BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
    finally:
        if owns_client:
            await client.aclose()

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"llama.cpp returned no choices: {body}")
    return choices[0].get("message", {}).get("content", "") or ""


async def health() -> bool:
    """Cheap reachability probe for the embedded llama-server."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{LLAMA_BASE_URL}/health")
            return r.status_code == 200
    except Exception:
        return False
