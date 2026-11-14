"""OpenRouter client: pinned provider, no fallbacks, thinking off, JSON output, cached.

`call()` is async so many patients can run at once; calls inside one consultation
stay in order because each round depends on the previous one.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from functools import lru_cache

import httpx
import yaml

from .cache import Cache, request_key
from .data import ROOT

URL = "https://openrouter.ai/api/v1/chat/completions"


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load((ROOT / "config" / "study.yaml").read_text())


@lru_cache(maxsize=1)
def api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in code/.env")


_cache: Cache | None = None


def cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def build_payload(model_key: str, messages: list[dict], *, temperature: float, seed: int,
                  max_tokens: int = 600) -> dict:
    m = config()["models"][model_key]
    payload = {
        "model": m["id"],
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "provider": {"order": [m["provider"]], "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True},
    }
    if m.get("reasoning") == "low":                 # gpt-oss: reasoning cannot be switched off; lowest effort
        payload["reasoning"] = {"effort": "low"}
        payload["max_tokens"] = max(max_tokens, 3000)
    elif m.get("reasoning") == "mandatory":           # anchor only: endpoint cannot turn thinking off
        payload["max_tokens"] = max(max_tokens, 8000)
    elif not m["id"].startswith("google/gemma-3"):   # gemma-3 has no thinking mode / reasoning param
        payload["reasoning"] = {"enabled": False}
    return payload


class LLMError(RuntimeError):
    pass


async def call(client: httpx.AsyncClient, sem: asyncio.Semaphore, model_key: str,
               messages: list[dict], *, temperature: float, seed: int, max_tokens: int = 600) -> dict:
    """Return the raw OpenRouter response (from cache if seen before)."""
    payload = build_payload(model_key, messages, temperature=temperature, seed=seed, max_tokens=max_tokens)
    key = request_key(payload)
    hit = cache().get(key)
    if hit is not None:
        hit["_cached"] = True
        return hit
    delay = 2.0
    for attempt in range(10):
        async with sem:
            try:
                r = await client.post(URL, json=payload, timeout=120,
                                      headers={"Authorization": f"Bearer {api_key()}"})
            except httpx.HTTPError as ex:
                err = repr(ex)
            else:
                if r.status_code == 200:
                    data = r.json()
                    if data.get("choices"):
                        cache().put(key, payload, data)
                        return data
                    err = json.dumps(data)[:300]
                else:
                    err = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code in (400, 401, 402, 404):
                        raise LLMError(err)
        print(f"[retry {attempt + 1}] {model_key}: {err[:160]}", file=sys.stderr, flush=True)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)
    raise LLMError(f"gave up after retries: {err}")


def content(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def parse_json(text: str) -> dict | None:
    """Parse the model's JSON. Tolerates code fences and text around one JSON object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                out = json.loads(m.group(0))
                return out if isinstance(out, dict) else None
            except json.JSONDecodeError:
                return None
    return None
