"""Spending stop: read OpenRouter's own counters before each batch of games.

Two independent limits (config/study.yaml -> budget):
  key_usage_stop_usd       total spend on this project's API key (the key is used only for this study)
  min_account_balance_usd  account credits minus account usage; protects the reserve
Plus an optional per-run cap (--max-spend) on new spend counted from our own call logs.
"""
from __future__ import annotations

import httpx

from .llm import api_key, config

BASE = "https://openrouter.ai/api/v1"


def status(timeout: float = 30) -> dict:
    h = {"Authorization": f"Bearer {api_key()}"}
    cr = httpx.get(f"{BASE}/credits", headers=h, timeout=timeout).json()["data"]
    ky = httpx.get(f"{BASE}/key", headers=h, timeout=timeout).json()["data"]
    return {"account_balance": round(cr["total_credits"] - cr["total_usage"], 4),
            "key_usage": round(ky.get("usage", 0.0) or 0.0, 4),
            "key_limit_remaining": ky.get("limit_remaining")}


def check(run_spend: float = 0.0, max_spend: float | None = None) -> tuple[bool, str, dict]:
    """Return (ok, reason, status). ok=False means: do not start another batch."""
    b = config().get("budget", {})
    import time
    s, err = None, None
    for attempt in range(3):              # a brief network blip should not stop a run: 3 tries, 20 s apart
        try:
            s = status()
            break
        except Exception as ex:  # noqa: BLE001
            err = ex
            if attempt < 2:
                time.sleep(20)
    if s is None:                         # still unreadable: stop rather than spend blind
        return False, f"could not read OpenRouter credits after 3 tries ({err!r:.120})", {}
    if s["key_usage"] >= b.get("key_usage_stop_usd", float("inf")):
        return False, f"key usage ${s['key_usage']:.2f} reached the stop (${b['key_usage_stop_usd']})", s
    if s["account_balance"] <= b.get("min_account_balance_usd", 0):
        return False, f"account balance ${s['account_balance']:.2f} at the reserve floor", s
    if max_spend is not None and run_spend >= max_spend:
        return False, f"this run spent ${run_spend:.2f} >= --max-spend ${max_spend:.2f}", s
    return True, "ok", s
