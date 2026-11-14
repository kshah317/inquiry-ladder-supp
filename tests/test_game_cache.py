"""Regression: replica 0 must make byte-identical requests to the pilot (every call a cache hit),
and replica > 0 must make new requests (no cache hit). No network is used: the HTTP transport
raises if anything tries to call the API. Run: python3 tests/test_game_cache.py"""
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.data import get_patient  # noqa: E402
from ddx.game import Agent, play  # noqa: E402
from ddx.llm import config  # noqa: E402


class NoNetwork(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise RuntimeError("network call attempted")


def _replay(rec, replica=0):
    cfg = config()
    a = rec["agents"][0]
    agent = Agent(a["name"], a["model"], a["role"], a["seed_id"], a.get("full_record", False), replica=replica)
    split, row = rec["pid"].split(":")

    async def go():
        async with httpx.AsyncClient(transport=NoNetwork()) as client:
            return await play(client, asyncio.Semaphore(4), get_patient(split, int(row)), [agent],
                              rounds=5, temperature=cfg["game"]["temperature"], base_seed=cfg["seed"])
    return asyncio.run(go())


def test_replica0_hits_cache_and_reproduces_pilot():
    f = ROOT / "results" / "pilot" / "gemma4_31b_generalist_r5_n98.jsonl"
    recs = [json.loads(line) for line in f.open(encoding="utf-8")][:5]
    for rec in recs:
        out = _replay(rec)
        assert all(u["cached"] for u in out["usage"])
        assert out["asked"] == rec["asked"] and out["panel"] == rec["panel"], rec["pid"]


def test_replica1_makes_new_requests():
    rec = json.loads((ROOT / "results" / "pilot" / "gemma4_31b_generalist_r5_n98.jsonl").open(encoding="utf-8").readline())
    try:
        _replay(rec, replica=1)
    except Exception as ex:  # noqa: BLE001  - LLMError wraps the transport error after retries
        assert "network call attempted" in repr(ex) or "gave up" in repr(ex)
        return
    raise AssertionError("replica 1 was served from cache")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as ex:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(ex)[:300])
    sys.exit(fails)
