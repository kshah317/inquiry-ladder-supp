"""Anchor run: Qwen3.8-Max plays the solo game on the main sample.

Chunk-safe for the laptop VM shell (~175 s per call): after --issue-for seconds no NEW (uncached)
request is sent, and requests already in flight are allowed to finish (they are cached), so no paid
call is lost. Games that hit the gate stop early and are simply replayed from cache next time.

usage: python3 scripts/21_run_anchor.py --estimate
       python3 scripts/21_run_anchor.py [--sample trial|main] [--issue-for 60] [--max-spend 45]
Output: results/main/ANCHOR/A.jsonl + MANIFEST.json (resumable).
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import budget, game, manifest  # noqa: E402
from ddx.cache import request_key  # noqa: E402
from ddx.data import get_patient  # noqa: E402
from ddx.llm import build_payload, cache, config  # noqa: E402
from ddx.panels import PANELS  # noqa: E402
from ddx.sample import load_sample  # noqa: E402

AGENT = PANELS["ANCHOR"][0]
OUT = ROOT / "results" / "main" / "ANCHOR"
PILOT_COST_PER_CALL = 0.0172            # mean of the 223 cached pilot calls


class Paused(Exception):
    pass


STATE = {"deadline": float("inf"), "new_calls": 0, "inflight": 0, "call_spend": 0.0, "max_spend": float("inf")}
_orig = game.call


async def gated_call(client, sem, model_key, messages, *, temperature, seed, max_tokens=600):
    key = request_key(build_payload(model_key, messages, temperature=temperature, seed=seed, max_tokens=max_tokens))
    if cache().get(key) is None:
        if time.time() > STATE["deadline"]:
            raise Paused()
        STATE["new_calls"] += 1
        STATE["inflight"] += 1
        try:
            resp = await _orig(client, sem, model_key, messages, temperature=temperature, seed=seed, max_tokens=max_tokens)
            STATE["call_spend"] += (resp.get("usage") or {}).get("cost") or 0     # cap counts every call as it returns
            if STATE["call_spend"] > STATE["max_spend"]:
                STATE["deadline"] = 0
            return resp
        finally:
            STATE["inflight"] -= 1
    return await _orig(client, sem, model_key, messages, temperature=temperature, seed=seed, max_tokens=max_tokens)

game.call = gated_call


async def main(a):
    cfg = config()
    s = load_sample(a.sample)
    split = s.split.iloc[0]
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "A.jsonl"
    done = {json.loads(x)["pid"] for x in f.open(encoding="utf-8")} if f.exists() else set()
    rows = [int(r) for r in s.row if f"{split}:{int(r)}" not in done]
    print(f"anchor [{a.sample}]: {len(rows)} games left of {len(s)}; rough estimate "
          f"{len(rows) * 6} calls ~${len(rows) * 6 * PILOT_COST_PER_CALL:.2f} (+ retries; cached calls are free)", flush=True)
    if a.estimate or not rows:
        return
    ok, why, st = budget.check()
    print("budget at start:", st, why, flush=True)
    if not ok:
        return
    t0 = time.time()
    STATE["deadline"] = t0 + a.issue_for
    STATE["max_spend"] = a.max_spend - a.concurrency * 0.05     # leave room for calls already in flight
    sem = asyncio.Semaphore(a.concurrency)
    res = {"finished": 0, "paused": 0, "failed": 0, "spend": 0.0}
    async with httpx.AsyncClient() as client:
        async def one(row):
            try:
                rec = await game.play(client, sem, get_patient(split, row), [AGENT], rounds=cfg["game"]["rounds"],
                                      temperature=cfg["game"]["temperature"], base_seed=cfg["seed"])
            except Paused:
                res["paused"] += 1
                return
            except Exception as ex:  # noqa: BLE001
                res["failed"] += 1
                msg = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} A {split}:{row} {ex!r:.300}"
                print("game failed:", msg, file=sys.stderr, flush=True)
                with (OUT / "failures.log").open("a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
                return
            rec.update(subgroup="A", panel_id="ANCHOR", sample=a.sample, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            with f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            res["finished"] += 1
            res["spend"] += sum((u.get("cost") or 0) for u in rec["usage"] if not u.get("cached"))
            if res["spend"] > a.max_spend and STATE["deadline"] > time.time():
                STATE["deadline"] = 0          # spending cap: stop sending new requests, let in-flight ones finish
                print(f"max-spend ${a.max_spend} reached: no new requests", flush=True)
        await asyncio.gather(*(one(r) for r in rows[: a.max_games]))
    info = {"seconds": round(time.time() - t0), **res, "new_calls_sent": STATE["new_calls"],
            "left": len(rows) - res["finished"]}
    print("anchor:", info, flush=True)
    m = manifest.build(OUT, [AGENT], a.sample, info)
    print(f"manifest: {m['games']['total']} games, fresh spend so far ${m['totals']['cost_fresh_usd']}, "
          f"cost/game ${m['totals']['cost_per_game_usd']}, truncated calls {m['totals']['truncated_calls']}", flush=True)
    print("budget now:", budget.status(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="main")
    ap.add_argument("--issue-for", type=float, default=60, help="seconds during which new requests may start")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-spend", type=float, default=45, help="USD of new spend after which no new request starts")
    ap.add_argument("--max-games", type=int, default=10_000)
    ap.add_argument("--estimate", action="store_true")
    asyncio.run(main(ap.parse_args()))
