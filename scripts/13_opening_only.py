"""Opening-only baseline on the MAIN sample: each team model's generalist diagnoses from sex, age and
opening complaint only (0 questions). This is v(empty) baseline (b) for Shapley:
  P0, P1 -> Gemma 4 31B;  P1-Q -> Qwen3.8 27B;  P2 -> mean of its four members (set before the main runs).
Agents use the same seed_id as in the panels; the 0-question prompt differs, so no call is shared.

usage: python3 scripts/13_opening_only.py [--models gemma4_31b,...] [--budget 150]
Output: results/main/opening_only/<model>.jsonl + opening_only_summary.json
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
from ddx import budget  # noqa: E402
from ddx.data import get_patient  # noqa: E402
from ddx.game import play  # noqa: E402
from ddx.llm import config  # noqa: E402
from ddx.panels import PANELS  # noqa: E402
from ddx.sample import load_sample  # noqa: E402
from ddx.shapley import bootstrap_ci  # noqa: E402

AGENTS = {a.model: a for a in PANELS["P2"]}          # the four generalists: G4 31B, G4 26B, Q3.8, Q3.6


async def run_model(client, sem, agent, sample, out, deadline):
    cfg = config()
    f = out / f"{agent.model}.jsonl"
    done = {json.loads(line)["pid"] for line in f.open(encoding="utf-8")} if f.exists() else set()
    rows = [int(r) for r in sample.row if f"test:{int(r)}" not in done]

    async def one(row):
        if time.time() > deadline:
            return None
        rec = await play(client, sem, get_patient("test", row), [agent], rounds=0,
                         temperature=cfg["game"]["temperature"], base_seed=cfg["seed"])
        rec.update(subgroup="A", panel_id="opening_only", sample="main")
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    await asyncio.gather(*(one(r) for r in rows))


def summarise(out):
    import numpy as np
    s = {}
    for f in sorted(out.glob("*.jsonl")):
        recs = [json.loads(line) for line in f.open(encoding="utf-8")]
        y = np.array([r["correct"] for r in recs], float)
        lo, hi = bootstrap_ci(y)
        s[f.stem] = {"n": len(recs), "top1": round(y.mean(), 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "abstain": sum(r["final"][0]["dx"] is None for r in recs),
                     "cost_usd": round(sum((u.get("cost") or 0) for r in recs for u in r["usage"]), 4)}
    p2 = [s[m]["top1"] for m in AGENTS if m in s and s[m]["n"] == 294]
    if len(p2) == 4:
        s["P2_mean_of_members"] = round(sum(p2) / 4, 4)
    (out / "opening_only_summary.json").write_text(json.dumps(s, indent=1))
    return s


async def main(a):
    ok, why, st = budget.check()
    print("budget:", st, why)
    if not ok:
        return
    out = ROOT / "results" / "main" / "opening_only"
    out.mkdir(parents=True, exist_ok=True)
    sample = load_sample("main")
    sem = asyncio.Semaphore(a.concurrency)
    deadline = time.time() + a.budget
    async with httpx.AsyncClient() as client:
        try:
            await asyncio.wait_for(asyncio.gather(*(run_model(client, sem, AGENTS[m], sample, out, deadline)
                                                    for m in a.models.split(","))), timeout=a.budget + 15)
        except asyncio.TimeoutError:
            print("time budget reached; run again to resume")
    print(json.dumps(summarise(out), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma4_31b,qwen38_27b,gemma4_26b_moe,qwen36_27b")
    ap.add_argument("--budget", type=float, default=140)
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main(ap.parse_args()))
