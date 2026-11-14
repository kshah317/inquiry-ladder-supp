"""Pilot: one generalist agent per model on the pilot sample (validation split).

Checks, per model: (a) accuracy is in a measurable range, (b) valid JSON / sensible questions,
(c) the capability match needed for P2, (d) the opening-only familiarity check (--rounds 0).

usage: python3 scripts/10_pilot.py --model gemma4_31b --n 49 [--rounds 5] [--role generalist]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.data import get_patient  # noqa: E402
from ddx.game import Agent, play  # noqa: E402
from ddx.llm import config  # noqa: E402
from ddx.sample import load_sample  # noqa: E402

MARKERS = pd.read_csv(ROOT / "prereg" / "marker_questions_draft.csv")
MARKERS = MARKERS[MARKERS.is_marker == 1].set_index("code").top_condition.to_dict()


def summarise(recs, model):
    m = config()["models"][model]
    tok_in = sum((u.get("prompt_tokens") or 0) for r in recs for u in r["usage"])
    tok_out = sum((u.get("completion_tokens") or 0) for r in recs for u in r["usage"])
    cost = sum((u.get("cost") or 0) for r in recs for u in r["usage"])
    n = len(recs)
    top3 = sum(r["truth"] in (r["final"][0]["top3"] or []) or r["correct"] for r in recs)
    hit = [any(c and MARKERS.get(c) == r["truth"] and rd.get("positive") for c, rd in zip(r["asked"], r["rounds"]))
           for r in recs]
    invalid_q = sum(not x["valid"] for r in recs for rd in r["rounds"] for x in rd["proposals"])
    n_q = sum(len(rd["proposals"]) for r in recs for rd in r["rounds"])
    return {
        "model": model, "n": n,
        "top1": round(sum(r["correct"] for r in recs) / n, 3),
        "top3": round(top3 / n, 3),
        "no_valid_dx": sum(r["final"][0]["dx"] is None for r in recs),
        "invalid_question_rate": round(invalid_q / max(n_q, 1), 3),
        "retries": sum(r["retries"] for r in recs),
        "share_rounds_answer_yes": round(sum(bool(rd.get("positive")) for r in recs for rd in r["rounds"]) / max(n_q, 1), 3),
        "marker_hit_rate": round(sum(hit) / n, 3),
        "top1_if_marker_hit": round(sum(r["correct"] for r, h in zip(recs, hit) if h) / max(sum(hit), 1), 3),
        "top1_if_no_hit": round(sum(r["correct"] for r, h in zip(recs, hit) if not h) / max(n - sum(hit), 1), 3),
        "reasoning_tokens": sum(((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) for r in recs for u in r["usage"]),
        "tokens_in": tok_in, "tokens_out": tok_out,
        "cost_usd_reported": round(cost, 4),
        "cost_usd_list_price": round(tok_in / 1e6 * m["price"][0] + tok_out / 1e6 * m["price"][1], 4),
    }


async def main(a):
    cfg = config()
    sample = load_sample("pilot").head(a.n)
    agent = Agent(name="Physician A", model=a.model, role=a.role, seed_id=f"pilot.{a.model}.{a.role}", full_record=a.full)
    out_dir = ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_{a.role}_r{a.rounds}_n{a.n}" + ("_full" if a.full else "")
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        async def one(row):
            p = get_patient("validate", int(row))
            return await play(client, sem, p, [agent], rounds=a.rounds,
                              temperature=cfg["game"]["temperature"], base_seed=cfg["seed"])
        recs = await asyncio.gather(*(one(r) for r in sample.row))
    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarise(recs, a.model)
    s["rounds"], s["role"], s["seconds"] = a.rounds, a.role, round(time.time() - t0)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--role", default="generalist")
    ap.add_argument("--full", action="store_true", help="agent sees the whole record (check-panel strong agent)")
    ap.add_argument("--concurrency", type=int, default=12)
    asyncio.run(main(ap.parse_args()))
