"""Replay test (asking vs reading): a FIXED reader diagnoses from each sub-team's question transcript.

For every game already played (panel x sub-team x patient), rebuild the 5-question transcript exactly
as the panel saw it, and ask one fixed reader - the Gemma 4 31B generalist (seed identity G4.gen.a) -
for a final diagnosis with the game's own final-diagnosis prompt, temperature, seed and retry rule.
No deliberation, no peers. Then
    e(S) = reader right on S's transcript      -> "evidence value" of what S chose to ask
    v(S) = S's own team answer (from the game) -> answer value
Evidence Shapley uses e(S); comparing e and v separates ASKING (e) from READING + VOTING (v - e).

Design choices (fixed before running):
- Reader seed depends on the patient only (= the solo final-diagnosis seed of G4.gen.a). So the
  reader's request is a function of (patient, transcript): identical transcripts give one identical
  request (asked once, cached), and on G4.gen.a's own solo transcripts the request is byte-identical to
  its solo game's final call -> served from cache, reader answer = that game's answer (built-in check).
- Invalid output: one retry with the reason, then abstain (counted wrong), as in the game.

usage: python3 scripts/50_replay.py [--panels P0,P1,P2,P1Q,CHECK] [--sample main|trial] [--budget 150]
       python3 scripts/50_replay.py --estimate      (count new calls, no spending)
Output: results/replay/<panel>/<subgroup>.jsonl; resumable (finished transcripts are skipped).
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
from ddx import budget, prompts  # noqa: E402
from ddx import metrics as M  # noqa: E402
from ddx.data import get_patient, reveal  # noqa: E402
from ddx.game import _ask, _seed, _usage, match_condition  # noqa: E402
from ddx.llm import build_payload, cache, config  # noqa: E402
from ddx.cache import request_key  # noqa: E402
from ddx.panels import PANELS, sg_name  # noqa: E402
from ddx.sample import load_sample  # noqa: E402
from ddx.text import qid_map, render_opening, render_reveal  # noqa: E402

READER = PANELS["P0"][0]                 # Gemma 4 31B generalist, seed identity G4.gen.a
assert READER.seed_id == "G4.gen.a" and READER.role == "generalist"
OUT = ROOT / "results" / "replay"


def transcript(p, asked):
    """Same text the game built: one line per round that asked a question (k = round number)."""
    return [f"Round {k} ({qid_map()[c]}): " + render_reveal(reveal(p, c)) for k, c in enumerate(asked, 1) if c]


def reader_request(p, asked, cfg):
    msgs = prompts.final_messages(READER.role, render_opening(p), transcript(p, asked), cfg["game"]["rounds"])
    seed = _seed(cfg["seed"], READER.sid, p.pid, "final")
    return msgs, seed


def valid_dx(obj):
    if obj is None:
        return "not valid JSON"
    if match_condition(obj.get("diagnosis")) is None:
        return f"diagnosis {obj.get('diagnosis')!r} is not in the CONDITION LIST"
    return None


def jobs(panels, pids_keep):
    """(panel, sg, pid, asked) for every game, grouped by unique (pid, asked)."""
    out, uniq = [], {}
    for name in panels:
        P = M.load_panel(name)
        for sg in M.SUBGROUPS:
            f = OUT / name / f"{sg_name(sg)}.jsonl"
            done = {json.loads(line)["pid"] for line in f.open(encoding="utf-8")} if f.exists() else set()
            for g in P.g(sg):
                if g["pid"] in done or (pids_keep and g["pid"] not in pids_keep):
                    continue
                asked = tuple(e["chosen"] for e in g["rounds"])
                out.append((name, sg_name(sg), g["pid"], asked, g["truth"]))
                uniq.setdefault((g["pid"], asked), []).append(len(out) - 1)
    return out, uniq


async def main(a):
    cfg = config()
    panels = a.panels.split(",")
    keep = None
    if a.sample != "main":
        s = load_sample(a.sample)
        keep = {f"{sp}:{int(r)}" for sp, r in zip(s.split, s.row)}
    todo, uniq = jobs(panels, keep)
    # how many of the unique reader requests are already in the cache?
    cached = 0
    for (pid, asked) in uniq:
        msgs, seed = reader_request(M._patient(pid), asked, cfg)
        if cache().get(request_key(build_payload(READER.model, msgs, temperature=cfg["game"]["temperature"], seed=seed))):
            cached += 1
    print(f"replay {panels} [{a.sample}]: {len(todo)} games to score, {len(uniq)} unique transcripts, "
          f"{cached} already cached -> ~{len(uniq) - cached} new calls (~${(len(uniq) - cached) * 0.00095:.2f} + retries)", flush=True)
    if a.estimate:
        return
    ok, why, st = budget.check()
    print("budget at start:", st, why, flush=True)
    if not ok:
        return
    t0 = time.time()
    state = {"done": 0, "spend": 0.0, "stop": None}
    sem = asyncio.Semaphore(a.concurrency)
    queue = list(uniq.items())
    files = {}

    def write(i, rec):
        name, sg, pid, asked, truth = todo[i]
        f = files.setdefault((name, sg), (OUT / name).mkdir(parents=True, exist_ok=True) or (OUT / name / f"{sg}.jsonl").open("a", encoding="utf-8"))
        f.write(json.dumps({"pid": pid, "truth": truth, "panel_id": name, "subgroup": sg, "asked": list(asked),
                            **rec, "correct": rec["dx"] == truth}, ensure_ascii=False) + "\n")
        f.flush()

    async with httpx.AsyncClient() as client:
        async def worker():
            while queue and state["stop"] is None:
                if time.time() - t0 > a.budget:
                    state["stop"] = "time budget reached"
                    return
                (pid, asked), idxs = queue.pop()
                p = M._patient(pid)
                msgs, seed = reader_request(p, asked, cfg)
                try:
                    obj, raw, us, rt = await _ask(client, sem, READER, msgs, cfg["game"]["temperature"], seed, valid_dx)
                except Exception as ex:  # noqa: BLE001
                    print("failed:", pid, repr(ex)[:200], file=sys.stderr, flush=True)
                    continue
                rec = {"dx": match_condition(obj.get("diagnosis")) if obj else None,
                       "confidence": float(obj.get("confidence", 0) or 0) if obj else 0.0,
                       "retries": rt, "usage": [{k: u.get(k) for k in ("cost", "prompt_tokens", "completion_tokens", "finish_reason", "provider", "cached")} for u in us]}
                for i in idxs:
                    write(i, rec)
                state["done"] += 1
                state["spend"] += sum((u.get("cost") or 0) for u in us if not u.get("cached"))
                if state["done"] % 500 == 0:
                    ok, why, st = await asyncio.to_thread(budget.check, state["spend"], a.max_spend)
                    print(f"  {state['done']} transcripts, new spend ${state['spend']:.2f}, {time.time()-t0:.0f}s, {st}", flush=True)
                    if not ok:
                        state["stop"] = why
        ws = [asyncio.create_task(worker()) for _ in range(a.concurrency)]
        try:
            await asyncio.wait_for(asyncio.gather(*ws), timeout=a.budget + 15)
        except asyncio.TimeoutError:
            state["stop"] = "cut at time budget (cached calls kept)"
    for f in files.values():
        f.close()
    left = len(queue)
    print(f"done {state['done']} transcripts this call, {left} left, new spend ${state['spend']:.3f}, "
          f"{time.time()-t0:.0f}s, stop: {state['stop']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="P0,P1,P2,P1Q,CHECK")
    ap.add_argument("--sample", default="main")
    ap.add_argument("--budget", type=float, default=140)
    ap.add_argument("--max-spend", type=float, default=15)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--estimate", action="store_true")
    asyncio.run(main(ap.parse_args()))
