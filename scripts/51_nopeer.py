"""No-peer reconsideration control.

For each full-panel game (sub-team ABCD) of P0, P1, P1Q, P2: every agent gets the same context as at
the real deliberation step (final-diagnosis request + its own final reply), but the last turn shows
NO peer answers (prompts.NOPEER_TASK). Same seed as the game's deliberation call, so only the prompt
differs. The team answer is re-voted with the game's tie rule; the coin generator is replayed from the
log to the exact state it had at the real final vote.

Built-in, free checks (cache only) for every game before any new call:
  - each agent's final request is a cache hit and gives the logged final answer
  - the REAL deliberation request, rebuilt from the log, is a cache hit and gives the logged revised answer
  - the replayed votes reproduce the logged question choices, pre-deliberation vote and final vote

usage: python3 scripts/51_nopeer.py --check            (cache-only checks, no spending)
       python3 scripts/51_nopeer.py --estimate
       python3 scripts/51_nopeer.py [--sample trial|main] [--panels P0,P1,P1Q,P2] [--budget 140]
Output: results/nopeer/<panel>.jsonl (one line per patient); resumable.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import budget, prompts  # noqa: E402
from ddx.cache import request_key  # noqa: E402
from ddx.data import get_patient, reveal  # noqa: E402
from ddx.game import _ask, _seed, _vote_info, match_condition, norm_conf  # noqa: E402
from ddx.llm import build_payload, cache, config, content, parse_json  # noqa: E402
from ddx.panels import PANELS  # noqa: E402
from ddx.sample import load_sample  # noqa: E402
from ddx.text import condition_display, qid_map, render_full_record, render_opening, render_reveal  # noqa: E402

OUT = ROOT / "results" / "nopeer"
CFG = config()
BASE, TEMP, ROUNDS = CFG["seed"], CFG["game"]["temperature"], CFG["game"]["rounds"]


class Miss(Exception):
    pass


def valid_dx(obj):
    if obj is None:
        return "not valid JSON"
    if match_condition(obj.get("diagnosis")) is None:
        return f"diagnosis {obj.get('diagnosis')!r} is not in the CONDITION LIST"
    return None


def cached_ask(agent, messages, seed):
    """Cache-only copy of game._ask: raises Miss instead of calling the API."""
    def get(msgs, s):
        hit = cache().get(request_key(build_payload(agent.model, msgs, temperature=TEMP, seed=s)))
        if hit is None:
            raise Miss(agent.seed_id)
        return content(hit)
    raw = get(messages, seed)
    obj = parse_json(raw)
    err = valid_dx(obj)
    if err is None:
        return obj, raw
    retry = messages + [{"role": "assistant", "content": raw},
                        {"role": "user", "content": f"Your answer was invalid: {err}. Reply again with one valid JSON object only."}]
    raw2 = get(retry, seed + 1)
    obj2 = parse_json(raw2)
    return (obj2 if valid_dx(obj2) is None else None), raw2


def dx_rec(obj):
    if obj is None:
        return None, 0.0
    try:
        c = float(obj.get("confidence", 0))
    except (TypeError, ValueError):
        c = 0.0
    return match_condition(obj.get("diagnosis")), c


def rebuild(panel, g):
    """Rebuild one full-panel game from its log; run all free checks. Returns the no-peer jobs + vote state."""
    agents = PANELS[panel]
    split, row = g["pid"].split(":")
    p = get_patient(split, int(row))
    opening = render_opening(p)
    full = render_full_record(p) if any(a.full_record for a in agents) else None
    transcript = [f"Round {e['k']} ({qid_map()[e['chosen']]}): " + render_reveal(reveal(p, e["chosen"]))
                  for e in g["rounds"] if e["chosen"]]
    # replay the coin generator through every earlier vote
    rng = random.Random(_seed(BASE, p.pid, "tie"))
    for a_, e in zip([None] * len(g["rounds"]), g["rounds"]):
        ch, _ = _vote_info([(x["code"], norm_conf(a.model, x["confidence"], "propose"))
                            for a, x in zip(agents, e["proposals"]) if x["valid"]], rng)
        assert ch == e["chosen"], ("round vote mismatch", panel, p.pid, e["k"])
    pre, _ = _vote_info([(f["dx"], norm_conf(a.model, f["confidence"], "final"))
                         for a, f in zip(agents, g["final"]) if f["dx"]], rng)
    assert pre == g["panel_pre"], ("pre vote mismatch", panel, p.pid)
    state = rng.getstate()
    real, _ = _vote_info([(f["dx"], norm_conf(a.model, f["confidence"], "final"))
                          for a, f in zip(agents, g["revised"]) if f["dx"]], rng)
    assert real == g["panel"], ("final vote mismatch", panel, p.pid)
    jobs = []
    for i, a in enumerate(agents):
        fm = prompts.final_messages(a.role, opening, transcript, ROUNDS, full if a.full_record else None)
        obj, raw = cached_ask(a, fm, _seed(BASE, a.sid, p.pid, "final"))
        assert dx_rec(obj)[0] == g["final"][i]["dx"], ("final mismatch", panel, p.pid, i)
        others = [f for j, f in enumerate(g["final"]) if j != i]
        peers = "\n".join(f"- {f['agent']}: {condition_display(f['dx']) if f['dx'] else 'no valid answer'}"
                          f" (confidence {f['confidence']:.0f}). {f['reasoning'] or ''}".rstrip() for f in others)
        dseed = _seed(BASE, a.sid, p.pid, "delib")
        robj, _ = cached_ask(a, prompts.delib_messages(fm, raw, peers), dseed)
        rdx = dx_rec(robj)[0] or g["final"][i]["dx"]
        assert rdx == g["revised"][i]["dx"], ("real delib mismatch", panel, p.pid, i)
        jobs.append((a, prompts.nopeer_messages(fm, raw), dseed))
    return p, jobs, state


async def main(a):
    panels = a.panels.split(",")
    keep = None
    if a.sample != "main":
        s = load_sample(a.sample)
        keep = {f"{sp}:{int(r)}" for sp, r in zip(s.split, s.row)}
    work, n_new = [], 0
    for name in panels:
        f = OUT / name / "ABCD.jsonl"
        done = {json.loads(x)["pid"] for x in f.open(encoding="utf-8")} if f.exists() else set()
        for line in (ROOT / "results" / "main" / name / "ABCD.jsonl").open(encoding="utf-8"):
            g = json.loads(line)
            if (keep and g["pid"] not in keep) or g["pid"] in done:
                continue
            p, jobs, state = rebuild(name, g)          # raises on any mismatch
            n_new += sum(cache().get(request_key(build_payload(ag.model, m, temperature=TEMP, seed=s))) is None
                         for ag, m, s in jobs)
            work.append((name, g, p, jobs, state))
    print(f"checks passed for {len(work)} games ({panels}, {a.sample}): finals, real deliberation and votes "
          f"reproduced from cache. No-peer calls not yet cached: {n_new} (~${n_new * 0.0011:.2f} + retries)", flush=True)
    if a.check or a.estimate:
        return
    ok, why, st = budget.check()
    print("budget at start:", st, why, flush=True)
    if not ok:
        return
    t0, spend, ndone, stop = time.time(), 0.0, 0, None
    sem = asyncio.Semaphore(a.concurrency)
    files = {}
    async with httpx.AsyncClient() as client:
        async def one(name, g, p, jobs, state):
            nonlocal spend, ndone
            res = await asyncio.gather(*(_ask(client, sem, ag, m, TEMP, s, valid_dx) for ag, m, s in jobs))
            agents = PANELS[name]
            np_ans, kept, us_all, rts = [], [], [], 0
            for i, (obj, raw, us, rt) in enumerate(res):
                dx, c = dx_rec(obj)
                if dx is None:                                   # game rule: keep pre-deliberation answer
                    dx, c = g["final"][i]["dx"], g["final"][i]["confidence"]
                    kept.append(i)
                np_ans.append({"dx": dx, "confidence": c, "reasoning": (obj or {}).get("reasoning")})
                us_all += [{k: u.get(k) for k in ("cost", "prompt_tokens", "completion_tokens", "finish_reason", "provider", "cached")} for u in us]
                rts += rt
            rng = random.Random()
            rng.setstate(state)
            panel_np, how = _vote_info([(x["dx"], norm_conf(ag.model, x["confidence"], "final"))
                                        for ag, x in zip(agents, np_ans) if x["dx"]], rng)
            rec = {"pid": g["pid"], "truth": g["truth"], "panel_id": name,
                   "models": [ag.model for ag in agents],
                   "final": [(f["dx"], f["confidence"]) for f in g["final"]],
                   "revised": [(f["dx"], f["confidence"]) for f in g["revised"]],
                   "nopeer": [(x["dx"], x["confidence"]) for x in np_ans],
                   "nopeer_reasoning": [x["reasoning"] for x in np_ans],
                   "kept_after_invalid": kept, "retries": rts,
                   "panel_pre": g["panel_pre"], "panel_real": g["panel"], "panel_np": panel_np, "vote_np": how,
                   "correct_pre": g["panel_pre"] == g["truth"], "correct_real": g["panel"] == g["truth"],
                   "correct_np": panel_np == g["truth"], "usage": us_all}
            fh = files.setdefault(name, (OUT / name).mkdir(parents=True, exist_ok=True) or (OUT / name / "ABCD.jsonl").open("a", encoding="utf-8"))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            spend += sum((u.get("cost") or 0) for u in us_all if not u.get("cached"))
            ndone += 1

        queue = list(work)

        async def worker():
            nonlocal stop
            while queue and stop is None:
                if time.time() - t0 > a.budget:
                    stop = "time budget reached"
                    return
                item = queue.pop(0)
                try:
                    await one(*item)
                except Exception as ex:  # noqa: BLE001
                    print("failed:", item[0], item[1]["pid"], repr(ex)[:200], file=sys.stderr, flush=True)
                if spend > a.max_spend:
                    stop = f"run spend ${spend:.2f} > --max-spend"
        ws = [asyncio.create_task(worker()) for _ in range(max(1, a.concurrency // 4))]
        try:
            await asyncio.wait_for(asyncio.gather(*ws), timeout=a.budget + 20)
        except asyncio.TimeoutError:
            stop = "cut at time budget (cached calls kept)"
    for fh in files.values():
        fh.close()
    print(f"done {ndone} games this call, {len(queue)} left, new spend ${spend:.3f}, {time.time()-t0:.0f}s, stop: {stop}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="P0,P1,P1Q,P2")
    ap.add_argument("--sample", default="main")
    ap.add_argument("--budget", type=float, default=120)
    ap.add_argument("--max-spend", type=float, default=10)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    asyncio.run(main(ap.parse_args()))
