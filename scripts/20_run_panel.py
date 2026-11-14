"""Run all 15 subgroups of one panel on a case sample. Resumable: finished games are skipped,
and every model call is cached, so an interrupted run can simply be started again.

usage: python scripts/20_run_panel.py --panel P1 [--sample main|check|trial] [--replica 0]
                                      [--budget 36000] [--max-spend 80]
Output: results/<out>/<panel>/<subgroup>.jsonl (one game per line) + MANIFEST.json + failures.log
  --out defaults to "main" for replica 0 and "main_r<k>" for repeats. trial/check samples are
  subsets of main, so their games land in results/main and are reused by the full run.

Spending stop: before the first game and after every --check-every finished games, the runner
reads OpenRouter's counters (src/ddx/budget.py) and stops starting new games when a limit is hit.
At most --games-in-flight games are open at once, so finished games are written steadily.
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
from ddx import budget, manifest  # noqa: E402
from ddx.data import get_patient  # noqa: E402
from ddx.game import play  # noqa: E402
from ddx.llm import config  # noqa: E402
from ddx.panels import PANELS, replicate, sg_name, subgroups  # noqa: E402
from ddx.sample import load_sample  # noqa: E402


def fresh_cost(rec) -> float:
    return sum((u.get("cost") or 0) for u in rec["usage"] if not u.get("cached"))


async def main(a):
    cfg = config()
    agents = replicate(PANELS[a.panel], a.replica)
    sample = load_sample(a.sample).head(a.n)
    split = sample.split.iloc[0]
    out_name = a.out or ("main" if a.replica == 0 else f"main_r{a.replica}")
    out = ROOT / "results" / out_name / a.panel
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.subgroups.split(",")) if a.subgroups else None
    todo = []
    for sg in subgroups(len(agents)):
        if only and sg_name(sg) not in only:
            continue
        f = out / f"{sg_name(sg)}.jsonl"
        done = {json.loads(line)["pid"] for line in f.open(encoding="utf-8")} if f.exists() else set()
        todo += [(sg, int(r)) for r in sample.row if f"{split}:{int(r)}" not in done]
    todo.sort(key=lambda t: (len(t[0]), t[0], t[1]))   # small subgroups first: cheap, usable early
    todo = todo[: a.max_games]
    print(f"{a.panel} r{a.replica} [{a.sample}] -> results/{out_name}: {len(todo)} games to play", flush=True)

    ok, why, st = budget.check()
    print(f"budget at start: {st} -> {why}", flush=True)
    if not ok:
        return
    t0 = time.time()
    state = {"finished": 0, "failed": 0, "spend": 0.0, "stop": None, "since_check": 0}
    sem = asyncio.Semaphore(a.concurrency)
    queue = list(reversed(todo))

    async with httpx.AsyncClient() as client:
        async def worker():
            while queue and state["stop"] is None:
                if time.time() - t0 > a.budget:
                    state["stop"] = state["stop"] or "time budget reached"
                    return
                sg, row = queue.pop()
                try:
                    rec = await play(client, sem, get_patient(split, row), [agents[i] for i in sg],
                                     rounds=cfg["game"]["rounds"], temperature=cfg["game"]["temperature"],
                                     base_seed=cfg["seed"], question_rule=a.question_rule)
                except Exception as ex:  # noqa: BLE001
                    state["failed"] += 1
                    msg = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {sg_name(sg)} {split}:{row} {ex!r:.300}"
                    print("game failed:", msg, file=sys.stderr, flush=True)
                    with (out / "failures.log").open("a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                    continue
                rec.update(subgroup=sg_name(sg), panel_id=a.panel, sample=a.sample,
                           ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                with (out / f"{sg_name(sg)}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                state["finished"] += 1
                state["spend"] += fresh_cost(rec)
                state["since_check"] += 1
                if state["since_check"] >= a.check_every:
                    state["since_check"] = 0
                    ok, why, st = await asyncio.to_thread(budget.check, state["spend"], a.max_spend)
                    print(f"  {state['finished']} done, run spend ${state['spend']:.2f}, "
                          f"{time.time() - t0:.0f}s, balance ${st.get('account_balance')}", flush=True)
                    if not ok:
                        state["stop"] = why
        workers = [asyncio.create_task(worker()) for _ in range(a.games_in_flight)]
        try:
            await asyncio.wait_for(asyncio.gather(*workers), timeout=a.budget + a.grace)
        except asyncio.TimeoutError:
            state["stop"] = state["stop"] or "time budget reached (in-flight games cut; their calls are cached)"

    info = {"seconds": round(time.time() - t0), "finished": state["finished"], "failed": state["failed"],
            "left": len(todo) - state["finished"], "new_spend_usd": round(state["spend"], 4),
            "stopped_because": state["stop"]}
    print(f"{a.panel}: {info}", flush=True)
    m = manifest.build(out, agents, a.sample, info)
    print(f"manifest: {m['games']['total']} games, fresh spend so far ${m['totals']['cost_fresh_usd']}, "
          f"cost/game ${m['totals']['cost_per_game_usd']}, truncated calls {m['totals']['truncated_calls']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=list(PANELS))
    ap.add_argument("--sample", default="main", help="main | check | trial (check, trial are subsets of main)")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subgroups", default=None, help="comma list, e.g. A,B,AB (default: all 15)")
    ap.add_argument("--budget", type=float, default=160, help="seconds; no new games start after this")
    ap.add_argument("--grace", type=float, default=10, help="extra seconds to let in-flight games finish")
    ap.add_argument("--max-spend", type=float, default=None, help="USD of new spend allowed in this call")
    ap.add_argument("--check-every", type=int, default=50, help="games between spending checks")
    ap.add_argument("--max-games", type=int, default=10_000_000)
    ap.add_argument("--games-in-flight", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=32, help="max simultaneous API calls")
    ap.add_argument("--question-rule", default="plurality", choices=["plurality", "rotate"],
                    help="how the asked question is selected (rotate = rotating proposer; use with --out rotate)")
    asyncio.run(main(ap.parse_args()))
