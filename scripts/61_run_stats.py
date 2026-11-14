"""Run statistics for appendix Table A.3, from the raw game logs (no model calls).
usage: python3 scripts/61_run_stats.py [main|main_r1]
Output: results/analysis/run_stats.json (run_stats_main_r1.json for the repeats) + markdown table printed."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = sys.argv[1] if len(sys.argv) > 1 else "main"          # "main" or "main_r1" (repeats)
out, lines = {}, []
for p in ["P0", "P1", "P1Q", "P2", "CHECK", "ANCHOR"]:
    d = ROOT / "results" / RUN / p
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files:
        continue
    games = calls = trunc = abst_q = abst_dx = 0
    cost = 0.0
    pids = set()
    tr_by = Counter()
    retr = Counter()
    nret = Counter()
    pair_votes = Counter()
    for f in files:
        for line in f.open(encoding="utf-8"):
            g = json.loads(line)
            games += 1
            pids.add(g["pid"])
            for u in g["usage"]:
                calls += 1
                cost += u.get("cost") or 0
                if u.get("finish_reason") == "length":
                    trunc += 1
                    tr_by[u.get("model")] += 1
            names = [a["name"] for a in g["agents"]]
            for i, n in enumerate(names):
                retr[n] += g["retries_by_agent"].get(n, 0)
                nret[n] += 1
            for e in g["rounds"]:
                abst_q += sum(not x["valid"] for x in e["proposals"])
                if len(names) == 2:
                    pair_votes[e["vote"]] += 1
            abst_dx += sum(f_["dx"] is None for f_ in g["revised"])
    n_pat = len(pids)
    pv = sum(pair_votes.values())
    rec = {"games": games, "patients": n_pat, "calls": calls, "cost_total_usd": round(cost, 2),
           "cost_per_patient_all_subteams": round(cost / n_pat, 4), "truncated_calls": trunc,
           "truncated_by_model": dict(tr_by), "truncated_share": trunc / calls,
           "retries_per_game_by_agent": {k: round(retr[k] / nret[k], 2) for k in retr},
           "abstained_proposals": abst_q, "final_answers_missing": abst_dx,
           "pair_question_votes": {k: round(v / pv, 3) for k, v in pair_votes.items()} if pv else {}}
    out[p] = rec
    pvs = rec["pair_question_votes"]
    trs = ", ".join(f"{k} {v}" for k, v in tr_by.items())
    lines.append(f"| {p} | {games:,} | {n_pat} | ${rec['cost_per_patient_all_subteams']:.3f} | {trunc}"
                 + (f" ({trs}; {100*trunc/calls:.1f}% of calls)" if trunc else "")
                 + f" | {', '.join(f'{k} {v}' for k, v in rec['retries_per_game_by_agent'].items())} | {abst_q} / {abst_dx} | "
                 + (f"{100*pvs.get('majority',0):.0f}% / {100*pvs.get('confidence',0):.0f}% / {100*pvs.get('coin',0):.0f}%" if pvs else "-") + " |")
(ROOT / "results" / "analysis" / ("run_stats.json" if RUN == "main" else f"run_stats_{RUN}.json")).write_text(json.dumps(out, indent=1))
print("\n".join(lines))
