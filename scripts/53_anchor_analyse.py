"""Anchor analysis. No model calls.
Output: results/analysis/anchor.json + ANCHOR_REPORT.md"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import stats as S  # noqa: E402

AN = {json.loads(x)["pid"]: json.loads(x) for x in (ROOT / "results/main/ANCHOR/A.jsonl").open(encoding="utf-8")}
pids = sorted(AN)
assert len(pids) == 294
D = json.loads((ROOT / "results/analysis/analysis.json").read_text())
idx = S.boot_idx(len(pids))
a = np.array([AN[p]["correct"] for p in pids], float)


def load(panel, sg):
    return {json.loads(x)["pid"]: json.loads(x) for x in (ROOT / f"results/main/{panel}/{sg}.jsonl").open(encoding="utf-8")}


def contrast(b):
    d = a - b
    lo, hi = S.ci(S.boot_means(d, idx))
    return {"anchor": a.mean(), "other": b.mean(), "diff_pts": 100 * d.mean(), "ci95": [100 * lo, 100 * hi],
            "mcnemar": S.mcnemar(a, b)}


usage = [u for p in pids for u in AN[p]["usage"]]
cost_game = np.array([sum((u.get("cost") or 0) for u in AN[p]["usage"]) for p in pids])
trunc_game = np.array([sum(u.get("finish_reason") == "length" for u in AN[p]["usage"]) for p in pids])
out = {"n": 294, "acc": {"mean": a.mean(), "ci95": list(S.ci(S.boot_means(a, idx)))},
       "cost_per_consultation_usd": float(cost_game.mean()),
       "calls": len(usage), "truncated_calls": int(trunc_game.sum()), "truncated_share": float(trunc_game.sum() / len(usage)),
       "reasoning_tokens_mean": float(np.mean([u.get("completion_tokens") or 0 for u in usage])),
       "retries": int(sum(AN[p]["retries"] for p in pids)),
       "abstained_proposals": int(sum(not x["valid"] for p in pids for e in AN[p]["rounds"] for x in e["proposals"])),
       "final_missing": int(sum(f["dx"] is None for p in pids for f in AN[p]["final"])),
       "games_with_lt5_questions": int(sum(sum(1 for e in AN[p]["rounds"] if e["chosen"]) < 5 for p in pids)),
       "acc_by_truncations_exploratory": {k: {"n": int(m.sum()), "acc": float(a[m].mean())} for k, m in
                                          (("0", trunc_game == 0), ("1", trunc_game == 1), ("2+", trunc_game >= 2))},
       "vs": {}}
for panel in ["P0", "P1", "P2", "P1Q"]:
    full = load(panel, "ABCD")
    t = np.array([full[p]["correct"] for p in pids], float)
    best = D["panels"][panel]["best_member"]
    solo = load(panel, best)
    s = np.array([solo[p]["correct"] for p in pids], float)
    tc = D["cost"][panel]["per_consultation_usd"]
    out["vs"][panel] = {"team": contrast(t), "best_member_" + best: contrast(s),
                        "team_cost_usd": tc, "cost_ratio_anchor_over_team": out["cost_per_consultation_usd"] / tc,
                        "solo_cost_usd": D["cost"][panel]["per_subgroup_usd"][best]}   # the named member's own solo cost
(ROOT / "results/analysis/anchor.json").write_text(json.dumps(out, indent=1, default=float))
f = lambda c: f"{c['diff_pts']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}], McNemar {c['mcnemar']['a_only']} vs {c['mcnemar']['b_only']}, p={c['mcnemar']['p']:.3f}"  # noqa: E731
L = ["# Anchor: Qwen3.8-Max alone", "",
     f"Top-1 {a.mean():.3f} [{out['acc']['ci95'][0]:.3f}, {out['acc']['ci95'][1]:.3f}], n = 294. Cost ${out['cost_per_consultation_usd']:.3f} per consultation.",
     f"Truncated at 8,000 tokens: {out['truncated_calls']} of {out['calls']} calls ({100*out['truncated_share']:.0f}%); retries {out['retries']}; "
     f"abstained proposals {out['abstained_proposals']}; no final answer {out['final_missing']}; games with < 5 questions {out['games_with_lt5_questions']}.",
     "Accuracy by truncated calls in the game (exploratory, confounded by case difficulty): " +
     ", ".join(f"{k}: {v['acc']:.3f} (n={v['n']})" for k, v in out["acc_by_truncations_exploratory"].items()), "",
     "| Comparison | Anchor − other, pts [95% CI] | Other's cost per consultation | Anchor cost ÷ other |", "|---|---|---|---|"]
for panel, v in out["vs"].items():
    L.append(f"| {panel} team | {f(v['team'])} | ${v['team_cost_usd']:.3f} | {v['cost_ratio_anchor_over_team']:.1f}× |")
    k = [x for x in v if x.startswith("best_member_")][0]
    L.append(f"| {panel} best member ({k[-1]}) | {f(v[k])} | ${v['solo_cost_usd']:.4f} | {out['cost_per_consultation_usd']/v['solo_cost_usd']:.0f}× |")
(ROOT / "results/analysis/ANCHOR_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
