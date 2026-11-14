"""P0-Q analyses fixed before its calls. Existing logs only, no API calls.
Output: results/analysis/p0q.json and P0Q_REPORT.md
"""
from __future__ import annotations
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddx import metrics as M, stats as S, nb
from ddx.metrics import load_panel, _slim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
SG = ["A", "B", "C", "D", "ABCD"]


def load_q(pids):
    games = {}
    for sg in SG:
        d = {}
        for line in open(ROOT / "results/main/P0Q" / f"{sg}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            d[r["pid"]] = _slim(r)
        assert set(pids) <= set(d), (sg, len(set(pids) - set(d)))
        games[sg] = {p: d[p] for p in pids}
    for d in games.values():
        for r in d.values():
            r["final"] = [tuple(x) for x in r["final"]]; r["revised"] = [tuple(x) for x in r["revised"]]
    return games


class Mini:                                   # just enough of metrics.Panel for indep_vote / error helpers
    def __init__(self, games, pids, truth):
        self.games, self.pids, self.truth, self.n = games, pids, truth, len(pids)
        g0 = games["ABCD"][pids[0]]
        self.models, self.roles = g0["models"], g0["roles"]
        self.name = "P0Q"

    def g(self, sg):
        key = sg if isinstance(sg, str) else "".join("ABCD"[i] for i in sg)
        return [self.games[key][p] for p in self.pids]

    def correct(self, sg, pre=False):
        return np.array([x["correct_pre" if pre else "correct"] for x in self.g(sg)], float)

    def solo_dx(self, i):
        return [x["final"][0] for x in self.g((i,))]


def pc(a, b, idx, margin=None):
    c = S.paired_contrast(a, b, idx, margin=margin)
    out = {"a": round(c["a"], 3), "b": round(c["b"], 3), "diff": round(100 * c["diff"], 2),
           "ci95": [round(100 * x, 2) for x in c["ci95"]], "ci90": [round(100 * x, 2) for x in c["ci90"]],
           "mcnemar": c.get("mcnemar")}
    if margin is not None:
        out["tost_p"] = round(c["tost"]["p_tost_t"], 4)
        out["equivalent_boot90"] = c["tost"]["equivalent_boot"]
    return out


P0, P1, P1Q, P2 = (load_panel(n) for n in ["P0", "P1", "P1Q", "P2"])
pids, truth = P0.pids, P0.truth
Q = Mini(load_q(pids), pids, truth)
idx = S.boot_idx(294)
F = (0, 1, 2, 3)
R = {}
team = Q.correct(F)
R["accuracy"] = {"team": round(float(team.mean()), 3), "team_pre": round(float(Q.correct(F, pre=True).mean()), 3),
                 "solo": [round(float(Q.correct((i,)).mean()), 3) for i in range(4)]}
# (1) C1-Q
R["C1Q_P1Q_minus_P0Q"] = pc(P1Q.correct(F), team, idx, margin=0.05)
# (2) family effect, same-model panels
R["P0Q_minus_P0"] = pc(team, P0.correct(F), idx)
# (3) role effect by family (difference in differences)
dd = (P1Q.correct(F) - team) - (P1.correct(F) - P0.correct(F))
R["role_effect_qwen_minus_gemma"] = {"diff": round(100 * dd.mean(), 2), "ci95": [round(100 * x, 2) for x in S.ci(S.boot_means(dd, idx))]}
# (4) team vs best member and vs independent vote
best = int(np.argmax([Q.correct((i,)).mean() for i in range(4)]))
iv, _ = M.indep_vote(Q, F)
R["team_minus_best_member"] = dict(pc(team, Q.correct((best,)), idx), best="ABCD"[best])
R["team_minus_indep_vote"] = dict(pc(team, iv, idx), indep_vote=round(float(iv.mean()), 3))
R["team_minus_indep_vote_coinfree"] = round(100 * float((team - M.indep_vote_expected(Q, F)).mean()), 2) if hasattr(M, "indep_vote_expected") else None
R["P1Q_minus_P0Q_indep_vote_note"] = "P1-Q team − indep vote is −2.4 (main report)"
# (5) error overlap, same-model Qwen
R["error_overlap"] = M.error_overlap(Q)
# (6) reader-free evidence value
asked = lambda g: [e["chosen"] for e in g["rounds"] if e["chosen"]]
nbv = {sg: np.array([nb.score(p, t, asked(g))[0] for p, t, g in zip(pids, truth, Q.g(sg))]) for sg in SG}
uni = np.array([nb.score(p, t, sum((asked(Q.games[k][p]) for k in "ABCD"), []))[0] for p, t in zip(pids, truth)])
mean_solo = np.mean([nbv[k] for k in "ABCD"], 0)
d = nbv["ABCD"] - mean_solo
R["nb"] = {"top1": {k: round(float(v.mean()), 3) for k, v in nbv.items()}, "union": round(float(uni.mean()), 3),
           "full_minus_mean_solo": {"diff": round(100 * d.mean(), 2), "ci95": [round(100 * x, 2) for x in S.ci(S.boot_means(d, idx))]}}
# (7) deliberation
R["deliberation"] = M.deliberation(Q, [F]) if True else None
# question votes and transcript diversity in the full panel
gs = Q.g(F)
from collections import Counter
nprop = Counter(sum(1 for x in e["props"] if x == e["chosen"]) for g in gs for e in g["rounds"] if e["chosen"])
tot = sum(nprop.values())
R["qvote_proposers_of_chosen"] = {k: round(v / tot, 3) for k, v in sorted(nprop.items())}
R["distinct_proposals_per_round"] = round(float(np.mean([len({x for x in e["props"] if x}) for g in gs for e in g["rounds"]])), 2)
# run statistics and cost
R["run"] = {sg: {"games": len(Q.games[sg]), "cost_mean": round(float(np.mean([g["cost"] for g in Q.g(sg)])), 5),
                 "truncated": int(sum(g["truncated"] for g in Q.g(sg))), "retries": int(sum(g["retries"] for g in Q.g(sg))),
                 "abstained_props": int(sum(1 for g in Q.g(sg) for e in g["rounds"] for x in e["props"] if x is None))} for sg in SG}
(OUT / "p0q.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
print(json.dumps(R, indent=1, default=float))
