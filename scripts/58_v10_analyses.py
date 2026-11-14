"""Analyses added for the submitted draft. Existing logs only, no API calls.

  matched     Table 2: team, members, mean member (query-matched), independent vote (call-matched)
  shapley     Shapley values with chance (1/49) as the empty-coalition value, spread and noise floor
  by_size     accuracy by coalition size (Appendix A.5)
  retention   who proposed the asked questions in the full-team episodes (Section 5.3)
  vote_fail   three-way split of patients where the independent vote is wrong (Appendix A.5)
  reference   Qwen3.8-Max minus the P0-Q team, and the persona contrast on Qwen without truncated calls

Output: results/analysis/v10.json
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddx import metrics as M, stats as S
from ddx.metrics import load_panel, _slim, CHANCE, FULL, SUBGROUPS
from ddx.panels import sg_name

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
r1 = lambda x: round(100 * float(x), 1)
r3 = lambda x: round(float(x), 3)


class Mini:
    """P0-Q has only A, B, C, D and ABCD, so it cannot use metrics.Panel."""
    def __init__(self, name, pids, truth):
        self.name, self.pids, self.truth, self.n = name, pids, truth, len(pids)
        self.games = {}
        for sg in ["A", "B", "C", "D", "ABCD"]:
            d = {}
            for line in open(ROOT / "results/main" / name / f"{sg}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                d[r["pid"]] = _slim(r)
            self.games[sg] = {p: d[p] for p in pids}
        for d in self.games.values():
            for r in d.values():
                r["final"] = [tuple(x) for x in r["final"]]
                r["revised"] = [tuple(x) for x in r["revised"]]
        self.models = self.games["ABCD"][pids[0]]["models"]

    def g(self, sg):
        return [self.games[sg if isinstance(sg, str) else sg_name(sg)][p] for p in self.pids]

    def correct(self, sg, pre=False):
        return np.array([x["correct_pre" if pre else "correct"] for x in self.g(sg)], float)

    def solo_dx(self, i):
        return [x["final"][0] for x in self.g((i,))]


def diff(a, b, idx):
    c = S.paired_contrast(a, b, idx, binary=False)
    return {"diff": r1(c["diff"]), "ci95": [r1(x) for x in c["ci95"]]}


def matched(P, idx):
    solo = np.stack([P.correct((i,)) for i in FULL])
    team, mean_member = P.correct(FULL), solo.mean(0)
    vote, _ = M.indep_vote(P, FULL)
    return {"team": r3(team.mean()), "members": [r3(x) for x in solo.mean(1)],
            "mean_member": r3(mean_member.mean()),
            "team_minus_mean_member": diff(team, mean_member, idx),
            "indep_vote": r3(vote.mean()),
            "vote_minus_mean_member": diff(vote, mean_member, idx),
            "team_minus_vote": {**diff(team, vote, idx), "mcnemar_p": S.mcnemar(team, vote)["p"]}}


def shapley(P, idx):
    phi = M.shapley_values(P, CHANCE)
    m = phi.mean(1)
    mask = M.masking_shapley(P, CHANCE).mean(1)
    return {"values": [r3(x) for x in m], "ci95": [[r3(x) for x in S.ci(S.boot_means(p, idx))] for p in phi],
            "spread": r3(m.max() - m.min()), "masking": [r3(x) for x in mask]}


def by_size(P):
    acc = {k: float(np.mean([P.correct(sg).mean() for sg in SUBGROUPS if len(sg) == k])) for k in (1, 2, 3, 4)}
    return {"mean_accuracy": {k: r3(v) for k, v in acc.items()},
            "fourth_minus_mean_of_three": r1(acc[4] - acc[3])}


def retention(P):
    top, share, n_share, how = [], 0, 0, Counter()
    for g in P.g(FULL):
        props = [{i for i, c in enumerate(e["props"]) if c is not None and c == e["chosen"]}
                 for e in g["rounds"] if e["chosen"]]
        how.update(e["vote"] for e in g["rounds"] if e["chosen"])
        top.append(max(sum(i in s for s in props) for i in FULL))
        for a, b in zip(props, props[1:]):
            share += bool(a & b)
            n_share += 1
    tot = sum(how.values())
    return {"asked_by_most_frequent_proposer": round(float(np.mean(top)), 2),
            "shares_proposer_with_previous": round(share / n_share, 4), "n_linked": [share, n_share],
            "question_rounds_by_resolution": {k: r3(v / tot) for k, v in how.items()}}


def vote_fail(P):
    vote, _ = M.indep_vote(P, FULL)
    solos = [P.solo_dx(i) for i in FULL]
    out = Counter()
    for k in np.where(vote == 0)[0]:
        dx = [solos[i][k][0] for i in FULL if solos[i][k][0]]
        c = Counter(dx)
        if P.truth[k] not in c:
            out["no_member_right"] += 1
        elif c[P.truth[k]] < max(c.values()):
            out["correct_minority_outvoted"] += 1
        else:
            out["tie_lost"] += 1
    return {"vote_wrong": int((vote == 0).sum()), **out}


def main():
    P = {k: load_panel(k) for k in ["P0", "P1", "P1Q", "P2", "CHECK"]}
    pids = P["P0"].pids
    P["P0Q"] = Mini("P0Q", pids, P["P0"].truth)
    idx = S.boot_idx(len(pids))
    main5 = ["P0", "P1", "P0Q", "P1Q", "P2"]
    out = {"matched": {k: matched(P[k], idx) for k in main5},
           "shapley_chance": {k: shapley(P[k], S.boot_idx(P[k].n)) for k in ["P0", "P1", "P1Q", "P2", "CHECK"]},
           "by_size": {k: by_size(P[k]) for k in ["P0", "P1", "P1Q", "P2"]},
           "retention": {k: retention(P[k]) for k in main5},
           "vote_fail": {k: vote_fail(P[k]) for k in main5}}

    anchor = {}
    for line in open(ROOT / "results/main/ANCHOR/A.jsonl", encoding="utf-8"):
        r = json.loads(line)
        anchor[r["pid"]] = float(bool(r["correct"]))
    a = np.array([anchor[p] for p in pids])
    c = S.paired_contrast(a, P["P0Q"].correct(FULL), idx)
    out["reference_minus_P0Q"] = {"diff": r1(c["diff"]), "ci95": [r1(x) for x in c["ci95"]], "mcnemar_p": c["mcnemar"]["p"]}

    # truncated calls are counted over the five played coalitions (A, B, C, D, ABCD), and the
    # subset keeps patients with no truncated call in the full-team episode of either team
    trunc = lambda Q, sgs: np.array([sum(g["truncated"] for g in gs) for gs in zip(*[Q.g(s) for s in sgs])])
    five = ["A", "B", "C", "D", "ABCD"]
    clean = (trunc(P["P1Q"], ["ABCD"]) == 0) & (trunc(P["P0Q"], ["ABCD"]) == 0)
    d = (P["P1Q"].correct(FULL) - P["P0Q"].correct(FULL))[clean]
    out["P1Q_minus_P0Q_no_truncated_call"] = {
        "n": int(clean.sum()), "diff": r1(d.mean()), "ci95": [r1(x) for x in S.ci(S.boot_means(d, S.boot_idx(len(d))))],
        "truncated_calls": {"P1Q": int(trunc(P["P1Q"], five).sum()), "P0Q": int(trunc(P["P0Q"], five).sum())}}
    (OUT / "v10.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
