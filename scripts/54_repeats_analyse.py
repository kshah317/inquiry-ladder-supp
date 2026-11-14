"""Repeats analysis: replica 1 of P0/P1/P2 (sub-teams A, B, C, D, ABCD) and the pool over replicas 0+1.
No model calls. Output: results/analysis/repeats.json + REPEATS_REPORT.md

Replica 0 = main run (results/main/), replica 1 = results/main_r1/ (fresh seeds "#r1").
Pooled = per-patient mean over the two replicas (values 0, .5, 1); bootstrap resamples patients, so the two
replicas of a patient stay together. McNemar is reported per replica only (pooled outcomes are not 0/1).
Best member: chosen post hoc per replica (as in the main analysis), plus an out-of-sample version
(best chosen on replica 0, scored on replica 1)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import metrics as M  # noqa: E402
from ddx import stats as S  # noqa: E402

OUT = ROOT / "results" / "analysis"
PANELS = ["P0", "P1", "P2"]
FULL = M.FULL
SG = {(0,): "A", (1,): "B", (2,): "C", (3,): "D", FULL: "ABCD"}
MARGIN = 0.05


class Lite(M.Panel):
    """Panel with only the 4 solo games and the full panel (replica 1)."""
    def __init__(self, name, games):
        self.name, self.games = name, games
        self.pids = sorted(games["ABCD"])
        for k in SG.values():
            assert sorted(games[k]) == self.pids, (name, k)
        g0 = games["ABCD"][self.pids[0]]
        self.models, self.roles = g0["models"], g0["roles"]
        self.truth = np.array([games["ABCD"][p]["truth"] for p in self.pids])
        self.n = len(self.pids)


def load_r1(name):
    games = {}
    for k in SG.values():
        d = {}
        for line in (ROOT / f"results/main_r1/{name}/{k}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            assert r["replica"] == 1 and all(a["replica"] == 1 for a in r["agents"])
            d[r["pid"]] = M._slim(r)
        games[k] = d
    return Lite(name, games)


def r4(x):
    if isinstance(x, dict):
        return {str(k): r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def arrays(P):
    """Per-patient arrays for one replica."""
    return {"team": P.correct(FULL), "team_pre": P.correct(FULL, pre=True),
            "solo": np.stack([P.correct((i,)) for i in FULL]),
            "iv": M.indep_vote(P, FULL)[0],
            "team_cf": M.team_vote_expected(P, FULL), "iv_cf": M.indep_vote_expected(P, FULL)}


def block(a, idx, binary):
    solo = a["solo"]
    best = int(solo.mean(1).argmax())
    sb = np.stack([S.boot_means(solo[i], idx) for i in range(4)]).max(0)
    tb = S.boot_means(a["team"], idx)
    pc = lambda x, y: S.paired_contrast(x, y, idx, binary=binary)  # noqa: E731
    return {"team": S.mean_ci(a["team"], idx), "team_pre": float(a["team_pre"].mean()),
            "solo": [float(s.mean()) for s in solo], "best_member": "ABCD"[best],
            "team_minus_best": pc(a["team"], solo[best]),
            "team_minus_best_reselected": {"mean": float((tb - sb).mean()), "ci95": S.ci(tb - sb)},
            "indep_vote": float(a["iv"].mean()), "team_minus_indep_vote": pc(a["team"], a["iv"]),
            "coin_free": {"team": float(a["team_cf"].mean()), "indep_vote": float(a["iv_cf"].mean()),
                          "team_minus_indep_vote": S.paired_contrast(a["team_cf"], a["iv_cf"], idx, binary=False)},
            "delib_gain": pc(a["team"], a["team_pre"])}


def primary(A, idx, binary):
    pc = lambda x, y, m=MARGIN: S.paired_contrast(x, y, idx, margin=m, binary=binary)  # noqa: E731
    out = {"C1_P1_vs_P0": pc(A["P1"]["team"], A["P0"]["team"]),
           "C2a_P2_vs_P0": pc(A["P2"]["team"], A["P0"]["team"]),
           "C2b_P2_vs_indep_vote": pc(A["P2"]["team"], A["P2"]["iv"])}
    out["coin_free"] = {"C1": pc(A["P1"]["team_cf"], A["P0"]["team_cf"]),
                        "C2a": pc(A["P2"]["team_cf"], A["P0"]["team_cf"]),
                        "C2b": pc(A["P2"]["team_cf"], A["P2"]["iv_cf"])}
    if binary:
        out["holm_C2_mcnemar"] = S.holm({"C2a": out["C2a_P2_vs_P0"]["mcnemar"]["p"],
                                         "C2b": out["C2b_P2_vs_indep_vote"]["mcnemar"]["p"]})
    return out


def main():
    R0 = {k: M.load_panel(k) for k in PANELS}
    R1 = {k: load_r1(k) for k in PANELS}
    pids = R0["P0"].pids
    for k in PANELS:
        assert R0[k].pids == pids == R1[k].pids and (R0[k].truth == R1[k].truth).all()
    idx = S.boot_idx(len(pids))
    A0 = {k: arrays(R0[k]) for k in PANELS}
    A1 = {k: arrays(R1[k]) for k in PANELS}
    AP = {k: {f: (A0[k][f] + A1[k][f]) / 2 for f in A0[k]} for k in PANELS}
    res = {"n": len(pids), "B": int(idx.shape[0]), "margin": MARGIN,
           "replica0": {"panels": {k: block(A0[k], idx, True) for k in PANELS}, "primary": primary(A0, idx, True)},
           "replica1": {"panels": {k: block(A1[k], idx, True) for k in PANELS}, "primary": primary(A1, idx, True)},
           "pooled": {"panels": {k: block(AP[k], idx, False) for k in PANELS}, "primary": primary(AP, idx, False)},
           "stability": {}}
    for k in PANELS:
        b0 = int(A0[k]["solo"].mean(1).argmax())
        # how much do the two replicas agree?  r1 - r0 for the team, patient-level agreement,
        # and the out-of-sample best member (chosen on r0, scored on r1)
        same_q = np.mean([[g0["rounds"][j]["chosen"] for j in range(len(g0["rounds"]))] ==
                          [g1["rounds"][j]["chosen"] for j in range(len(g1["rounds"]))]
                          for g0, g1 in zip(R0[k].g(FULL), R1[k].g(FULL))])
        res["stability"][k] = {
            "team_r1_minus_r0": S.paired_contrast(A1[k]["team"], A0[k]["team"], idx),
            "team_agreement": float((A0[k]["team"] == A1[k]["team"]).mean()),
            "solo_r1_minus_r0": [float(A1[k]["solo"][i].mean() - A0[k]["solo"][i].mean()) for i in range(4)],
            "best_r0": "ABCD"[b0],
            "r1_team_minus_best_chosen_on_r0": S.paired_contrast(A1[k]["team"], A1[k]["solo"][b0], idx),
            "full_panel_same_question_sequence_share": float(same_q)}
    res = r4(res)
    (OUT / "repeats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report(res)


def report(res):
    p = lambda c: f"{100*c['diff']:+.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]"  # noqa: E731
    L = ["# Repeats: replica 1 and pooled (replicas 0+1)", "",
         f"n = {res['n']} patients; bootstrap B = {res['B']} over patients; pooled = per-patient mean of two replicas.", ""]
    for r, name in (("replica0", "Replica 0 (main run, for reference)"), ("replica1", "Replica 1"), ("pooled", "Pooled (0+1)")):
        L += [f"## {name}", "", "| Panel | Team | Solo A/B/C/D | Team − best | Team − best (reselected) | Indep. vote | Team − indep. vote | coin-free | Delib. gain |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, b in res[r]["panels"].items():
            rs = b["team_minus_best_reselected"]
            L.append(f"| {k} | {b['team']['mean']:.3f} | {'/'.join(f'{s:.3f}' for s in b['solo'])} | {p(b['team_minus_best'])} ({b['best_member']}) | "
                     f"{100*rs['mean']:+.1f} [{100*rs['ci95'][0]:+.1f}, {100*rs['ci95'][1]:+.1f}] | {b['indep_vote']:.3f} | {p(b['team_minus_indep_vote'])} | "
                     f"{p(b['coin_free']['team_minus_indep_vote'])} | {p(b['delib_gain'])} |")
        pr = res[r]["primary"]
        L += ["", "| Contrast | Diff, pts [95% CI] | 90% CI | TOST p (±5) | McNemar | coin-free |", "|---|---|---|---|---|---|"]
        for key, cf in (("C1_P1_vs_P0", "C1"), ("C2a_P2_vs_P0", "C2a"), ("C2b_P2_vs_indep_vote", "C2b")):
            c = pr[key]
            mc = f"{c['mcnemar']['a_only']} vs {c['mcnemar']['b_only']}, p={c['mcnemar']['p']:.3f}" if "mcnemar" in c else "—"
            L.append(f"| {key} | {p(c)} | [{100*c['ci90'][0]:+.1f}, {100*c['ci90'][1]:+.1f}] | {c['tost']['p_tost_t']:.3f} | {mc} | {p(pr['coin_free'][cf])} |")
        L.append("")
    L += ["## Stability across replicas", "", "| Panel | Team r1 − r0 | Patient agreement | Solo r1 − r0 (A/B/C/D) | r1 team − best chosen on r0 | Same question sequence |",
          "|---|---|---|---|---|---|"]
    for k, s in res["stability"].items():
        L.append(f"| {k} | {p(s['team_r1_minus_r0'])} | {s['team_agreement']:.3f} | {'/'.join(f'{100*x:+.1f}' for x in s['solo_r1_minus_r0'])} | "
                 f"{p(s['r1_team_minus_best_chosen_on_r0'])} ({s['best_r0']}) | {s['full_panel_same_question_sequence_share']:.3f} |")
    (OUT / "REPEATS_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
