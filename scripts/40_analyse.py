"""Full analysis of the main runs (no model calls). Replaces scripts/90_preview_primary.py.

usage: python3 scripts/40_analyse.py [--B 10000] [--perm 2000]
Writes results/analysis/analysis.json (every number) and results/analysis/ANALYSIS_REPORT.md (tables).
Bootstrap: B resamples of patients, one shared index matrix per sample size (seed 20261120).
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import metrics as M  # noqa: E402
from ddx import stats as S  # noqa: E402
from ddx.panels import PAIR_TYPE, sg_name  # noqa: E402

OUT = ROOT / "results" / "analysis"
PANELS = ["P0", "P1", "P1Q", "P2", "CHECK"]
FULL = M.FULL
MARGIN = 0.05


def r3(x):
    if isinstance(x, dict):
        return {str(k): r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def panel_block(P, idx):
    """Team vs members vs independent vote, credit (Shapley both baselines, LOO, masking), sub-team synergy."""
    team = P.correct(FULL)
    solo = np.stack([P.correct((i,)) for i in FULL])
    best = int(solo.mean(1).argmax())
    iv, _ = M.indep_vote(P, FULL)
    # best member re-selected inside every bootstrap draw (sensitivity for the same-data selection bias)
    sb = np.stack([S.boot_means(solo[i], idx) for i in FULL]).max(0)
    tb = S.boot_means(team, idx)
    out = {
        "n": P.n, "team": S.mean_ci(team, idx), "team_pre_delib": float(P.correct(FULL, pre=True).mean()),
        "solo": [float(s.mean()) for s in solo], "best_member": "ABCD"[best],
        "team_minus_best": S.paired_contrast(team, solo[best], idx),
        "team_minus_best_reselected_ci95": S.ci(tb - sb), "team_minus_best_reselected_mean": float((tb - sb).mean()),
        "team_minus_mean_member": S.paired_contrast(team, solo.mean(0), idx, binary=False),
        "indep_vote": float(iv.mean()),
        "team_minus_indep_vote": S.paired_contrast(team, iv, idx),
        "indep_vote_minus_best": S.paired_contrast(iv, solo[best], idx),
        "oracle_any_solo_right": float(solo.max(0).mean()),
    }
    # coin averaged out (sensitivity): exact expectation over the final-vote coin, team and independent vote
    ive, tve = M.indep_vote_expected(P, FULL), M.team_vote_expected(P, FULL)
    out["coin_free"] = {"indep_vote": float(ive.mean()), "team": float(tve.mean()),
                        "team_minus_indep_vote": S.paired_contrast(tve, ive, idx, binary=False)}
    v0b = M.v0_opening(P)
    out["v0"] = {"chance": M.CHANCE, "opening_only": v0b}
    for tag, v0 in [("chance", M.CHANCE), ("opening_only", v0b)]:
        phi = M.shapley_values(P, v0)
        out[f"shapley_{tag}"] = [S.mean_ci(phi[i], idx) for i in FULL]
        assert np.allclose(phi.sum(0), team - v0)          # efficiency, per patient
    lo = M.loo(P)
    out["loo"] = [S.mean_ci(lo[i], idx) for i in FULL]
    mk = M.masking_shapley(P)
    out["masking_shapley_chance"] = [S.mean_ci(mk[i], idx) for i in FULL]
    phi = M.shapley_values(P, M.CHANCE)
    # max-min spread of Shapley (credit concentration), with CI
    spread = S.boot_means(phi, idx)
    out["shapley_spread"] = {"mean": float(phi.mean(1).max() - phi.mean(1).min()),
                             "ci95": S.ci(spread.max(0) - spread.min(0))}
    # every sub-team of 2+: team - best member, team - its members' independent vote
    by_size = {}
    for k in (2, 3, 4):
        sgs = [sg for sg in M.SUBGROUPS if len(sg) == k]
        d_best, d_iv = [], []
        for sg in sgs:
            t = P.correct(sg)
            b = max(sg, key=lambda i: solo[i].mean())
            d_best.append(t - solo[b])
            d_iv.append(t - M.indep_vote(P, sg)[0])
        db, di = np.mean(d_best, 0), np.mean(d_iv, 0)
        by_size[k] = {"n_subteams": len(sgs), "team_minus_best": S.mean_ci(db, idx),
                      "team_minus_indep_vote": S.mean_ci(di, idx),
                      "team_acc": float(np.mean([P.correct(sg).mean() for sg in sgs]))}
    out["subteams_by_size"] = by_size
    # pairs, by pair type
    pairs = {}
    types = PAIR_TYPE.get(P.name, {pr: "check" for pr in combinations(FULL, 2)})
    for pr in combinations(FULL, 2):
        t = P.correct(pr)
        b = max(pr, key=lambda i: solo[i].mean())
        ivp = M.indep_vote(P, pr)[0]
        pairs[sg_name(pr)] = {"type": types[pr], "team": float(t.mean()), "best_member": float(solo[b].mean()),
                              "synergy": float((t - solo[b]).mean()), "minus_indep_vote": float((t - ivp).mean()),
                              "_d": t - solo[b], "_div": t - ivp}
    pt = {}
    for ty in sorted(set(types.values())):
        ps = [v for v in pairs.values() if v["type"] == ty]
        pt[ty] = {"n_pairs": len(ps), "synergy": S.mean_ci(np.mean([v["_d"] for v in ps], 0), idx),
                  "minus_indep_vote": S.mean_ci(np.mean([v["_div"] for v in ps], 0), idx)}
    for v in pairs.values():
        v.pop("_d"); v.pop("_div")
    out["pairs"], out["pair_types"] = pairs, pt
    out["vote_resolution_full"] = M.vote_resolution(P, FULL)
    out["vote_resolution_pairs"] = M.vote_resolution(P, (0, 1))
    return out


def primary(Ps, idx, mask=None):
    """C1: P1 vs P0 (TOST, margin 5 pts). C2a: P2 vs P0. C2b: P2 vs its members' independent vote."""
    sel = (lambda x: x) if mask is None else (lambda x: x[mask])
    p0, p1, p2 = (sel(Ps[k].correct(FULL)) for k in ("P0", "P1", "P2"))
    iv2 = sel(M.indep_vote(Ps["P2"], FULL)[0])
    out = {"C1_P1_vs_P0": S.paired_contrast(p1, p0, idx, margin=MARGIN),
           "C2a_P2_vs_P0": S.paired_contrast(p2, p0, idx, margin=MARGIN),
           "C2b_P2_vs_indep_vote": S.paired_contrast(p2, iv2, idx, margin=MARGIN)}
    out["holm_C2_mcnemar"] = S.holm({"C2a": out["C2a_P2_vs_P0"]["mcnemar"]["p"],
                                     "C2b": out["C2b_P2_vs_indep_vote"]["mcnemar"]["p"]})
    # secondary context: P1-Q vs P0, P1 vs its own independent vote
    e = {k: sel(M.team_vote_expected(Ps[k], FULL)) for k in ("P0", "P1", "P2")}
    out["coin_free"] = {"C1": S.paired_contrast(e["P1"], e["P0"], idx, margin=MARGIN, binary=False),
                        "C2a": S.paired_contrast(e["P2"], e["P0"], idx, binary=False),
                        "C2b": S.paired_contrast(e["P2"], sel(M.indep_vote_expected(Ps["P2"], FULL)), idx, binary=False)}
    out["P1Q_vs_P0"] = S.paired_contrast(sel(Ps["P1Q"].correct(FULL)), p0, idx)
    return out


def persona(Ps, idx):
    out = {}
    for name in ("P1", "P1Q"):
        P = Ps[name]
        gen = P.roles.index("generalist")
        res = {}
        for i, role in enumerate(P.roles):
            if role not in M.ROLE_FIELD:
                continue
            fld = M.ROLE_FIELD[role]
            r = {}
            for src in ("solo", "full"):
                for lab, rounds in [("early_r1_2", (1, 2)), ("late_r4_5", (4, 5)), ("all", (1, 2, 3, 4, 5))]:
                    a = M.field_share(P, i, fld, rounds, src)
                    b = M.field_share(P, gen, fld, rounds, src)
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    sub = S.boot_idx(int(ok.sum()), idx.shape[0])
                    r[f"{src}_{lab}"] = {"specialist": float(np.nanmean(a)), "generalist": float(np.nanmean(b)),
                                         **{k: v for k, v in S.paired_contrast(a[ok], b[ok], sub, binary=False).items()
                                            if k in ("diff", "ci95")}}
            a, b = M.dx_field_share(P, i, fld), M.dx_field_share(P, gen, fld)
            ok = ~np.isnan(a) & ~np.isnan(b)
            r["final_dx_in_field_solo"] = {"specialist": float(np.nanmean(a)), "generalist": float(np.nanmean(b)),
                                          **{k: v for k, v in S.paired_contrast(a[ok], b[ok], S.boot_idx(int(ok.sum()), idx.shape[0]), binary=False).items() if k in ("diff", "ci95")}}
            res[role] = r
        out[name] = res
    # P0 reference: do four generalist seeds differ by chance as much? (field share spread among P0 agents)
    P = Ps["P0"]
    out["P0_reference_field_share_all_rounds_solo"] = {
        fld: [float(np.nanmean(M.field_share(P, i, fld, (1, 2, 3, 4, 5)))) for i in FULL] for fld in M.ROLE_FIELD.values()}
    return out


def in_field(Ps, idx):
    """Exploratory: specialists' Shapley (chance v0) on patients whose condition is in vs out of their field."""
    cf = M.condition_field()
    out = {}
    for name in ("P1", "P1Q"):
        P = Ps[name]
        phi = M.shapley_values(P, M.CHANCE)
        res = {}
        for i, role in enumerate(P.roles):
            fld = M.ROLE_FIELD.get(role)
            if not fld:
                continue
            f = np.array([cf[t][0] for t in P.truth]); amb = np.array([cf[t][1] for t in P.truth])
            inn, outf = (f == fld) & ~amb, (f != fld) & ~amb
            res[role] = {"n_in": int(inn.sum()), "shapley_in": float(phi[i][inn].mean()),
                         "shapley_out": float(phi[i][outf].mean()),
                         "solo_acc_in": float(P.correct((i,))[inn].mean()), "solo_acc_out": float(P.correct((i,))[outf].mean())}
        out[name] = res
    return out


def markers(Ps, I):
    out = {}
    for name, P in Ps.items():
        idx = I(P)
        avail = np.array([M.marker_available(p) for p in P.pids])
        r = {"patients_with_marker_available": int(avail.sum()), "n": P.n}
        hit = {sg_name(sg): M.marker_hit(P, sg) for sg in M.SUBGROUPS}
        for k in (1, 2, 3, 4):
            sgs = [sg_name(sg) for sg in M.SUBGROUPS if len(sg) == k]
            h = np.mean([hit[s] for s in sgs], 0)
            acc_h = [P.correct(s)[hit[s] == 1].mean() for s in sgs if hit[s].sum() > 0]
            acc_nh = [P.correct(s)[(hit[s] == 0) & avail].mean() for s in sgs]
            r[f"size{k}"] = {"hit_rate": float(h.mean()), "hit_rate_if_available": float(h[avail].mean()),
                             "acc_given_hit": float(np.mean(acc_h)), "acc_given_no_hit_but_available": float(np.mean(acc_nh))}
        solo_mean = np.mean([hit[s] for s in "ABCD"], 0)
        r["full_minus_mean_solo_hit"] = S.paired_contrast(hit["ABCD"], solo_mean, idx, binary=False)
        r["full_minus_best_solo_hit"] = S.paired_contrast(hit["ABCD"], max((hit[s] for s in "ABCD"), key=np.mean), idx, binary=False)
        out[name] = r
    return out


def coin_sensitivity(Ps, idx):  # idx: 294-patient draws (CHECK only uses counts)
    out = {}
    for name, P in Ps.items():
        r = {}
        for sg in M.SUBGROUPS:
            if len(sg) == 1:
                continue
            a, c = P.correct(sg), M.coin_vote(P, sg)
            r[sg_name(sg)] = {"orig": float(a.mean()), "coin": float(c.mean()), "n_changed": int((a != c).sum())}
        out[name] = {"subgroups": r, "max_abs_change": float(max(abs(v["orig"] - v["coin"]) for v in r.values()))}
    p0, p1, p2 = (M.coin_vote(Ps[k], FULL) for k in ("P0", "P1", "P2"))
    iv2 = M.indep_vote(Ps["P2"], FULL, coin_only=True)[0]
    out["primary_coin_only"] = {"C1": S.paired_contrast(p1, p0, idx, margin=MARGIN),
                                "C2a": S.paired_contrast(p2, p0, idx), "C2b": S.paired_contrast(p2, iv2, idx)}
    return out


def replay(Ps, I):
    """Asking vs reading. e(S) = fixed Gemma-generalist reader on S's transcript; v(S) = S's own answer.
    For the Gemma generalist G4.gen.a (P0-A, P1-D, P2-A) e = v on its solo game by construction, so
    v(full) - v(ref) = [e(full) - e(ref)] (asking) + [v(full) - e(full)] (reading + deliberation + vote)."""
    v0_reader = M.opening_only("gemma4_31b")
    out = {"reader": "gemma4_31b generalist (G4.gen.a)", "v0_reader_opening_only": v0_reader}
    ref = {"P0": 0, "P1": 3, "P2": 0}
    for name, P in Ps.items():
        idx = I(P)
        e = M.replay_correct(P)
        solo_e = np.stack([e[s] for s in "ABCD"])
        best = int(solo_e.mean(1).argmax())
        r = {"e": {k: float(v.mean()) for k, v in e.items()},
             "v": {sg_name(sg): float(P.correct(sg).mean()) for sg in M.SUBGROUPS},
             "e_full_minus_best_solo_e": S.paired_contrast(e["ABCD"], solo_e[best], idx),
             "e_full_minus_mean_solo_e": S.paired_contrast(e["ABCD"], solo_e.mean(0), idx, binary=False),
             "v_full_minus_e_full": S.paired_contrast(P.correct(FULL), e["ABCD"], idx),
             "diversity": M.transcript_diversity(P)}
        by = {}
        for k in (1, 2, 3, 4):
            sgs = [sg_name(sg) for sg in M.SUBGROUPS if len(sg) == k]
            ek = np.mean([e[s] for s in sgs], 0); vk = np.mean([P.correct(s) for s in sgs], 0)
            by[k] = {"e": S.mean_ci(ek, idx), "v": float(vk.mean()), "v_minus_e": S.mean_ci(vk - ek, idx)}
        r["by_size"] = by
        for tag, v0 in [("chance", M.CHANCE), ("reader_opening_only", v0_reader)]:
            ph = M.evidence_shapley(P, v0)
            r[f"evidence_shapley_{tag}"] = [S.mean_ci(ph[i], idx) for i in FULL]
        ans = M.shapley_values(P, M.CHANCE); ev = M.evidence_shapley(P, M.CHANCE)
        r["answer_minus_evidence_shapley_chance"] = [S.mean_ci(ans[i] - ev[i], idx) for i in FULL]
        if name in ref:
            j = ref[name]
            r["decomposition_vs_gemma_generalist_solo"] = {
                "ref_agent": "ABCD"[j], "total": S.paired_contrast(P.correct(FULL), P.correct((j,)), idx),
                "asking": S.paired_contrast(e["ABCD"], e["ABCD"[j]], idx),
                "reading_deliberation_vote": S.paired_contrast(P.correct(FULL), e["ABCD"], idx)}
            assert np.allclose(e["ABCD"[j]], P.correct((j,))), "reader must reproduce G4.gen.a's solo games"
        out[name] = r
    idx = I(Ps["P0"])
    E = {k: M.replay_correct(Ps[k])["ABCD"] for k in ("P0", "P1", "P2", "P1Q")}
    # Exploratory (post hoc, flagged): "one model asks, another reads" = a Qwen solo transcript read by Gemma.
    eq = M.replay_correct(Ps["P1Q"]); eg = M.replay_correct(Ps["P0"])
    q_mean = np.mean([eq[s] for s in "ABCD"], 0); g_mean = np.mean([eg[s] for s in "ABCD"], 0)
    qgen = M.replay_correct(Ps["P2"])["C"]                    # Qwen3.8 generalist asks, Gemma reads
    out["exploratory_qwen_asks_gemma_reads"] = {
        "qwen_gen_transcript_vs_P0_team": S.paired_contrast(qgen, Ps["P0"].correct(FULL), idx),
        "qwen_gen_transcript_vs_P2_team": S.paired_contrast(qgen, Ps["P2"].correct(FULL), idx),
        "qwen_gen_transcript_vs_P2_joint_transcript": S.paired_contrast(qgen, M.replay_correct(Ps["P2"])["ABCD"], idx),
        "mean_4_qwen_solo_transcripts_vs_mean_4_gemma_solo_transcripts": S.paired_contrast(q_mean, g_mean, idx, binary=False),
        "qwen_reads_own_vs_gemma_reads_qwen_mean_4": S.paired_contrast(np.mean([Ps["P1Q"].correct((i,)) for i in FULL], 0), q_mean, idx, binary=False)}
    out["asking_contrasts_full_panel"] = {f"{a}_vs_P0": S.paired_contrast(E[a], E["P0"], idx) for a in ("P1", "P2", "P1Q")}
    return out


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    Ps = {k: M.load_panel(k) for k in PANELS}
    idx294 = S.boot_idx(294, a.B)
    idx98 = S.boot_idx(98, a.B)
    I = lambda P: idx294 if P.n == 294 else idx98  # noqa: E731
    res = {"settings": {"B": a.B, "perm": a.perm, "seed": S.SEED_DEFAULT, "tost_margin": MARGIN}}
    res["panels"] = {k: panel_block(P, I(P)) for k, P in Ps.items()}
    print("panels done", flush=True)
    res["primary"] = primary(Ps, idx294)
    avail = np.array([M.marker_available(p) for p in Ps["P0"].pids])
    nm = ~avail
    res["primary_no_marker_subset"] = primary(Ps, S.boot_idx(int(nm.sum()), a.B), mask=nm) | {"n": int(nm.sum())}
    res["markers"] = markers(Ps, I)
    print("primary + markers done", flush=True)
    res["error_overlap"] = {k: M.error_overlap(P, B=a.perm) for k, P in Ps.items()}
    print("overlap done", flush=True)
    res["deliberation"] = {k: M.deliberation(P) for k, P in Ps.items()}
    res["deliberation_full_panel"] = {k: M.deliberation(P, [FULL]) for k, P in Ps.items()}
    res["persona_check"] = persona(Ps, idx294)
    res["in_field_exploratory"] = in_field(Ps, idx294)
    res["coin_tiebreak_sensitivity"] = coin_sensitivity(Ps, idx294)
    res["cost"] = {k: M.cost(P) for k, P in Ps.items()}
    if all((M.REPLAY / k).exists() for k in PANELS):
        res["replay"] = replay(Ps, I)
    (OUT / "analysis.json").write_text(json.dumps(r3(res), indent=1))
    print("wrote", OUT / "analysis.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--perm", type=int, default=2000)
    main(ap.parse_args())
