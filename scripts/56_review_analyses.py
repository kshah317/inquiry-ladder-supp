"""Robustness analyses. Existing logs only, no API calls.

  nb        reader-free evidence value of every transcript (src/ddx/nb.py), incl. the union of the 4 solo transcripts
  decomp    LLM-reader decomposition: second term split into (reading + first vote) and deliberation;
            reference = mean of the four Gemma generalist seeds (P0 solos) instead of one seed
  qvote     how the question vote is resolved in full panels (splits, wins per agent)
  errors    when the independent / team vote is wrong: is the winning wrong answer shared (>= 2 votes)?
  sc8       self-consistency: vote over 8 Gemma generalist solo games (replica 0 + 1) vs the anchor
  crossfit  best member chosen on replica 1, evaluated on replica 0
  clean     P2 restricted to full-panel games with no invalid proposal or answer
  cost      billed $ per million tokens per model, from the logged usage
  marker    primary contrasts on patients WITH a marker available (complements the no-marker subset)
Output: results/analysis/review.json and REVIEW_REPORT.md
"""
from __future__ import annotations
import json, random, sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddx import metrics as M, stats as S, nb
from ddx.metrics import load_panel, SUBGROUPS, sg_name, replay_correct, _slim
from ddx.game import _seed, _vote_info, norm_conf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
PANELS = ["P0", "P1", "P1Q", "P2", "CHECK"]
R = {}


def pc(a, b, idx):
    c = S.paired_contrast(a, b, idx)
    return {"diff": round(100 * c["diff"], 2), "ci95": [round(100 * x, 2) for x in c["ci95"]], "a": c["a"], "b": c["b"],
            "mcnemar_p": c.get("mcnemar", {}).get("p")}


def pcc(a, b, idx):                       # continuous (p_truth), in points
    d = np.asarray(a) - np.asarray(b)
    dr = S.boot_means(d, idx)
    return {"diff": round(100 * d.mean(), 2), "ci95": [round(100 * x, 2) for x in S.ci(dr)]}


def asked(g):
    return [e["chosen"] for e in g["rounds"] if e["chosen"]]


def load_jsonl(f, pids):
    d = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["pid"] in pids:
            d[r["pid"]] = _slim(r)
    assert sorted(d) == sorted(pids), f
    return [d[p] for p in sorted(pids)]


P = {n: load_panel(n) for n in PANELS}
IDX = {294: S.boot_idx(294), 98: S.boot_idx(98)}

# ---------------------------------------------------------------- nb
nbres = {}
NB = {}
for n, pan in P.items():
    idx = IDX[pan.n]
    e1, ep = {}, {}
    for sg in SUBGROUPS:
        s = [nb.score(p, t, asked(g)) for p, t, g in zip(pan.pids, pan.truth, pan.g(sg))]
        e1[sg_name(sg)] = np.array([x[0] for x in s]); ep[sg_name(sg)] = np.array([x[1] for x in s])
    uni = [nb.score(p, t, sum((asked(pan.games[k][p]) for k in "ABCD"), [])) for p, t in zip(pan.pids, pan.truth)]
    opn = [nb.score(p, t, []) for p, t in zip(pan.pids, pan.truth)]
    e1["union"] = np.array([x[0] for x in uni]); ep["union"] = np.array([x[1] for x in uni])
    e1["opening"] = np.array([x[0] for x in opn]); ep["opening"] = np.array([x[1] for x in opn])
    solo1 = np.mean([e1[k] for k in "ABCD"], 0); solop = np.mean([ep[k] for k in "ABCD"], 0)
    best = max("ABCD", key=lambda k: e1[k].mean())
    NB[n] = (e1, ep)
    nbres[n] = {
        "top1": {k: round(float(v.mean()), 3) for k, v in e1.items()},
        "p_truth": {k: round(float(v.mean()), 3) for k, v in ep.items()},
        "full_minus_mean_solo": pcc(e1["ABCD"], solo1, idx), "full_minus_mean_solo_p": pcc(ep["ABCD"], solop, idx),
        "full_minus_best_solo": dict(pc(e1["ABCD"], e1[best], idx), best=best),
        "union_minus_full": pc(e1["union"], e1["ABCD"], idx), "union_minus_full_p": pcc(ep["union"], ep["ABCD"], idx),
        "full_minus_opening": pc(e1["ABCD"], e1["opening"], idx),
        "pairs_minus_mean_members": pcc(np.mean([e1[sg_name(s)] for s in SUBGROUPS if len(s) == 2], 0), solo1, idx),
    }
for n in ["P1", "P1Q", "P2"]:
    nbres[n]["full_minus_P0full"] = pc(NB[n][0]["ABCD"], NB["P0"][0]["ABCD"], IDX[294])
    nbres[n]["full_minus_P0full_p"] = pcc(NB[n][1]["ABCD"], NB["P0"][1]["ABCD"], IDX[294])
# full record and 5 questions in the abstract: how much a 5-question budget can reveal at most is not computed (NP-hard);
full = [nb.score(p, t, nb.full_record_codes(p)) for p, t in zip(P["P0"].pids, P["P0"].truth)]
nbres["sanity_full_record_top1"] = round(float(np.mean([x[0] for x in full])), 3)
R["nb"] = nbres

# ---------------------------------------------------------------- decomposition (LLM reader)
dec = {}
idx = IDX[294]
e0 = replay_correct(P["P0"])
v0s = np.mean([P["P0"].correct((i,)) for i in range(4)], 0)
e0s = np.mean([e0[k] for k in "ABCD"], 0)
for n in ["P0", "P1", "P2"]:
    pan = P[n]; e = replay_correct(pan)
    v, vpre, et = pan.correct(M.FULL), pan.correct(M.FULL, pre=True), e["ABCD"]
    dec[n] = {
        "total_vs_mean_gemma_seed": pc(v, v0s, idx),
        "asking_vs_mean_seed": pcc(et, e0s, idx),
        "after_questions": pcc(v, et, idx),
        "after_split_reading_first_vote": pcc(vpre, et, idx),
        "after_split_deliberation": pcc(v, vpre, idx),
        "residual_mean_seed_e_minus_v": pcc(e0s, v0s, idx),
    }
R["decomp"] = dec

# ---------------------------------------------------------------- question votes in full panels
qv = {}
for n in ["P0", "P1", "P1Q", "P2"]:
    pan = P[n]; gs = pan.g(M.FULL)
    how = Counter(e["vote"] for g in gs for e in g["rounds"])
    nprop = Counter(sum(1 for x in e["props"] if x == e["chosen"]) for g in gs for e in g["rounds"] if e["chosen"])
    wins = np.array([[sum(1 for e in g["rounds"] if e["chosen"] and e["props"][i] == e["chosen"]) for i in range(4)] for g in gs])
    distinct = np.mean([len({x for x in e["props"] if x}) for g in gs for e in g["rounds"]])
    tot = sum(how.values()); totp = sum(nprop.values())
    qv[n] = {"how": {k: round(v / tot, 3) for k, v in how.items()},
             "proposers_of_chosen": {k: round(v / totp, 3) for k, v in sorted(nprop.items())},
             "distinct_proposals_per_round": round(float(distinct), 2),
             "win_share_by_agent": [round(float(x), 3) for x in wins.mean(0) / 5],
             "games_one_agent_wins_ge4": round(float((wins.max(1) >= 4).mean()), 3),
             "games_one_agent_wins_all5": round(float((wins.max(1) == 5).mean()), 3)}
    # does inquiry dominated by one agent give better or worse evidence? (descriptive)
    dom = wins.max(1) >= 4
    e1 = NB[n][0]["ABCD"]
    qv[n]["nb_top1_dominated_vs_not"] = [round(float(e1[dom].mean()), 3) if dom.any() else None,
                                          round(float(e1[~dom].mean()), 3), int(dom.sum())]
R["qvote"] = qv

# ---------------------------------------------------------------- errors: is the wrong plurality shared?
er = {}
for n in ["P0", "P1", "P1Q", "P2"]:
    pan = P[n]
    ok, winners = M.indep_vote(pan, (0, 1, 2, 3))
    solos = [pan.solo_dx(i) for i in range(4)]
    shared, wrong = 0, 0
    for k in range(pan.n):
        if not ok[k]:
            wrong += 1
            w = winners[k]
            if w is not None and sum(1 for i in range(4) if solos[i][k][0] == w) >= 2:
                shared += 1
    team_wrong = [g for g in pan.g(M.FULL) if not g["correct_pre"]]
    t_shared = sum(1 for g in team_wrong if sum(1 for d, _ in g["final"] if d == g["panel_pre"]) >= 2)
    anyright = np.max([[s[k][0] == pan.truth[k] for k in range(pan.n)] for s in solos], 0)
    pairs = {}
    for i, j in combinations(range(4), 2):
        t = M.PAIR_TYPE.get(n, {}).get((i, j)) if isinstance(M.PAIR_TYPE.get(n), dict) else None
        bw = sum(1 for k in range(pan.n) if solos[i][k][0] != pan.truth[k] and solos[j][k][0] != pan.truth[k])
        pairs[f"{'ABCD'[i]}{'ABCD'[j]}"] = bw
    er[n] = {"indep_vote_wrong": wrong, "of_which_winner_shared_by_ge2": shared,
             "team_pre_vote_wrong": len(team_wrong), "of_which_shared_by_ge2": t_shared,
             "oracle_any_solo_right": round(float(anyright.mean()), 3), "both_wrong_cases_per_pair": pairs}
R["errors"] = er

# ---------------------------------------------------------------- sc8 vs anchor
pids = P["P0"].pids; truth = P["P0"].truth
r1 = {k: load_jsonl(ROOT / "results" / "main_r1" / "P0" / f"{k}.jsonl", pids) for k in "ABCD"}
r0 = {k: P["P0"].g(("ABCD".index(k),)) for k in "ABCD"}
games8 = [r0[k] for k in "ABCD"] + [r1[k] for k in "ABCD"]
sc8, sc8cost = [], []
for k, p in enumerate(pids):
    ch = [(g[k]["final"][0][0], norm_conf("gemma4_31b", g[k]["final"][0][1], "final")) for g in games8 if g[k]["final"][0][0]]
    w, _ = _vote_info(ch, random.Random(_seed(M.BASE_SEED, p, "sc8")))
    sc8.append(float(w == truth[k])); sc8cost.append(sum(g[k]["cost"] for g in games8))
sc8 = np.array(sc8)
anc = np.array([g["correct"] for g in load_jsonl(ROOT / "results" / "main" / "ANCHOR" / "A.jsonl", pids)], float)
r1solo = np.mean([[g["correct"] for g in r1[k]] for k in "ABCD"], 1)
R["sc8"] = {"sc8_top1": S.mean_ci(sc8, idx), "sc8_cost_per_patient": round(float(np.mean(sc8cost)), 4),
            "anchor_minus_sc8": pc(anc, sc8, idx), "sc8_minus_P0team": pc(sc8, P["P0"].correct(M.FULL), idx),
            "sc8_minus_indep_vote4": pc(sc8, M.indep_vote(P["P0"], (0, 1, 2, 3))[0], idx),
            "r1_P0_solo_acc": [round(float(x), 3) for x in r1solo]}

# ---------------------------------------------------------------- crossfit best member
cf = {}
for n in ["P0", "P1", "P2"]:
    pan = P[n]
    r1s = {k: np.array([g["correct"] for g in load_jsonl(ROOT / "results" / "main_r1" / n / f"{k}.jsonl", pids)], float) for k in "ABCD"}
    b = max("ABCD", key=lambda k: r1s[k].mean())
    cf[n] = dict(pc(pan.correct(M.FULL), pan.correct(("ABCD".index(b),)), idx), chosen_on_r1=b)
R["crossfit"] = cf

# ---------------------------------------------------------------- P2 clean games
pan = P["P2"]
clean = np.array([all(x for e in g["rounds"] for x in e["props"]) and all(d for d, _ in g["final"]) for g in pan.g(M.FULL)])
sub = pan.subset(clean); sub0 = P["P0"].subset(clean); ids = S.boot_idx(int(clean.sum()))
R["p2_clean"] = {"n_clean": int(clean.sum()),
                 "C2a_clean": pc(sub.correct(M.FULL), sub0.correct(M.FULL), ids),
                 "C2b_clean": pc(sub.correct(M.FULL), M.indep_vote(sub, (0, 1, 2, 3))[0], ids)}

# ---------------------------------------------------------------- marker-available subset
mk = np.array([M.marker_available(p) for p in pids])
ids = S.boot_idx(int(mk.sum()))
s0, s1, s2 = P["P0"].subset(mk), P["P1"].subset(mk), P["P2"].subset(mk)
R["marker_subset"] = {"n": int(mk.sum()), "C1": pc(s1.correct(M.FULL), s0.correct(M.FULL), ids),
                      "C2a": pc(s2.correct(M.FULL), s0.correct(M.FULL), ids),
                      "C2b": pc(s2.correct(M.FULL), M.indep_vote(s2, (0, 1, 2, 3))[0], ids)}

# ---------------------------------------------------------------- billed cost per model
cost = {}
for f in [ROOT / "results/main/P2" / f"{k}.jsonl" for k in "ABCD"] + [ROOT / "results/main/CHECK/C.jsonl", ROOT / "results/main/ANCHOR/A.jsonl"]:
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        for u in r.get("usage", []):
            m = u.get("model") or r["agents"][0]["model"]
            c = cost.setdefault(m, {"in": 0, "out": 0, "cost": 0.0, "pc": 0.0, "cc": 0.0, "cached_tok": 0, "reason": 0})
            c["in"] += u.get("prompt_tokens") or 0; c["out"] += u.get("completion_tokens") or 0
            c["cost"] += u.get("cost") or 0
            cd = u.get("cost_details") or {}
            c["pc"] += cd.get("upstream_inference_prompt_cost") or 0; c["cc"] += cd.get("upstream_inference_completions_cost") or 0
            c["cached_tok"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            c["reason"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
R["billed_rates"] = {m: {"usd_per_M_in": round(c["pc"] / max(c["in"], 1) * 1e6, 3), "usd_per_M_out": round(c["cc"] / max(c["out"], 1) * 1e6, 3),
                         "usd_per_M_total_blend": round(c["cost"] / max(c["in"] + c["out"], 1) * 1e6, 3),
                         "cached_share_in": round(c["cached_tok"] / max(c["in"], 1), 3), "reasoning_share_out": round(c["reason"] / max(c["out"], 1), 3),
                         "tokens_in": c["in"], "tokens_out": c["out"], "cost": round(c["cost"], 3)} for m, c in cost.items()}
R["transcript_diversity"] = {n: M.transcript_diversity(P[n]) for n in ["P0", "P1", "P1Q", "P2"]}

(OUT / "review.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
print(json.dumps(R, indent=1, default=float))

# ---------------------------------------------------------------- anchor transcripts, reader-free (appended)
ag = load_jsonl(ROOT / "results" / "main" / "ANCHOR" / "A.jsonl", pids)
an = np.array([nb.score(p, t, asked(g))[0] for p, t, g in zip(pids, truth, ag)])
nq = np.array([len(asked(g)) for g in ag])
gem = NB["P0"][0]["A"]
R["anchor_nb"] = {"anchor_nb_top1": round(float(an.mean()), 3), "anchor_minus_gemma_solo_A": pc(an, gem, idx),
                  "anchor_minus_mean_P0_solo": pcc(an, np.mean([NB["P0"][0][k] for k in "ABCD"], 0), idx),
                  "anchor_games_lt5_questions": int((nq < 5).sum()),
                  "anchor_nb_top1_5q_games": round(float(an[nq == 5].mean()), 3)}
(OUT / "review.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
print(json.dumps(R["anchor_nb"], indent=1, default=float))
