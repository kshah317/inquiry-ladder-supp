"""Independent verification of the repeat run (replica 1) and pooled replicas.
Own loading, vote, bootstrap, TOST and McNemar code. Reads only results/*/jsonl and
prereg/confidence_scales.json (percentile rule re-implemented, not imported). No API calls, no cache."""
import json, random, bisect, hashlib, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
PANELS, SOLO = ["P0", "P1", "P2"], ["A", "B", "C", "D"]
RUNS = {0: "main", 1: "main_r1"}
SCALES = json.loads((ROOT / "prereg/confidence_scales.json").read_text())

def pct(model, conf):
    d = (SCALES.get(model) or {}).get("final") or []
    if not d: return conf / 100
    return (bisect.bisect_left(d, conf) + bisect.bisect_right(d, conf)) / 2 / len(d)

def load(path):
    g = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            o = json.loads(line); g[o["pid"]] = o  # last line wins
    return g

D = {r: {P: {sg: load(ROOT / f"results/{RUNS[r]}/{P}/{sg}.jsonl") for sg in SOLO + ["ABCD"]} for P in PANELS} for r in RUNS}
PIDS = sorted(D[0]["P0"]["ABCD"])
out = {"integrity": {}, "spot": [], "replica": {}, "contrasts": {}, "stability": {}}

# ---------- integrity
integ = {"n_unique": {}, "lines": {}, "bad_replica_agents": 0, "bad_top_replica": 0, "pidset_equal": True,
         "truth_mismatch_vs_r0": 0, "truth_mismatch_within": 0, "correct_flag_mismatch": 0, "model_mismatch_solo_vs_team": 0}
for r in RUNS:
    for P in PANELS:
        for sg, g in D[r][P].items():
            integ["n_unique"][f"r{r}/{P}/{sg}"] = len(g)
            integ["lines"][f"r{r}/{P}/{sg}"] = sum(1 for l in open(ROOT / f"results/{RUNS[r]}/{P}/{sg}.jsonl", encoding="utf-8") if l.strip())
            if sorted(g) != PIDS: integ["pidset_equal"] = False
            for pid, o in g.items():
                if o["correct"] != (o["panel"] == o["truth"]): integ["correct_flag_mismatch"] += 1
                if o["truth"] != D[0][P]["ABCD"][pid]["truth"]: integ["truth_mismatch_vs_r0" if r else "truth_mismatch_within"] += 1
                if r == 1:
                    integ["bad_replica_agents"] += sum(a.get("replica") != 1 for a in o["agents"])
                    integ["bad_top_replica"] += o.get("replica") != 1
            if sg in SOLO:
                for pid, o in g.items():
                    if o["agents"][0]["model"] != D[r][P]["ABCD"][pid]["agents"][SOLO.index(sg)]["model"]:
                        integ["model_mismatch_solo_vs_team"] += 1
out["integrity"] = integ

# ---------- votes
def tied_set(ans):
    """ans=[(dx, model, conf)]; plurality -> max within-model percentile -> remaining tie."""
    ch = [(d, pct(m, c)) for d, m, c in ans if d]
    if not ch: return [], None
    cnt = Counter(d for d, _ in ch); top = max(cnt.values())
    t = sorted(d for d, k in cnt.items() if k == top)
    if len(t) == 1: return t, "plurality"
    b = {d: max(c for o, c in ch if o == d) for d in t}; mx = max(b.values())
    t = sorted(d for d in t if b[d] == mx)
    return t, ("confidence" if len(t) == 1 else "coin")

def seeded(P, pid, r):
    return random.Random(int(hashlib.md5(f"verify|{P}|{pid}|{r}".encode()).hexdigest()[:8], 16))

V = {}  # V[r][P][key] -> np.array over PIDS
ncoin = {}
for r in RUNS:
    V[r] = {}
    for P in PANELS:
        T = D[r][P]["ABCD"]; v = {}
        v["team"] = np.array([float(T[p]["correct"]) for p in PIDS])
        v["pre"] = np.array([float(T[p]["correct_pre"]) for p in PIDS])
        for s in SOLO:
            v[s] = np.array([float(D[r][P][s][p]["correct"]) for p in PIDS])
        iv, ive, tve, nc, team_in_tied, nmin, nmax = [], [], [], Counter(), 0, [], []
        for p in PIDS:
            tr = T[p]["truth"]
            ans = [(D[r][P][s][p]["final"][0]["dx"], D[r][P][s][p]["agents"][0]["model"], D[r][P][s][p]["final"][0]["confidence"]) for s in SOLO]
            t, how = tied_set(ans); nc[f"indep_{how}"] += 1
            w = seeded(P, p, r).choice(t) if t else None
            nmin.append(float(t == [tr])); nmax.append(float(tr in t))
            iv.append(float(w == tr)); ive.append(sum(d == tr for d in t) / len(t) if t else 0.0)
            tans = [(f["dx"], a["model"], f["confidence"]) for f, a in zip(T[p]["revised"], T[p]["agents"])]
            tt, thow = tied_set(tans); nc[f"team_{thow}"] += 1
            tve.append(sum(d == tr for d in tt) / len(tt) if tt else 0.0)
            team_in_tied += (T[p]["panel"] in tt) if tt else (T[p]["panel"] is None)
        v["indep"], v["indep_cf"], v["team_cf"], v["indep_min"], v["indep_max"] = map(np.array, (iv, ive, tve, nmin, nmax))
        nc["team_logged_answer_in_my_tied_set"] = team_in_tied
        ncoin[f"r{r}/{P}"] = dict(nc)
        V[r][P] = v
V["pooled"] = {P: {k: (V[0][P][k] + V[1][P][k]) / 2 for k in V[0][P]} for P in PANELS}
out["vote_paths"] = ncoin

B = 10000
G = np.random.default_rng(20261102)
BIDX = G.integers(0, len(PIDS), size=(B, len(PIDS)))

def ci(d):
    m = d[BIDX].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], [float(np.percentile(m, 5)), float(np.percentile(m, 95))]

def tost(d, eps=0.05):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0: return 0.0 if abs(m) < eps else 1.0
    p_lo = 1 - stats.t.cdf((m + eps) / se, n - 1)   # H0: diff <= -eps
    p_hi = stats.t.cdf((m - eps) / se, n - 1)       # H0: diff >= +eps
    return float(max(p_lo, p_hi))

def mcnemar(a, b):
    ao, bo = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(min(ao, bo), ao + bo, 0.5).pvalue if ao + bo else 1.0
    return ao, bo, float(p)

def contrast(a, b, binary):
    d = a - b; c95, c90 = ci(d)
    res = {"diff": float(d.mean()), "ci95": c95, "ci90": c90, "tost_p": tost(d)}
    if binary:
        ao, bo, p = mcnemar(a, b); res.update(mcnemar_a_only=ao, mcnemar_b_only=bo, mcnemar_p=p)
    return res

for r in [0, 1, "pooled"]:
    R = {}
    for P in PANELS:
        v = V[r][P]
        solo = {s: float(v[s].mean()) for s in SOLO}
        best = max(SOLO, key=lambda s: solo[s])
        R[P] = {"team": float(v["team"].mean()), "team_pre": float(v["pre"].mean()), "solo": solo, "best": best,
                "team_minus_best": contrast(v["team"], v[best], r != "pooled"),
                "indep_vote": float(v["indep"].mean()), "team_minus_indep": contrast(v["team"], v["indep"], r != "pooled"),
                "indep_vote_coinfree": float(v["indep_cf"].mean()),
                "indep_vote_coin_range": [float(v["indep_min"].mean()), float(v["indep_max"].mean())], "team_coinfree": float(v["team_cf"].mean()),
                "team_minus_indep_coinfree": contrast(v["team_cf"], v["indep_cf"], False)}
    out["replica"][str(r)] = R
    bi = r != "pooled"
    out["contrasts"][str(r)] = {
        "C1": contrast(V[r]["P1"]["team"], V[r]["P0"]["team"], bi),
        "C2a": contrast(V[r]["P2"]["team"], V[r]["P0"]["team"], bi),
        "C2b": contrast(V[r]["P2"]["team"], V[r]["P2"]["indep"], bi),
        "C1_coinfree": contrast(V[r]["P1"]["team_cf"], V[r]["P0"]["team_cf"], False),
        "C2a_coinfree": contrast(V[r]["P2"]["team_cf"], V[r]["P0"]["team_cf"], False),
        "C2b_coinfree": contrast(V[r]["P2"]["team_cf"], V[r]["P2"]["indep_cf"], False)}

for P in PANELS:
    a0, a1 = V[0][P]["team"], V[1][P]["team"]
    b0 = out["replica"]["0"][P]["best"]
    out["stability"][P] = {"team_r1_minus_r0": contrast(a1, a0, True), "same_correctness_share": float((a0 == a1).mean()),
                           "r0_best": b0, "solo_r1_minus_r0": {s: float(V[1][P][s].mean() - V[0][P][s].mean()) for s in SOLO}, "r1_team_minus_r1_solo_of_r0_best": contrast(a1, V[1][P][b0], True)}

# ---------- spot checks (replica 1)
rs = random.Random(31)
for P, pid in [(P_, rs.choice(PIDS)) for P_ in PANELS]:  # one random pid per panel
    T = D[1][P]["ABCD"][pid]
    tt, how = tied_set([(f["dx"], a["model"], f["confidence"]) for f, a in zip(T["revised"], T["agents"])])
    out["spot"].append({"panel": P, "pid": pid, "truth": T["truth"], "team_revised": [[f["dx"], f["confidence"]] for f in T["revised"]],
                        "logged_panel": T["panel"], "logged_vote": T["vote"], "my_tied": tt, "my_how": how,
                        "logged_correct": T["correct"], "check_correct": T["correct"] == (T["panel"] == T["truth"]) and T["panel"] in tt,
                        "solo_finals": {s: [D[1][P][s][pid]["final"][0]["dx"], D[1][P][s][pid]["final"][0]["confidence"], D[1][P][s][pid]["correct"]] for s in SOLO},
                        "solo_panel_equals_final": all(D[1][P][s][pid]["panel"] == D[1][P][s][pid]["final"][0]["dx"] for s in SOLO)})
out["integrity"]["solo_panel_neq_final_dx"] = sum(D[r][P][s][q]["panel"] != D[r][P][s][q]["final"][0]["dx"] for r in RUNS for P in PANELS for s in SOLO for q in PIDS)

# ---------- comparison with the main analysis (read only after the numbers above are computed)
M = json.loads((ROOT / "results/analysis/repeats.json").read_text())
cmp = []
CI_TOL = 0.0035  # one patient step = 1/294 = 0.0034; bootstrap percentiles sit on this grid
def chk(name, mine, theirs, tol):
    if isinstance(mine, list):
        ok = all(abs(a - b) <= tol for a, b in zip(mine, theirs))
        gap = max(abs(a - b) for a, b in zip(mine, theirs))
    elif isinstance(mine, str): ok, gap = mine == theirs, 0
    else: ok, gap = abs(mine - theirs) <= tol, abs(mine - theirs)
    cmp.append({"q": name, "mine": mine, "main": theirs, "match": bool(ok), "gap": gap})
r4 = lambda x: round(x, 4)
for r, key in [("0", "replica0"), ("1", "replica1"), ("pooled", "pooled")]:
    for P in PANELS:
        a, b = out["replica"][r][P], M[key]["panels"][P]
        chk(f"{key}/{P}/team", r4(a["team"]), b["team"]["mean"], 5e-5)
        chk(f"{key}/{P}/team_pre", r4(a["team_pre"]), b["team_pre"], 5e-5)
        chk(f"{key}/{P}/solo", [r4(a["solo"][s]) for s in SOLO], b["solo"], 5e-5)
        chk(f"{key}/{P}/best", a["best"], b["best_member"], 0)
        chk(f"{key}/{P}/team_minus_best", r4(a["team_minus_best"]["diff"]), b["team_minus_best"]["diff"], 5e-5)
        chk(f"{key}/{P}/team_minus_best_ci95", a["team_minus_best"]["ci95"], b["team_minus_best"]["ci95"], CI_TOL)
        chk(f"{key}/{P}/indep_vote(coin)", r4(a["indep_vote"]), b["indep_vote"], 5e-5)
        chk(f"{key}/{P}/indep_vote_main_in_my_coin_range", 1.0, float(a["indep_vote_coin_range"][0] - 5e-5 <= b["indep_vote"] <= a["indep_vote_coin_range"][1] + 5e-5), 0)
        chk(f"{key}/{P}/team_minus_indep(coin)", r4(a["team_minus_indep"]["diff"]), b["team_minus_indep_vote"]["diff"], 5e-5)
        chk(f"{key}/{P}/cf_team", r4(a["team_coinfree"]), b["coin_free"]["team"], 5e-5)
        chk(f"{key}/{P}/cf_indep", r4(a["indep_vote_coinfree"]), b["coin_free"]["indep_vote"], 5e-5)
        chk(f"{key}/{P}/cf_team_minus_indep", r4(a["team_minus_indep_coinfree"]["diff"]), b["coin_free"]["team_minus_indep_vote"]["diff"], 5e-5)
        chk(f"{key}/{P}/cf_team_minus_indep_ci95", a["team_minus_indep_coinfree"]["ci95"], b["coin_free"]["team_minus_indep_vote"]["ci95"], CI_TOL)
        if r != "pooled":
            mb = b["team_minus_best"]["mcnemar"]; x = a["team_minus_best"]
            chk(f"{key}/{P}/team_minus_best_mcnemar", [x["mcnemar_a_only"], x["mcnemar_b_only"], r4(x["mcnemar_p"])], [mb["a_only"], mb["b_only"], mb["p"]], 5e-5)
    for c, mk in [("C1", "C1_P1_vs_P0"), ("C2a", "C2a_P2_vs_P0"), ("C2b", "C2b_P2_vs_indep_vote"),
                  ("C1_coinfree", ("coin_free", "C1")), ("C2a_coinfree", ("coin_free", "C2a")), ("C2b_coinfree", ("coin_free", "C2b"))]:
        a = out["contrasts"][r][c]; b = M[key]["primary"][mk] if isinstance(mk, str) else M[key]["primary"][mk[0]][mk[1]]
        chk(f"{key}/{c}/diff", r4(a["diff"]), b["diff"], 5e-5)
        chk(f"{key}/{c}/ci95", a["ci95"], b["ci95"], CI_TOL)
        chk(f"{key}/{c}/ci90", a["ci90"], b["ci90"], CI_TOL)
        chk(f"{key}/{c}/tost_p", r4(a["tost_p"]), b["tost"]["p_tost_t"], 0.005)
        if r != "pooled" and "mcnemar_a_only" in a:
            chk(f"{key}/{c}/mcnemar", [a["mcnemar_a_only"], a["mcnemar_b_only"], r4(a["mcnemar_p"])], [b["mcnemar"]["a_only"], b["mcnemar"]["b_only"], b["mcnemar"]["p"]], 5e-5)
for P in PANELS:
    a, b = out["stability"][P], M["stability"][P]
    chk(f"stab/{P}/r1_minus_r0", r4(a["team_r1_minus_r0"]["diff"]), b["team_r1_minus_r0"]["diff"], 5e-5)
    chk(f"stab/{P}/r1_minus_r0_ci95", a["team_r1_minus_r0"]["ci95"], b["team_r1_minus_r0"]["ci95"], CI_TOL)
    x = a["team_r1_minus_r0"]; chk(f"stab/{P}/r1_minus_r0_mcnemar", [x["mcnemar_a_only"], x["mcnemar_b_only"], r4(x["mcnemar_p"])], [b["team_r1_minus_r0"]["mcnemar"][k] for k in ("a_only", "b_only", "p")], 5e-5)
    chk(f"stab/{P}/same_correctness", r4(a["same_correctness_share"]), b["team_agreement"], 5e-5)
    chk(f"stab/{P}/solo_r1_minus_r0", [r4(a["solo_r1_minus_r0"][s]) for s in SOLO], b["solo_r1_minus_r0"], 5e-5)
    chk(f"stab/{P}/r0_best", a["r0_best"], b["best_r0"], 0)
    y = a["r1_team_minus_r1_solo_of_r0_best"]; z = b["r1_team_minus_best_chosen_on_r0"]
    chk(f"stab/{P}/r1_team_minus_r0best", r4(y["diff"]), z["diff"], 5e-5)
    chk(f"stab/{P}/r1_team_minus_r0best_ci95", y["ci95"], z["ci95"], CI_TOL)
    chk(f"stab/{P}/r1_team_minus_r0best_mcnemar", [y["mcnemar_a_only"], y["mcnemar_b_only"], r4(y["mcnemar_p"])], [z["mcnemar"][k] for k in ("a_only", "b_only", "p")], 5e-5)
out["comparison"] = {"n_checks": len(cmp), "n_match": sum(c["match"] for c in cmp),
                     "mismatches": [c for c in cmp if not c["match"]], "all": cmp}

json.dump(out, open(ROOT / "results/analysis/verify/v_repeats_out.json", "w"), indent=1)
print("ok")
