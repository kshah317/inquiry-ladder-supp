"""Independent verification of the anchor analysis.
Own code; reads only jsonl logs and analysis.json (no cache, no API)."""
import json, os
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = lambda *p: os.path.join(ROOT, *p)
SEED, B = 4242, 10000
rng = np.random.default_rng(SEED)

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pids = [r["pid"] for r in rows]
    assert len(pids) == len(set(pids)), f"duplicate pids in {path}"
    return {r["pid"]: r for r in rows}

A = load(R("results/main/ANCHOR/A.jsonl"))
pids = sorted(A)
n = len(pids)
a = np.array([bool(A[p]["correct"]) for p in pids], float)

# consistency of `correct`
corr_mismatch = sum(bool(A[p]["correct"]) != ((A[p]["final"][0].get("dx") if A[p]["final"] else None) == A[p]["truth"]) for p in pids)

idx = rng.integers(0, n, size=(B, n))
def ci(x):
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

usage = [u for p in pids for u in A[p]["usage"]]
cost_game = np.array([sum(u.get("cost") or 0 for u in A[p]["usage"]) for p in pids])
trunc_game = np.array([sum(u.get("finish_reason") == "length" for u in A[p]["usage"]) for p in pids])
n_trunc = int(trunc_game.sum())
abst = sum(1 for p in pids for r in A[p]["rounds"] for pr in r["proposals"] if pr.get("valid") is False)
final_missing = sum(1 for p in pids if not A[p]["final"] or not A[p]["final"][0].get("dx"))
chosen = np.array([sum(1 for r in A[p]["rounds"] if r.get("chosen")) for p in pids])
bytr = {}
for k, mask in (("0", trunc_game == 0), ("1", trunc_game == 1), ("2+", trunc_game >= 2)):
    bytr[k] = {"n": int(mask.sum()), "acc": float(a[mask].mean())}

out = {"seed": SEED, "B": B, "n": n, "pid_unique": True,
       "correct_vs_final_mismatch": int(corr_mismatch),
       "acc": {"mean": float(a.mean()), "ci95": ci(a)},
       "cost_per_consultation_usd": float(cost_game.mean()),
       "calls": len(usage), "truncated_calls": n_trunc, "truncated_share": n_trunc / len(usage),
       "retries": int(sum(A[p]["retries"] for p in pids)),
       "abstained_proposals": abst, "final_missing": final_missing,
       "games_with_lt5_questions": int((chosen < 5).sum()),
       "acc_by_truncations": bytr, "vs": {}}

an = json.load(open(R("results/analysis/analysis.json")))
for P in ("P0", "P1", "P2", "P1Q"):
    best = an["panels"][P]["best_member"]
    team_cost = an["cost"][P]["per_consultation_usd"]
    res = {"best_member": best}
    for label, sg in (("team", "ABCD"), ("best_member_" + best, best)):
        O = load(R("results/main", P, sg + ".jsonl"))
        assert set(O) == set(pids), f"pid set mismatch {P}/{sg}"
        o = np.array([bool(O[p]["correct"]) for p in pids], float)
        d = a - o
        ao, bo = int(((a == 1) & (o == 0)).sum()), int(((a == 0) & (o == 1)).sum())
        p = binomtest(ao, ao + bo, 0.5).pvalue if ao + bo else 1.0
        oc = float(np.mean([sum(u.get("cost") or 0 for u in O[q]["usage"]) for q in pids]))
        res[label] = {"anchor": float(a.mean()), "other": float(o.mean()), "diff_pts": 100 * float(d.mean()),
                      "ci95_pts": [100 * v for v in ci(d)],
                      "mcnemar": {"a_only": ao, "b_only": bo, "p": float(p)},
                      "other_cost_raw_usd": oc}
    res["team_cost_usd_analysis_json"] = team_cost
    res["solo_cost_usd_analysis_json"] = an["cost"][P]["solo_mean_usd"]
    res["cost_ratio_anchor_over_team"] = out["cost_per_consultation_usd"] / team_cost
    bl = "best_member_" + best
    res["best_member_cost_raw_usd"] = res[bl]["other_cost_raw_usd"]
    res["cost_ratio_anchor_over_best_member_raw"] = out["cost_per_consultation_usd"] / res[bl]["other_cost_raw_usd"]
    res["cost_ratio_anchor_over_solo_mean"] = out["cost_per_consultation_usd"] / an["cost"][P]["solo_mean_usd"]
    res["cost_ratio_anchor_over_team_raw"] = out["cost_per_consultation_usd"] / res["team"]["other_cost_raw_usd"]
    out["vs"][P] = res

# compare against claimed anchor.json
C = json.load(open(R("results/analysis/anchor.json")))
flags, checks = [], 0
def cmp(name, mine, theirs, kind):
    global checks
    checks += 1
    if kind == "count" and int(mine) != int(theirs): flags.append(f"{name}: mine {mine} vs claimed {theirs}")
    if kind == "pt" and abs(mine - theirs) > 0.1: flags.append(f"{name}: mine {mine:.3f} vs claimed {theirs:.3f} pts")
    if kind == "ci" and abs(mine - theirs) > 0.5: flags.append(f"{name}: mine {mine:.3f} vs claimed {theirs:.3f} pts (CI)")
    if kind == "usd" and abs(mine - theirs) > 1e-4: flags.append(f"{name}: mine {mine:.5f} vs claimed {theirs:.5f}")
    if kind == "rel" and abs(mine - theirs) > 0.01 * abs(theirs): flags.append(f"{name}: mine {mine:.3f} vs claimed {theirs:.3f}")
cmp("n", n, C["n"], "count")
cmp("acc", 100 * out["acc"]["mean"], 100 * C["acc"]["mean"], "pt")
for i in range(2): cmp(f"acc ci[{i}]", 100 * out["acc"]["ci95"][i], 100 * C["acc"]["ci95"][i], "ci")
cmp("cost", out["cost_per_consultation_usd"], C["cost_per_consultation_usd"], "usd")
for k in ("calls", "truncated_calls", "retries", "abstained_proposals", "final_missing", "games_with_lt5_questions"):
    cmp(k, out[k], C[k], "count")
cmp("truncated_share", 100 * out["truncated_share"], 100 * C["truncated_share"], "pt")
for k in bytr:
    cmp(f"bytr {k} n", bytr[k]["n"], C["acc_by_truncations_exploratory"][k]["n"], "count")
    cmp(f"bytr {k} acc", 100 * bytr[k]["acc"], 100 * C["acc_by_truncations_exploratory"][k]["acc"], "pt")
for P, res in out["vs"].items():
    cv = C["vs"][P]
    for label in ("team", "best_member_" + res["best_member"]):
        if label not in cv: flags.append(f"{P}: claimed has no {label}"); continue
        m, t = res[label], cv[label]
        cmp(f"{P} {label} other", 100 * m["other"], 100 * t["other"], "pt")
        cmp(f"{P} {label} diff", m["diff_pts"], t["diff_pts"], "pt")
        for i in range(2): cmp(f"{P} {label} ci[{i}]", m["ci95_pts"][i], t["ci95"][i], "ci")  # claimed diff CIs already in pts
        cmp(f"{P} {label} a_only", m["mcnemar"]["a_only"], t["mcnemar"]["a_only"], "count")
        cmp(f"{P} {label} b_only", m["mcnemar"]["b_only"], t["mcnemar"]["b_only"], "count")
        cmp(f"{P} {label} p", m["mcnemar"]["p"], t["mcnemar"]["p"], "rel")
    cmp(f"{P} team cost", res["team_cost_usd_analysis_json"], cv["team_cost_usd"], "usd")
    cmp(f"{P} solo cost", res["solo_cost_usd_analysis_json"], cv["solo_cost_usd"], "usd")
    cmp(f"{P} cost ratio", res["cost_ratio_anchor_over_team"], cv["cost_ratio_anchor_over_team"], "rel")
out["comparison"] = {"checks": checks, "flags": flags}
json.dump(out, open(R("results/analysis/verify/verify_anchor.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("acc", "cost_per_consultation_usd", "calls", "truncated_calls", "retries",
      "abstained_proposals", "final_missing", "games_with_lt5_questions", "acc_by_truncations", "correct_vs_final_mismatch")}, indent=0))
for P, r in out["vs"].items():
    for label in ("team", "best_member_" + r["best_member"]):
        m = r[label]
        print(P, label, f"{m['diff_pts']:+.1f} [{m['ci95_pts'][0]:+.1f},{m['ci95_pts'][1]:+.1f}] {m['mcnemar']} cost_raw {m['other_cost_raw_usd']:.4f}")
    print(P, "ratio", round(r["cost_ratio_anchor_over_team"], 2), "raw", round(r["cost_ratio_anchor_over_team_raw"], 2),
          "best raw cost", round(r["best_member_cost_raw_usd"], 4), "x", round(r["cost_ratio_anchor_over_best_member_raw"], 1),
          "solo-mean x", round(r["cost_ratio_anchor_over_solo_mean"], 1))
print("checks", checks, "flags", flags)
