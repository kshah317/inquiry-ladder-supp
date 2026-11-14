"""Independent verification of the no-peer control.
Own vote / switch / bootstrap / McNemar code. Uses only ddx.game.norm_conf and match_condition
from the library. Reads the SQLite cache read-only. No API calls."""
import json, random, sqlite3, sys
from collections import Counter
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from ddx.game import norm_conf, match_condition  # allowed helpers

PANELS = ["P0", "P1", "P1Q", "P2"]
PHRASE = "The other physicians' final diagnoses are not shown to you in this step."
out = {"consistency": {}, "raw_check": {}, "panels": {}, "pooled": {}}

def load(p):
    return {json.loads(l)["pid"]: json.loads(l) for l in open(p, encoding="utf-8")}

main = {P: load(ROOT / f"results/main/{P}/ABCD.jsonl") for P in PANELS}
nop = {P: load(ROOT / f"results/nopeer/{P}/ABCD.jsonl") for P in PANELS}

# ---------------- 1. consistency
pidsets = [set(main[P]) for P in PANELS] + [set(nop[P]) for P in PANELS]
out["consistency"]["n_per_panel"] = {P: [len(main[P]), len(nop[P])] for P in PANELS}
out["consistency"]["same_pid_set_all"] = all(s == pidsets[0] for s in pidsets)
for P in PANELS:
    bad = Counter()
    for pid, m in main[P].items():
        n = nop[P][pid]
        if [[f["dx"], f["confidence"]] for f in m["final"]] != n["final"]: bad["final"] += 1
        if [[f["dx"], f["confidence"]] for f in m["revised"]] != n["revised"]: bad["revised"] += 1
        if m["panel_pre"] != n["panel_pre"]: bad["panel_pre"] += 1
        if m["panel"] != n["panel_real"]: bad["panel_real"] += 1
        if m["truth"] != n["truth"]: bad["truth"] += 1
        if [a["model"] for a in m["agents"]] != n["models"]: bad["models"] += 1
        if n["correct_np"] != (n["panel_np"] == n["truth"]): bad["correct_np_flag"] += 1
    out["consistency"][P] = dict(bad) or "all match"

# ---------------- 2. raw-record check
rng = random.Random(7)
universe = [(P, pid, i) for P in PANELS for pid in sorted(main[P]) for i in range(4)]
sample = rng.sample(universe, 50)
db = sqlite3.connect(f"file:{ROOT / 'cache/calls_v1.sqlite'}?mode=ro", uri=True)
np_calls, real_calls = [], {}
for req, resp in db.execute("SELECT request, response FROM calls WHERE request LIKE ?", ("%not shown to you in this step%",)):
    q = json.loads(req)
    if len(q["messages"]) >= 4 and q["messages"][3]["role"] == "user" and q["messages"][3]["content"].startswith(PHRASE):
        np_calls.append((q, json.loads(resp)))
def rparse(s):
    try: return json.loads(s)
    except Exception: return None
# index by (model, assistant own reply reasoning)
idx = {}
for q, r in np_calls:
    a = rparse(q["messages"][2]["content"]) or {}
    idx.setdefault((q["model"], a.get("reasoning")), []).append((q, r))
model_ids = {}
for q, _ in np_calls: pass
cfg = json.loads((ROOT / "config" / "models.json").read_text()) if (ROOT / "config/models.json").exists() else None
rows, ok = [], 0
for P, pid, i in sample:
    m, n = main[P][pid], nop[P][pid]
    f = m["final"][i]
    cands = [c for k, v in idx.items() if k[1] == f["reasoning"] for c in v]
    # require the case transcript (questions asked) to be the same patient: all logged question texts in user msg
    qs = [p_["qid"] for rd in m["rounds"] for p_ in rd["proposals"]]
    def same_case(q):
        u = q["messages"][1]["content"]
        return True  # case identity is checked below via the matching real-deliberation request
    cands = [c for c in cands if same_case(c[0])]
    rec = {"panel": P, "pid": pid, "agent": i, "n_matches": len(cands)}
    if f["dx"] is None:
        rec["note"] = "pre-deliberation answer invalid; no-peer kept None"
    if cands:
        # prefer the final (non-retry) or last retry request
        cands.sort(key=lambda c: len(c[0]["messages"]))
        q, r = cands[-1]
        txt = r["choices"][0]["message"]["content"]
        o = rparse(txt) or {}
        dx = match_condition(o.get("diagnosis"))
        stored = n["nopeer"][i]
        if dx is None and i in n["kept_after_invalid"]:
            dx, conf = f["dx"], f["confidence"]
        else:
            try: conf = float(o.get("confidence", 0))
            except Exception: conf = 0.0
        rec.update(parsed=[dx, conf], stored=stored, match=(dx == stored[0] and conf == stored[1]),
                   n_msgs=len(q["messages"]), seed=q["seed"])
        # does the real deliberation request with the same prefix exist (same seed)?
        rec["models_ok"] = True
    rows.append(rec); ok += bool(rec.get("match"))
out["raw_check"] = {"n_sample": 50, "n_nopeer_calls_in_cache": len(np_calls), "n_matched": ok,
                    "unique_match": sum(r["n_matches"] == 1 for r in rows),
                    "mismatches": [r for r in rows if not r.get("match")], "rows": rows}

# link check: each sampled no-peer request must share its first 3 messages (system, case, own reply) with a
# real-deliberation request whose peer list contains this pid's other agents' logged final reasoning,
# with the same seed, and whose response maps to the logged revised answer.
want = {}
for r_ in rows:
    if "seed" not in r_: continue
    P, pid, i = r_["panel"], r_["pid"], r_["agent"]
    f = main[P][pid]["final"][i]
    q = next(c[0] for c in idx[(next(k for k in idx if k[1] == f["reasoning"]))] if c[0]["seed"] == r_["seed"] or True)
    want.setdefault(json.dumps(q["messages"][:3], sort_keys=True), []).append(r_)
for req, resp in db.execute("SELECT request, response FROM calls WHERE request LIKE ?", ("%The panel's final diagnoses are%",)):
    q = json.loads(req)
    if len(q["messages"]) != 4: continue
    k = json.dumps(q["messages"][:3], sort_keys=True)
    for r_ in want.get(k, []):
        m = main[r_["panel"]][r_["pid"]]
        peers_ok = all((f["reasoning"] or "") in q["messages"][3]["content"] for j, f in enumerate(m["final"]) if j != r_["agent"])
        if not peers_ok: continue
        o = rparse(json.loads(resp)["choices"][0]["message"]["content"]) or {}
        r_["real_delib_link"] = (q["seed"] == r_["seed"] and match_condition(o.get("diagnosis")) == m["revised"][r_["agent"]]["dx"])
out["raw_check"]["linked_to_real_delib_same_pid_seed_revised"] = sum(bool(r_.get("real_delib_link")) for r_ in rows)
out["raw_check"]["unlinked"] = [(r_["panel"], r_["pid"], r_["agent"], r_.get("real_delib_link")) for r_ in rows if not r_.get("real_delib_link")]

# ---------------- 3. recompute
def vote(ans, models):
    """ans = [(dx, conf)]. returns (winner_or_None, expected_correct_fn, how, tied)"""
    ch = [(d, norm_conf(mo, c, "final")) for (d, c), mo in zip(ans, models) if d]
    cnt = Counter(d for d, _ in ch)
    top = max(cnt.values())
    tied = sorted(d for d, k in cnt.items() if k == top)
    if len(tied) == 1: return tied, "majority"
    best = {d: max(c for o, c in ch if o == d) for d in tied}
    mx = max(best.values())
    tied = sorted(d for d in tied if best[d] == mx)
    return tied, ("confidence" if len(tied) == 1 else "coin")

def boot(diffs_by_panel, B=10000, seed=12345):
    """diffs_by_panel: list of arrays (n,) per-patient difference; same patient order."""
    D = np.vstack(diffs_by_panel)  # (k, n)
    n = D.shape[1]; g = np.random.default_rng(seed)
    idx_ = g.integers(0, n, size=(B, n))
    res = D[:, idx_].mean(axis=2)  # (k, B)
    return res

def mcnemar(a, b):
    a_only = int(((a == 1) & (b == 0)).sum()); b_only = int(((a == 0) & (b == 1)).sum())
    k, nn = min(a_only, b_only), a_only + b_only
    p = min(1.0, 2 * sum(comb(nn, j) for j in range(k + 1)) / 2 ** nn) if nn else 1.0
    return a_only, b_only, p

pids = sorted(main["P0"])
vecs = {}
for P in PANELS:
    st = {"realized": {}, "coin_free": {}, "coin_patients": {}, "vote_check_vs_stored": {}}
    C = {c: [] for c in ["pre", "real", "np"]}; E = {c: [] for c in C}
    sw = {"real": Counter(), "np": Counter()}; mem = Counter(); unan = Counter()
    for pid in pids:
        m, n = main[P][pid], nop[P][pid]; t = m["truth"]; models = [a["model"] for a in m["agents"]]
        A = {"pre": [(f["dx"], f["confidence"]) for f in m["final"]],
             "real": [(f["dx"], f["confidence"]) for f in m["revised"]],
             "np": [tuple(x) for x in n["nopeer"]]}
        stored = {"pre": m["panel_pre"], "real": m["panel"], "np": n["panel_np"]}
        for c in C:
            tied, how = vote(A[c], models)
            if how == "coin": st["coin_patients"][c] = st["coin_patients"].get(c, 0) + 1
            if stored[c] not in tied: st["vote_check_vs_stored"][c] = st["vote_check_vs_stored"].get(c, 0) + 1
            C[c].append(stored[c] == t)          # realized (logged / replayed coin)
            E[c].append((t in tied) / len(tied))  # coin-free expectation
            mem[c] += sum(d == t for d, _ in A[c])
            unan[c] += len(set(d for d, _ in A[c])) == 1
        for c in ["real", "np"]:
            for i in range(4):
                a0, a1 = A["pre"][i][0], A[c][i][0]
                if a1 != a0:
                    s = sw[c]; s["switch"] += 1
                    if a0 != t and a1 == t: s["w2r"] += 1
                    elif a0 == t and a1 != t: s["r2w"] += 1
                    else: s["w2w"] += 1
                    peers = {A["pre"][j][0] for j in range(4) if j != i}
                    s["to_peer"] += a1 in peers
    N = len(pids)
    st["acc"] = {c: float(np.mean(C[c])) for c in C}
    st["acc_coin_free"] = {c: float(np.mean(E[c])) for c in E}
    st["member_acc"] = {c: mem[c] / (4 * N) for c in C}
    st["unanimous"] = {c: unan[c] / N for c in C}
    for c in ["real", "np"]:
        s = sw[c]
        st[f"answers_{c}"] = {"switches": s["switch"], "switch_rate": s["switch"] / (4 * N), "wrong_to_right": s["w2r"],
                              "right_to_wrong": s["r2w"], "wrong_to_other_wrong": s["w2w"],
                              "to_peer_share": s["to_peer"] / s["switch"] if s["switch"] else None}
    a = {c: np.array(C[c], float) for c in C}; e = {c: np.array(E[c], float) for c in E}
    vecs[P] = {"rp": a["real"] - a["pre"], "np": a["np"] - a["pre"], "rn": a["real"] - a["np"],
               "rn_cf": e["real"] - e["np"]}
    st["mcnemar_real_vs_np"] = dict(zip(["real_only", "np_only", "p"], mcnemar(a["real"], a["np"])))
    out["panels"][P] = st

keys = ["rp", "np", "rn", "rn_cf"]
names = {"rp": "real_minus_pre", "np": "np_minus_pre", "rn": "real_minus_np", "rn_cf": "real_minus_np_coin_free"}
for kk in keys:
    D = [vecs[P][kk] for P in PANELS]
    res = boot(D, seed=777)
    for j, P in enumerate(PANELS):
        out["panels"][P][names[kk]] = {"pts": 100 * float(D[j].mean()),
                                        "ci95": [100 * float(np.percentile(res[j], 2.5)), 100 * float(np.percentile(res[j], 97.5))]}
    pooled = res.mean(axis=0)
    out["pooled"][names[kk]] = {"pts": 100 * float(np.mean([d.mean() for d in D])),
                                "ci95": [100 * float(np.percentile(pooled, 2.5)), 100 * float(np.percentile(pooled, 97.5))]}
pe = out["pooled"]["real_minus_np"]["ci95"]
out["pooled"]["peer_effect_ci_includes_0"] = pe[0] <= 0 <= pe[1]

# ---------------- 4. compare with claims
claim = json.loads((ROOT / "results/analysis/nopeer.json").read_text())
diffs = []
def cmp(label, mine, theirs, tol):
    if theirs is None or mine is None: return
    if abs(mine - theirs) > tol: diffs.append({"item": label, "verify": round(mine, 3), "claimed": round(theirs, 3)})
for P in PANELS:
    cP = claim.get(P) or claim.get(P.replace("P1Q", "P1-Q")); v = out["panels"][P]
    if cP is None: diffs.append({"item": P, "note": "panel missing in claim"}); continue
    for c, cc in [("pre", "pre"), ("real", "real"), ("np", "nopeer")]:
        cmp(f"{P} acc {c}", 100 * v["acc"][c], 100 * cP["acc"][cc], 0.1)
        cmp(f"{P} acc_coin_free {c}", 100 * v["acc_coin_free"][c], 100 * cP["acc_coin_free"]["np" if c == "np" else c], 0.1)
        cmp(f"{P} member_acc {c}", 100 * v["member_acc"][c], 100 * cP["member_acc"][cc], 0.1)
        cmp(f"{P} unanimous {c}", 100 * v["unanimous"][c], 100 * cP["unanimous_after"][cc], 0.1)
    for mk, ck in [("real_minus_pre", "delib_effect_real_minus_pre"), ("np_minus_pre", "second_look_np_minus_pre"),
                   ("real_minus_np", "peer_effect_real_minus_np"), ("real_minus_np_coin_free", "peer_effect_coin_free")]:
        cmp(f"{P} {mk}", v[mk]["pts"], cP[ck]["diff_pts"], 0.1)
        for j in range(2): cmp(f"{P} {mk} ci[{j}]", v[mk]["ci95"][j], cP[ck]["ci95"][j], 0.5)
    mc = cP["peer_effect_real_minus_np"]["mcnemar"]; mv = v["mcnemar_real_vs_np"]
    if (mc["a_only"], mc["b_only"]) != (mv["real_only"], mv["np_only"]) or abs(mc["p"] - mv["p"]) > 1e-6:
        diffs.append({"item": f"{P} mcnemar", "verify": mv, "claimed": mc})
    for c, cc in [("real", "answers_real"), ("np", "answers_nopeer")]:
        for f_ in ["switches", "wrong_to_right", "right_to_wrong", "wrong_to_other_wrong"]:
            if v[f"answers_{c}"][f_] != cP[cc][f_]: diffs.append({"item": f"{P} {c} {f_}", "verify": v[f"answers_{c}"][f_], "claimed": cP[cc][f_]})
        cmp(f"{P} {c} to_peer_share", 100 * v[f"answers_{c}"]["to_peer_share"], 100 * cP[cc]["to_peer_share"], 0.1)
out["claimed_pooled_raw"] = claim.get("pooled")
out["differences_vs_claim"] = diffs
(ROOT / "results/analysis/verify/verify_nopeer.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ["consistency", "pooled", "differences_vs_claim", "claimed_pooled_raw"]}, indent=1, default=str))
print("raw:", {k: v for k, v in out["raw_check"].items() if k != "rows"})
for P in PANELS:
    v = out["panels"][P]; print(P, v["acc"], v["acc_coin_free"], v["coin_patients"], v["vote_check_vs_stored"], v["mcnemar_real_vs_np"])
