"""Independent verifier for the review analyses. Own code only (no src/ddx imports). No cache, no API.
Writes v_review_out.json; V_REVIEW.md is written by v_review_report.py."""
import json, bisect, random, sys, itertools
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_review_nb import NB, parse, TOP, E, PARENT, fit
ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
OUT = {}
if not (Path(__file__).resolve().parent / "_v_review_nb.pkl").exists(): fit()
nb = NB()

# ---------- patients
samp = pd.read_csv(ROOT / "data/samples/main.csv")
assert (samp.split == "test").all() and len(samp) == 294
test = pd.read_csv(ROOT / "data/raw/release_test_patients", usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"])
PIDS = []; PAT = {}
for r in samp.row:
    t = test.iloc[int(r)]; pid = f"test:{int(r)}"; PIDS.append(pid)
    PAT[pid] = dict(age=int(t.AGE), sex=t.SEX, truth=t.PATHOLOGY, ie=t.INITIAL_EVIDENCE, rec=parse(t.EVIDENCES))
assert (samp.PATHOLOGY.values == np.array([PAT[p]["truth"] for p in PIDS])).all()
OPEN = {p: nb.opening(d["age"], d["sex"], d["ie"]) for p, d in PAT.items()}
_CL = {}
def cll(pid, c):
    k = (pid, c)
    if k not in _CL: _CL[k] = nb.code_ll(PAT[pid]["rec"], PAT[pid]["ie"], c)
    return _CL[k]
UNKNOWN = Counter()
def score(pid, asked):
    ll = OPEN[pid].copy()
    for c in dict.fromkeys(a for a in asked if a):
        if c not in E: UNKNOWN[c] += 1; continue
        ll += cll(pid, c)
    post = np.exp(ll - ll.max()); post /= post.sum()
    t = nb.ci[PAT[pid]["truth"]]
    return float(post.argmax() == t), float(post[t])

# ---------- logs
def load(path):
    g = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            j = json.loads(line); g[j["pid"]] = j
    return {p: g[p] for p in PIDS if p in g}
def asked_of(g): return [r.get("chosen") for r in g.get("rounds", []) if r.get("chosen")]
SOLO = list("ABCD")
PANELS = ["P0", "P1", "P1Q", "P2"]
L = {(pn, s): load(RES / "main" / pn / f"{s}.jsonl") for pn in PANELS for s in SOLO + ["ABCD"]}
for k, v in L.items(): assert len(v) == 294, (k, len(v))
R1 = {(pn, s): load(RES / "main_r1" / pn / f"{s}.jsonl") for pn in ["P0", "P1", "P2"] for s in SOLO + ["ABCD"]}
for k, v in R1.items(): assert len(v) == 294, (k, len(v))
ANC = load(RES / "main/ANCHOR/A.jsonl"); assert len(ANC) == 294
RP = {(pn, s): load(RES / "replay" / pn / f"{s}.jsonl") for pn in PANELS for s in SOLO + ["ABCD"]}
for k, v in RP.items(): assert len(v) == 294, (k, len(v))
direct_followup = sum(1 for v in list(L.values()) + [ANC] for g in v.values() for c in asked_of(g) if c in PARENT)
asked_field_mismatch = sum(1 for v in list(L.values()) + [ANC] for g in v.values() if "asked" in g and [a for a in g["asked"] if a] != asked_of(g))
OUT["integrity"] = dict(direct_followup_asks=direct_followup, asked_field_vs_rounds_chosen_mismatch=asked_field_mismatch)

def vec(f): return np.array([f(p) for p in PIDS], float)
rng = np.random.default_rng(20261103)
BOOT = rng.integers(0, 294, size=(10000, 294))
def ci(d):
    m = d[BOOT].mean(1); return [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)]

# ---------- 1 NB sanity
o = [score(p, []) for p in PIDS]
full = [score(p, TOP) for p in PIDS]
OUT["1"] = dict(open_top1=np.mean([a for a, _ in o]), open_ptruth=np.mean([b for _, b in o]),
                full_top1=np.mean([a for a, _ in full]), full_ptruth=np.mean([b for _, b in full]), n_top_level=len(TOP))

# ---------- 2/3/4 NB of transcripts
NBT = {}
for (pn, s), d in L.items():
    NBT[(pn, s)] = vec(lambda p: score(p, asked_of(d[p]))[0])
OUT["2"] = {pn: {s: NBT[(pn, s)].mean() for s in SOLO + ["ABCD"]} for pn in PANELS}
OUT["3"] = {}; OUT["4"] = {}
for pn in PANELS:
    u = vec(lambda p: score(p, [c for s in SOLO for c in asked_of(L[(pn, s)][p])])[0])
    OUT["3"][pn] = u.mean()
    d = NBT[(pn, "ABCD")] - np.mean([NBT[(pn, s)] for s in SOLO], 0)
    OUT["4"][pn] = dict(diff=d.mean() * 100, ci=ci(d))

# ---------- 5 anchor
an = vec(lambda p: score(p, asked_of(ANC[p]))[0])
OUT["5"] = dict(top1=an.mean(), lt5=sum(len(asked_of(g)) < 5 for g in ANC.values()),
                rounds_hist=dict(Counter(len(asked_of(g)) for g in ANC.values())))

# ---------- 6 question votes
OUT["6"] = {}
for pn in PANELS:
    k = Counter(); vt = Counter(); nr = 0
    for g in L[(pn, "ABCD")].values():
        for r in g["rounds"]:
            ch = r.get("chosen")
            if not ch: k["none"] += 1; continue
            nr += 1
            k[sum(1 for q in r["proposals"] if q.get("valid") and q.get("code") == ch)] += 1
            vt[r.get("vote")] += 1
    OUT["6"][pn] = dict(n_rounds=nr, share={str(i): k[i] / nr for i in range(0, 5)}, null_chosen=k["none"],
                        vote={str(a): b / nr for a, b in vt.items()})

# ---------- vote with tie rule
SC = json.load(open(ROOT / "prereg/confidence_scales.json"))
def pct(model, conf):
    dist = (SC.get(model) or {}).get("final") or []
    if conf is None: return -1
    if not dist: return conf / 100
    return (bisect.bisect_left(dist, conf) + bisect.bisect_right(dist, conf)) / 2 / len(dist)
def tied_winners(choices):
    """choices = [(dx, conf, model)]; returns list of tied winners after plurality + max percentile"""
    ch = [(d, pct(m, c)) for d, c, m in choices if d]
    if not ch: return [None]
    cnt = Counter(d for d, _ in ch); top = max(cnt.values())
    t = [d for d, k in cnt.items() if k == top]
    if len(t) == 1: return t
    best = {d: max(c for o, c in ch if o == d) for d in t}; m = max(best.values())
    return sorted(d for d in t if best[d] == m)
def solo_choice(g):
    f = g["final"][0]; return (f.get("dx"), f.get("confidence"), g["agents"][0]["model"])
COND = set(nb.conds)
bad_dx = sum(1 for v in L.values() for g in v.values() for f in g["final"] if f.get("dx") and f["dx"] not in COND)
OUT["integrity"]["final_dx_not_condition"] = bad_dx

# ---------- 7 independent vote wrong -> >=2 support
OUT["7"] = {}
for pn in PANELS:
    lo_w = hi_w = 0; ge2_lo = ge2_hi = 0; coin = 0; seeded_w = 0; seeded_ge2 = 0
    r = random.Random(7)
    for p in PIDS:
        chs = [solo_choice(L[(pn, s)][p]) for s in SOLO]
        tw = tied_winners(chs); truth = PAT[p]["truth"]
        cnt = Counter(d for d, _, _ in chs if d)
        sup = cnt[tw[0]] if tw[0] else 0   # same count for every tied winner
        if len(tw) > 1: coin += 1
        wrong_opts = [w for w in tw if w != truth]
        # always wrong?
        if not (truth in tw): lo_w += 1; ge2_lo += sup >= 2
        if wrong_opts: hi_w += 1; ge2_hi += sup >= 2
        w = r.choice(tw)
        if w != truth: seeded_w += 1; seeded_ge2 += sup >= 2
    OUT["7"][pn] = dict(coin_patients=coin, wrong_min=lo_w, ge2_at_min=ge2_lo, wrong_max=hi_w, ge2_at_max=ge2_hi,
                        seeded=[seeded_ge2, seeded_w])

# ---------- 8 SC8
G8 = [L[("P0", s)] for s in SOLO] + [R1[("P0", s)] for s in SOLO]
models8 = Counter(g["agents"][0]["model"] for d in G8 for g in d.values())
sc_lo = sc_hi = 0; coin = 0; sc_seed = np.zeros(294); sc_lo_v = np.zeros(294); r = random.Random(8)
costs = []
for i, p in enumerate(PIDS):
    chs = [solo_choice(d[p]) for d in G8]
    tw = tied_winners(chs); truth = PAT[p]["truth"]
    if len(tw) > 1: coin += 1
    sc_lo += all(w == truth for w in tw); sc_hi += truth in tw
    sc_lo_v[i] = all(w == truth for w in tw)
    sc_seed[i] = r.choice(tw) == truth
    costs.append(sum(u.get("cost") or 0 for d in G8 for u in d[p].get("usage", [])))
anc_c = vec(lambda p: float(bool(ANC[p]["correct"])))
anc_chk = sum(1 for p in PIDS if bool(ANC[p]["correct"]) != (ANC[p].get("panel") == PAT[p]["truth"]))
def mcn(a, b):
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    return n10, n01, float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
d = anc_c - sc_seed
OUT["8"] = dict(models=dict(models8), coin_patients=coin, top1_range=[sc_lo / 294, sc_hi / 294], top1_seeded=sc_seed.mean(),
                mean_cost=float(np.mean(costs)), anchor_top1=anc_c.mean(), anchor_correct_vs_panel_mismatch=anc_chk,
                diff_seeded=d.mean() * 100, ci_seeded=ci(d), mcnemar_seeded=mcn(anc_c, sc_seed),
                diff_range=[(anc_c.mean() - sc_hi / 294) * 100, (anc_c.mean() - sc_lo / 294) * 100],
                mcnemar_lowcoin=mcn(anc_c, sc_lo_v))

# ---------- 9 decomposition
def cv(d, key="correct"): return vec(lambda p: float(bool(d[p][key])))
v0 = np.mean([cv(L[("P0", s)]) for s in SOLO]); e0 = np.mean([cv(RP[("P0", s)]) for s in SOLO])
OUT["9"] = {}
for pn in ["P1", "P2", "P1Q"]:
    v = cv(L[(pn, "ABCD")]).mean(); vp = cv(L[(pn, "ABCD")], "correct_pre").mean(); e = cv(RP[(pn, "ABCD")]).mean()
    OUT["9"][pn] = dict(total=(v - v0) * 100, asking=(e - e0) * 100, read_vote=(vp - e) * 100, delib=(v - vp) * 100,
                        residual=(e0 - v0) * 100)
rp_pid_truth_mismatch = sum(1 for d in RP.values() for p, g in d.items() if g.get("truth") and g["truth"] != PAT[p]["truth"])
OUT["integrity"]["replay_truth_mismatch"] = rp_pid_truth_mismatch

# ---------- 10 best member from r1
OUT["10"] = {}
for pn in ["P0", "P1", "P2"]:
    acc1 = {s: cv(R1[(pn, s)]).mean() for s in SOLO}
    b = max(SOLO, key=lambda s: (acc1[s], -SOLO.index(s)))
    d = cv(L[(pn, "ABCD")]) - cv(L[(pn, b)])
    OUT["10"][pn] = dict(r1_acc=acc1, best=b, diff=d.mean() * 100, ci=ci(d))

# ---------- 11 billed rates
agg = {}
for d in [L[("P2", s)] for s in SOLO] + [ANC]:
    for g in d.values():
        for u in g.get("usage", []):
            m = u.get("model"); a = agg.setdefault(m, Counter())
            a["pt"] += u.get("prompt_tokens") or 0; a["ct"] += u.get("completion_tokens") or 0
            a["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            cd = u.get("cost_details") or {}
            a["pc"] += cd.get("upstream_inference_prompt_cost") or 0; a["cc"] += cd.get("upstream_inference_completions_cost") or 0
            a["n"] += 1
OUT["11"] = {m: dict(n=a["n"], in_per_M=a["pc"] / a["pt"] * 1e6, out_per_M=a["cc"] / a["ct"] * 1e6 if a["ct"] else None,
                     cached_share=a["cached"] / a["pt"]) for m, a in agg.items() if a["pt"]}
OUT["integrity"]["unknown_codes"] = dict(UNKNOWN)

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(conv(OUT), open(Path(__file__).resolve().parent / "v_review_out.json", "w"), indent=1)
print(json.dumps(conv(OUT), indent=1))
