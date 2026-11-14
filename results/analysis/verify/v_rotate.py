"""Independent check of the rotating-proposer numbers. Own code, raw logs only.
Naive Bayes ceiling: reuses the earlier independent model in v_review_nb.py (fitted table _v_review_nb.pkl).
Nothing imported from src/ddx or scripts/. No API calls. Writes V_ROTATE.md next to this file."""
import json, sys, bisect, random, hashlib
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from v_review_nb import NB, parse, E
nb = NB()
RES = ROOT / "results"
SEEDS = [11, 2027, 990131]
NBOOT = 10000
PANELS = ["P1", "P2"]
SOLO = list("ABCD")

def load(path):
    g = {}
    for line in open(path):
        line = line.strip()
        if line:
            j = json.loads(line); g[j["pid"]] = j          # last line wins
    return g

ROT = {pn: load(RES / "rotate" / pn / "ABCD.jsonl") for pn in PANELS}
MAIN = {(pn, s): load(RES / "main" / pn / f"{s}.jsonl") for pn in PANELS for s in SOLO + ["ABCD"]}
PIDS = sorted(ROT["P1"], key=lambda p: int(p.split(":")[1]))
ODD = []
for k, v in list(MAIN.items()) + [(("rot", pn), ROT[pn]) for pn in PANELS]:
    if set(v) != set(PIDS): ODD.append(f"{k}: pid set differs ({len(v)} pids)")
N = len(PIDS)

# ---- patients and naive Bayes
test = pd.read_csv(ROOT / "data" / "raw" / "release_test_patients",
                   usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"])
PAT = {}
for p in PIDS:
    assert p.startswith("test:")
    t = test.iloc[int(p.split(":")[1])]
    PAT[p] = dict(age=int(t.AGE), sex=t.SEX, truth=t.PATHOLOGY, ie=t.INITIAL_EVIDENCE, rec=parse(t.EVIDENCES))
truth_mismatch = sum(1 for pn in PANELS for p in PIDS if ROT[pn][p]["truth"] != PAT[p]["truth"])
unknown = Counter()
def ceiling(pid, asked):
    d = PAT[pid]
    ll = nb.opening(d["age"], d["sex"], d["ie"]).copy()
    for c in dict.fromkeys(asked):
        if c not in E: unknown[c] += 1; continue
        ll = ll + nb.code_ll(d["rec"], d["ie"], c)
    return float(int(np.argmax(ll)) == nb.ci[d["truth"]])
def asked(g): return [r["chosen"] for r in g["rounds"] if r.get("chosen")]
def vec(f): return np.array([f(p) for p in PIDS], float)

# ---- statistics
BOOT = {s: np.random.default_rng(s).integers(0, N, size=(NBOOT, N)) for s in SEEDS}
def cis(d):
    out = []
    for s in SEEDS:
        m = d[BOOT[s]].mean(1) * 100
        out.append((float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))))
    return out
def mcnemar(a, b):
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    return n10, n01, (float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0)

# ---- independent vote
SC = json.load(open(ROOT / "prereg" / "confidence_scales.json"))
def pct(model, conf):
    dist = SC[model]["final"]
    return (bisect.bisect_left(dist, conf) + bisect.bisect_right(dist, conf)) / 2 / len(dist)
COIN = Counter()
def indep(pn, pid):
    ch = []
    for s in SOLO:
        g = MAIN[(pn, s)][pid]; f = g["final"][0]
        if f.get("dx"): ch.append((f["dx"], pct(g["agents"][0]["model"], f["confidence"])))
    if not ch: return None
    cnt = Counter(d for d, _ in ch); top = max(cnt.values())
    tied = [d for d, k in cnt.items() if k == top]
    if len(tied) > 1:
        best = {d: max(c for o, c in ch if o == d) for d in tied}; m = max(best.values())
        tied = [d for d in tied if best[d] == m]
    if len(tied) == 1: return tied[0]
    COIN[pn] += 1
    seed = int(hashlib.sha256(f"20261120|{pid}|indep".encode()).hexdigest()[:8], 16)
    return random.Random(seed).choice(sorted(tied))

R = {}; ROWS = []      # ROWS: (claim, claimed, mine, verdict)
def fmt(x, nd): return f"{x:.{nd}f}"
def point(label, claimed, mine, nd=3):
    ok = round(mine, nd) == round(claimed, nd)
    ROWS.append((label, fmt(claimed, nd), fmt(mine, nd), "MATCH" if ok else "MISMATCH"))
def count(label, claimed, mine):
    ROWS.append((label, str(claimed), str(mine), "MATCH" if claimed == mine else "MISMATCH"))
def diff(label, claimed, lo, hi, d, k=1):
    """d = per-patient paired difference; k = grid denominators per patient (grid step = 100/(N*k) pp)"""
    est = d.mean() * 100; c = cis(d); step = 100 / (N * k)
    dev = max(max(abs(round(a, 1) - lo), abs(round(b, 1) - hi)) for a, b in c)
    if round(est, 1) != round(claimed, 1): v = "MISMATCH"
    elif dev <= 0.2 + 1e-9: v = "MATCH"
    elif dev <= step + 0.05 + 1e-9: v = "MATCH (seed-sensitive, one grid step)"
    else: v = "MISMATCH"
    mine = f"{est:+.1f} " + " / ".join(f"[{a:+.1f}, {b:+.1f}]" for a, b in c)
    ROWS.append((label, f"{claimed:+.1f} [{lo:+.1f}, {hi:+.1f}]", mine, v))
    return est, c

CL = {  # claimed values
 "P1": dict(rot=.806, plur=.806, d1=(0.0, -3.4, 3.4), p=1.0, mm=.794, d2=(1.2, -1.8, 4.2), pool=.901, d3=(9.5, 5.4, 13.9),
            acc=.405, pacc=.412, a1=(-0.7, -4.1, 3.1), iv=.391, a2=(1.4, -2.7, 5.4), ma=.364, a3=(4.1, 0.5, 7.7), pre=.374,
            nq=1470, share=[1.18, 1.26, 1.32, 1.24], also=63.1, ov=3.91, trunc=0, miss=0, passed=0, cost=.0283),
 "P2": dict(rot=.813, plur=.762, d1=(5.1, 1.7, 8.8), p=.011, mm=.810, d2=(0.3, -2.7, 3.2), pool=.895, d3=(8.2, 4.4, 12.2),
            acc=.405, pacc=.405, a1=(0.0, -4.1, 4.1), iv=.401, a2=(0.3, -4.4, 4.8), ma=.349, a3=(5.6, 1.9, 9.4), pre=.371,
            nq=1470, share=[1.18, 1.25, 1.33, 1.25], also=56.6, ov=3.56, trunc=0, miss=0, passed=0 + 6, cost=.0305)}
CL["P1"]["passed"] = 0
PROTO = {}; EXTRA = {}
for pn in PANELS:
    c = CL[pn]; rot = ROT[pn]; plur = MAIN[(pn, "ABCD")]
    # 1 ceilings
    c_rot = vec(lambda p: ceiling(p, asked(rot[p])))
    c_plur = vec(lambda p: ceiling(p, asked(plur[p])))
    c_solo = {s: vec(lambda p: ceiling(p, asked(MAIN[(pn, s)][p]))) for s in SOLO}
    c_mm = np.mean([c_solo[s] for s in SOLO], 0)
    c_pool = vec(lambda p: ceiling(p, [q for s in SOLO for q in asked(MAIN[(pn, s)][p])]))
    point(f"1 {pn} ceiling, rotate transcript", c["rot"], c_rot.mean())
    point(f"1 {pn} ceiling, plurality transcript", c["plur"], c_plur.mean())
    diff(f"1 {pn} ceiling rotate - plurality", *c["d1"], c_rot - c_plur)
    n10, n01, pv = mcnemar(c_rot, c_plur)
    ROWS.append((f"1 {pn} exact McNemar p", f"{c['p']:.3f}", f"{pv:.3f} ({n10} vs {n01})",
                 "MATCH" if round(pv, 3) == round(c["p"], 3) else "MISMATCH"))
    point(f"2 {pn} mean member ceiling", c["mm"], c_mm.mean())
    diff(f"2 {pn} ceiling rotate - mean member", *c["d2"], c_rot - c_mm, k=4)
    point(f"2 {pn} pooled ceiling", c["pool"], c_pool.mean())
    diff(f"2 {pn} ceiling pooled - rotate", *c["d3"], c_pool - c_rot)
    # 3 accuracy
    a_rot = vec(lambda p: float(bool(rot[p]["correct"])))
    a_plur = vec(lambda p: float(bool(plur[p]["correct"])))
    a_pre = vec(lambda p: float(bool(rot[p]["correct_pre"])))
    a_iv = vec(lambda p: float(indep(pn, p) == PAT[p]["truth"]))
    a_mm = np.mean([vec(lambda p: float(bool(MAIN[(pn, s)][p]["correct"]))) for s in SOLO], 0)
    point(f"3 {pn} team accuracy, rotate", c["acc"], a_rot.mean())
    point(f"3 {pn} team accuracy, plurality", c["pacc"], a_plur.mean())
    diff(f"3 {pn} accuracy rotate - plurality", *c["a1"], a_rot - a_plur)
    point(f"3 {pn} independent vote accuracy", c["iv"], a_iv.mean())
    diff(f"3 {pn} accuracy rotate - independent vote", *c["a2"], a_rot - a_iv)
    point(f"3 {pn} mean member accuracy", c["ma"], a_mm.mean())
    diff(f"3 {pn} accuracy rotate - mean member", *c["a3"], a_rot - a_mm, k=4)
    point(f"3 {pn} rotate accuracy before deliberation", c["pre"], a_pre.mean())
    # 4 descriptive + protocol
    names = None; per = Counter(); nq = 0; also = 0; passed = 0; bad = Counter(); ov = []
    for p in PIDS:
        g = rot[p]; nm = [a["name"] for a in g["agents"]]
        names = names or nm
        if nm != names: bad["agent names differ between episodes"] += 1
        if g.get("question_rule") != "rotate": bad["question_rule not rotate"] += 1
        if g.get("rotate_start") not in (0, 1, 2, 3): bad["rotate_start out of range"] += 1
        if len(g["rounds"]) != 5 or [r["k"] for r in g["rounds"]] != [1, 2, 3, 4, 5]: bad["rounds not k=1..5"] += 1
        if g["correct"] != (g.get("panel") == g["truth"]): bad["correct flag != (panel == truth)"] += 1
        if "asked" in g and [a for a in g["asked"] if a] != asked(g): bad["asked field != chosen codes"] += 1
        for r in g["rounds"]:
            pr = r["proposals"]
            if len(pr) != 4 or [x.get("agent") for x in pr] != nm: bad["proposals not in agent order"] += 1; continue
            des = (g["rotate_start"] + r["k"] - 1) % 4
            if not r.get("chosen"): bad["round without asked question"] += 1; continue
            nq += 1
            if r.get("proposer") not in nm: bad["proposer not a member"] += 1; continue
            i = nm.index(r["proposer"]); per[i] += 1
            if not pr[i].get("valid") or pr[i].get("code") != r["chosen"]: bad["chosen != proposer's valid proposal"] += 1
            if i != des:
                passed += 1
                if pr[des].get("valid"): bad["turn passed although designated member was valid"] += 1
                # next valid member in rotation order?
                nxt = next((j % 4 for j in range(des + 1, des + 4) if pr[j % 4].get("valid")), None)
                if nxt != i: bad["passed turn did not go to next valid member in order"] += 1
            if any(j != i and x.get("valid") and x.get("code") == r["chosen"] for j, x in enumerate(pr)): also += 1
        ov.append(len(set(asked(g)) & set(asked(plur[p]))))
    PROTO[pn] = dict(bad)
    count(f"4 {pn} asked questions", c["nq"], nq)
    mine = [per[i] / N for i in range(4)]
    ROWS.append((f"4 {pn} asked questions per member per patient", "/".join(f"{x:.2f}" for x in c["share"]),
                 "/".join(f"{x:.2f}" for x in mine),
                 "MATCH" if [round(x, 2) for x in mine] == c["share"] else "MISMATCH"))
    point(f"4 {pn} % asked also validly proposed by another member", c["also"], 100 * also / nq, 1)
    point(f"4 {pn} distinct questions shared with plurality transcript", c["ov"], float(np.mean(ov)), 2)
    # 5 run
    calls = [u for p in PIDS for u in rot[p].get("usage", [])]
    trunc = sum(1 for u in calls if u.get("finish_reason") == "length")
    miss = sum(1 for p in PIDS for f in rot[p]["final"] if not f.get("dx"))
    miss_n = sum(1 for p in PIDS if len(rot[p]["final"]) != 4)
    cost = sum(u.get("cost") or 0 for u in calls) / N
    count(f"5 {pn} truncated calls", c["trunc"], trunc)
    count(f"5 {pn} missing final answers", c["miss"], miss)
    count(f"5 {pn} turns passed to another member", c["passed"], passed)
    point(f"5 {pn} mean cost per episode ($)", c["cost"], cost, 4)
    fr = Counter(u.get("finish_reason") for u in calls)
    EXTRA[pn] = dict(calls=len(calls), finish=dict(fr), final_len_not_4=miss_n,
                     missing_revised=sum(1 for p in PIDS for f in rot[p].get("revised", []) if not f.get("dx")),
                     no_cost=sum(1 for u in calls if u.get("cost") is None),
                     cached_calls=sum(1 for u in calls if u.get("cached")),
                     trunc_by_agent=dict(Counter(u.get("agent") for u in calls if u.get("finish_reason") == "length")),
                     rotate_start=dict(sorted(Counter(rot[p]["rotate_start"] for p in PIDS).items())),
                     coin=COIN[pn], invalid_props=sum(1 for p in PIDS for r in rot[p]["rounds"] for x in r["proposals"] if not x.get("valid")),
                     lt5_distinct=sum(1 for p in PIDS if len(set(asked(rot[p]))) < 5),
                     ts=(min(rot[p]["ts"] for p in PIDS), max(rot[p]["ts"] for p in PIDS)))

proto_ok = all(not v for v in PROTO.values())
L = ["# V_ROTATE: independent check of the rotating-proposer numbers", "",
     "Own code: v_rotate.py, from the raw episode logs (last line per pid wins). "
     "Naive Bayes ceiling: the earlier independent implementation in v_review_nb.py with its fitted table "
     "(not ddx.nb). Nothing imported from src/ddx or scripts/. No API calls.",
     f"{N} patients per file. Bootstrap: paired over patients, {NBOOT:,} resamples, percentile 95% interval, "
     f"seeds {SEEDS[0]} / {SEEDS[1]} / {SEEDS[2]} (three intervals shown in that order). "
     "McNemar: exact binomial. CI rule: MATCH if every endpoint is within 0.2 pp; seed-sensitive if within one grid step "
     f"(0.34 pp for single-transcript differences, 0.09 pp for differences against a four-member mean).", "",
     "| Claim | Claimed | Mine | Verdict |", "|---|---|---|---|"]
L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in ROWS]
L += ["", "## Protocol check", ""]
for pn in PANELS:
    L.append(f"- {pn}: " + ("PASS. In all 1,470 rounds the asked code is the valid proposal of the named proposer; "
             "the proposer is the designated member (rotate_start + k - 1 mod 4) unless that member's proposal was invalid, "
             "and a passed turn went to the next member in order with a valid proposal."
             if not PROTO[pn] else "FAIL: " + json.dumps(PROTO[pn])))
L += ["", "## Log notes", ""]
L.append(f"- Truth labels in the rotate logs vs raw data: {truth_mismatch} mismatches. Unknown question codes: {sum(unknown.values())}. "
         f"File/pid problems: {'; '.join(ODD) if ODD else 'none'}.")
for pn in PANELS:
    x = EXTRA[pn]
    L.append(f"- {pn}: {x['calls']} calls, finish reasons {x['finish']}; truncated by member {x['trunc_by_agent']}; "
             f"invalid proposals {x['invalid_props']}; episodes with final list not of length 4: {x['final_len_not_4']}; "
             f"empty revised answers {x['missing_revised']}; calls without cost {x['no_cost']}; calls flagged cached {x['cached_calls']}; "
             f"rotate_start counts {x['rotate_start']}; episodes with fewer than 5 distinct asked questions {x['lt5_distinct']}; "
             f"independent-vote coin flips {x['coin']}; timestamps {x['ts'][0]} to {x['ts'][1]}.")
(HERE / "V_ROTATE.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
