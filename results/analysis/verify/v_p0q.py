"""Independent verifier for the P0-Q checks. Own code (reuses only the earlier verifier's v_review_nb). No src/ddx, no scripts/, no cache, no API."""
import json, bisect, sys, itertools
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binomtest, t as tdist
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_review_nb import NB, parse, E
ROOT = Path(__file__).resolve().parents[3]; RES = ROOT / "results"
samp = pd.read_csv(ROOT / "data/samples/main.csv")
PIDS = [f"test:{int(r)}" for r in samp.row]; assert len(set(PIDS)) == 294
def load(path):
    g = {}
    for line in open(path):
        line = line.strip()
        if line: j = json.loads(line); g[j["pid"]] = j
    return g
SUBS = ['A','B','C','D','AB','AC','AD','BC','BD','CD','ABC','ABD','ACD','BCD','ABCD']
L = {}; OUT = {}
raw = {s: load(RES/"main/P0Q"/f"{s}.jsonl") for s in "A B C D ABCD".split()}
OUT["1_counts"] = {s: [len(v), sum(p in v for p in PIDS)] for s, v in raw.items()}
for s, v in raw.items(): L[("P0Q", s)] = {p: v[p] for p in PIDS}
for pn in ["P0", "P1", "P1Q", "P2"]:
    for s in SUBS:
        v = load(RES/"main"/pn/f"{s}.jsonl"); L[(pn, s)] = {p: v[p] for p in PIDS}
TRUTH = {p: L[("P0Q","ABCD")][p]["truth"] for p in PIDS}
assert all(TRUTH[p] == t for p, t in zip(PIDS, samp.PATHOLOGY))
# 1 integrity
a, c = L[("P0Q","A")], L[("P2","C")]
OUT["1"] = dict(
    seed_ids=[a[PIDS[0]]["agents"][0]["seed_id"], c[PIDS[0]]["agents"][0]["seed_id"]],
    asked_diff=sum(a[p]["asked"] != c[p]["asked"] for p in PIDS),
    dx_diff=sum(a[p]["final"][0]["dx"] != c[p]["final"][0]["dx"] for p in PIDS),
    correct_diff=sum(a[p]["correct"] != c[p]["correct"] for p in PIDS),
    rounds_diff=sum([r.get("chosen") for r in a[p]["rounds"]] != [r.get("chosen") for r in c[p]["rounds"]] for p in PIDS),
    final_full_diff=sum(a[p]["final"] != c[p]["final"] for p in PIDS),
    correct_vs_truth_bad=sum(bool(g[p]["correct"]) != (g[p]["panel"] == TRUTH[p]) for g in [L[("P0Q",s)] for s in "A B C D ABCD".split()] for p in PIDS))
def vec(f): return np.array([float(f(p)) for p in PIDS])
def corr(pn, s): return vec(lambda p: bool(L[(pn, s)][p]["correct"]))
rng = np.random.default_rng(20261103_2)
BOOT = rng.integers(0, 294, size=(10000, 294))
def ci(d, lv=95):
    m = d[BOOT].mean(1); q = (100-lv)/2
    return [round(float(np.percentile(m, q))*100, 2), round(float(np.percentile(m, 100-q))*100, 2)]
def mcn(x, y):
    b = int(((x==1)&(y==0)).sum()); c_ = int(((x==0)&(y==1)).sum())
    return b, c_, float(binomtest(b, b+c_, .5).pvalue) if b+c_ else 1.0
def tost(d, m=.05):
    n = len(d); se = d.std(ddof=1)/np.sqrt(n); mu = d.mean()
    p1 = 1 - tdist.cdf((mu + m)/se, n-1); p2 = tdist.cdf((mu - m)/se, n-1); return float(max(p1, p2))
def comp(x, y):
    d = x - y; return dict(diff=round(d.mean()*100, 2), ci95=ci(d), ci90=ci(d, 90), mcnemar=mcn(x, y), tost=round(tost(d), 4))
# 2 accuracy
team = corr("P0Q","ABCD")
OUT["2"] = dict(team=team.mean(), pre=vec(lambda p: bool(L[("P0Q","ABCD")][p]["correct_pre"])).mean(),
                pre_recomputed=vec(lambda p: L[("P0Q","ABCD")][p]["panel_pre"] == TRUTH[p]).mean(),
                solos={s: corr("P0Q", s).mean() for s in "ABCD"},
                solos_from_dx={s: vec(lambda p: L[("P0Q",s)][p]["final"][0]["dx"] == TRUTH[p]).mean() for s in "ABCD"})
# 3 independent vote
SC = json.load(open(ROOT/"prereg/confidence_scales.json"))
def pct(model, conf):
    dist = SC[model]["final"]
    if conf is None: return -1
    return (bisect.bisect_left(dist, conf) + bisect.bisect_right(dist, conf))/2/len(dist)
def tied(pn, p):
    ch = [(L[(pn,s)][p]["final"][0].get("dx"), L[(pn,s)][p]["final"][0].get("confidence"), L[(pn,s)][p]["agents"][0]["model"]) for s in "ABCD"]
    ch = [(d, pct(m, c)) for d, c, m in ch if d]
    if not ch: return [None]
    cnt = Counter(d for d, _ in ch); top = max(cnt.values()); t = [d for d, k in cnt.items() if k == top]
    if len(t) == 1: return t
    best = {d: max(c for o, c in ch if o == d) for d in t}; mx = max(best.values())
    return sorted(d for d in t if best[d] == mx)
TW = {p: tied("P0Q", p) for p in PIDS}
coinp = [p for p in PIDS if len(TW[p]) > 1]
lo = vec(lambda p: all(w == TRUTH[p] for w in TW[p])); hi = vec(lambda p: TRUTH[p] in TW[p])
exp = vec(lambda p: sum(w == TRUTH[p] for w in TW[p])/len(TW[p]))
r = np.random.default_rng(3); seeded = vec(lambda p: r.choice(TW[p]) == TRUTH[p])
OUT["3"] = dict(coin_patients=len(coinp), coin_with_truth=int(sum(hi-lo)), range=[lo.mean(), hi.mean()], expected=exp.mean(),
                seeded=seeded.mean(), team_minus_seeded=comp(team, seeded), team_minus_lo=comp(team, lo), team_minus_hi=comp(team, hi),
                coin_detail=[(p, TW[p], TRUTH[p]) for p in coinp])
# 4-7
p1q = corr("P1Q","ABCD"); p0 = corr("P0","ABCD"); p1 = corr("P1","ABCD")
OUT["4"] = comp(p1q, team); OUT["5"] = comp(team, p0)
dd = (p1q - team) - (p1 - p0); OUT["6"] = dict(diff=round(dd.mean()*100, 2), ci95=ci(dd))
best = max("ABCD", key=lambda s: corr("P0Q", s).mean()); OUT["7"] = dict(best=best, **comp(team, corr("P0Q", best)))
# 8 error overlap
dx = np.array([[L[("P0Q",s)][p]["final"][0].get("dx") or "<none>" for p in PIDS] for s in "ABCD"], dtype=object)
tr = np.array([TRUTH[p] for p in PIDS], dtype=object); W = dx != tr[None]
pairs = list(itertools.combinations(range(4), 2))
def ov(D, Wm):
    nb_ = ns = 0; rat = []
    for i, j in pairs:
        b = Wm[i] & Wm[j]; s_ = b & (D[i] == D[j]) & (D[i] != "<none>")
        nb_ += b.sum(); ns += s_.sum(); rat.append(s_.sum()/max(b.sum(), 1))
    return ns/nb_, float(np.mean(rat)), [round(x, 3) for x in rat], int(nb_), int(ns)
obs = ov(dx, W)
rg = np.random.default_rng(99); nul_pool = []; nul_mean = []
strata = [np.flatnonzero(tr == c_) for c_ in np.unique(tr)]
for _ in range(2000):
    D = np.empty_like(dx); Wm = np.empty_like(W)
    for i in range(4):
        perm = np.arange(294)
        for ix in strata: perm[ix] = rg.permutation(ix)
        D[i] = dx[i, perm]; Wm[i] = W[i, perm]
    o = ov(D, Wm); nul_pool.append(o[0]); nul_mean.append(o[1])
OUT["8"] = dict(pooled_counts=obs[0], mean_of_pairs=obs[1], per_pair=obs[2], both=obs[3], same=obs[4],
                null_pooled=float(np.mean(nul_pool)), null_mean_of_pairs=float(np.mean(nul_mean)),
                null95=[float(x) for x in np.percentile(nul_mean, [2.5, 97.5])],
                p_upper=float((1 + (np.array(nul_mean) >= obs[1] - 1e-12).sum())/2001))
# 10 deliberation
rows = []
for p in PIDS:
    g = L[("P0Q","ABCD")][p]
    for i in range(4):
        pre, post = g["final"][i].get("dx"), g["revised"][i].get("dx")
        rows.append((pre == TRUTH[p], post == TRUTH[p], pre != post))
R = np.array(rows)
OUT["10"] = dict(n=len(R), switch=R[:,2].mean(), w2r=int((~R[:,0] & R[:,1]).sum()), r2w=int((R[:,0] & ~R[:,1]).sum()),
                 acc_pre=R[:,0].mean(), acc_post=R[:,1].mean())
# 11 question votes
k = Counter(); nr = 0; none_ = 0
for p in PIDS:
    for rd in L[("P0Q","ABCD")][p]["rounds"]:
        ch = rd.get("chosen")
        if not ch: none_ += 1; continue
        nr += 1; k[sum(1 for q in rd["proposals"] if q.get("valid") and q.get("code") == ch)] += 1
OUT["11"] = dict(n_rounds=nr, null_chosen=none_, share={i: k[i]/nr for i in range(5)})
# 9 NB
nb = NB(); test = pd.read_csv(ROOT/"data/raw/release_test_patients", usecols=["AGE","SEX","PATHOLOGY","EVIDENCES","INITIAL_EVIDENCE"])
PAT = {}
for rr, pid in zip(samp.row, PIDS):
    t = test.iloc[int(rr)]; assert t.PATHOLOGY == TRUTH[pid]
    PAT[pid] = (nb.opening(int(t.AGE), t.SEX, t.INITIAL_EVIDENCE), parse(t.EVIDENCES), t.INITIAL_EVIDENCE)
def score(pid, asked):
    op, rec, ie = PAT[pid]; ll = op.copy()
    for c_ in dict.fromkeys(x for x in asked if x):
        if c_ in E: ll = ll + nb.code_ll(rec, ie, c_)
    return float(ll.argmax() == nb.ci[TRUTH[pid]])
def asked_of(g): return [rd.get("chosen") for rd in g["rounds"] if rd.get("chosen")]
OUT["9"] = {s: vec(lambda p: score(p, asked_of(L[("P0Q",s)][p]))).mean() for s in "A B C D ABCD".split()}
OUT["9"]["union"] = vec(lambda p: score(p, [c_ for s in "ABCD" for c_ in asked_of(L[("P0Q",s)][p])])).mean()
json.dump(OUT, open(Path(__file__).resolve().parent/"v_p0q_out.json", "w"), indent=1, default=str)
for kk, v in OUT.items(): print(kk, json.dumps(v, default=str)[:900])
