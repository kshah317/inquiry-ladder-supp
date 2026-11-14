#!/usr/bin/env python3
"""Independent re-computation of the v10 paper numbers from the raw episode logs.

Reads only results/main/<TEAM>/<COALITION>.jsonl and prereg/confidence_scales.json.
Standard library + numpy only. Writes V_V10.md next to this script.
"""
import bisect
import hashlib
import itertools
import json
import math
import os
import random
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
MAIN = os.path.join(ROOT, "results", "main")
SCALES = json.load(open(os.path.join(ROOT, "prereg", "confidence_scales.json")))

MEMBERS = "ABCD"
COALS = ["".join(c) for r in range(1, 5) for c in itertools.combinations(MEMBERS, r)]
B = 10000
SEEDS = (101, 202, 303)
NOTES = []


# Written after the numbers above were final, from one reading of results/analysis/v10.json.
# This script does not read that file.
POST_NOTE = [
    "## Comparison with results/analysis/v10.json (opened once, after the numbers above were final)",
    "",
    "Every v10.json value that has a recomputed counterpart in the table agrees with it after rounding, except as noted below. Values in v10.json that the table does not cover (for example vote minus mean member intervals, Shapley intervals other than P1 A and B, retention for P1, P0-Q, P1-Q) were not recomputed.",
    "",
    "- P0-Q team minus mean member: v10.json has the interval [1.8, 8.9], the paper text has [+2.0, +8.8]. The recomputed interval is [+1.87 to +1.96, +8.84], which is within 0.2 pp of both, but the paper and v10.json do not agree with each other for this row.",
    "- P2 share linked to the previous question: v10.json stores 0.785 (three decimals). The unrounded value is 923/1176 = 0.78486, so the percentage is 78%, not 79%. The 79% looks like a second rounding of 0.785.",
    "- P0 full team minus mean 3-agent coalition: v10.json has 0.3, recomputed +0.26, same after rounding. P2 confidence share: 0.317 in both.",
]


def load(team, coal):
    """pid -> episode, last line wins."""
    out, dup = {}, 0
    with open(os.path.join(MAIN, team, coal + ".jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep["pid"] in out:
                dup += 1
            out[ep["pid"]] = ep
    if dup:
        NOTES.append("%s/%s: %d repeated pid lines (last line kept)" % (team, coal, dup))
    return out


DATA = {}


def get(team, coal):
    if (team, coal) not in DATA:
        DATA[(team, coal)] = load(team, coal)
    return DATA[(team, coal)]


PIDS = sorted(get("P0", "ABCD").keys())
assert len(PIDS) == 294


def pids_for(team):
    return sorted(get("CHECK", "ABCD").keys()) if team == "CHECK" else PIDS


def cvec(team, coal):
    eps = get(team, coal)
    pl = pids_for(team)
    miss = [p for p in pl if p not in eps]
    if miss:
        NOTES.append("%s/%s: %d patients missing" % (team, coal, len(miss)))
    return np.array([1.0 if eps[p]["correct"] else 0.0 for p in pl])


def ci(d, seed):
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    m = d[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def ci3(d):
    return [ci(d, s) for s in SEEDS]


def ci_ref(d, seed=7, total=200000, chunk=10000):
    """large-B reference interval, used only to judge seed-sensitive rows."""
    rng = np.random.default_rng(seed)
    n = len(d)
    ms = []
    for _ in range(total // chunk):
        ms.append(d[rng.integers(0, n, size=(chunk, n))].mean(axis=1))
    lo, hi = np.percentile(np.concatenate(ms), [2.5, 97.5])
    return float(lo), float(hi)


# ---------- independent vote ----------
def norm_conf(model, conf):
    sc = SCALES.get(model, {}).get("final")
    if conf is None:
        conf = 0.0
    if not sc:
        return conf / 100.0
    return (bisect.bisect_left(sc, conf) + bisect.bisect_right(sc, conf)) / 2.0 / len(sc)


def indep_vote(team, pid):
    """returns (winner, category) ; category in ok / none / fewer / tie"""
    votes = []
    for m in MEMBERS:
        ep = get(team, m)[pid]
        f = ep["final"][0]
        dx = f.get("dx")
        if dx is None or dx == "":
            continue
        votes.append((dx, norm_conf(ep["agents"][0]["model"], f.get("confidence"))))
    truth = get(team, "A")[pid]["truth"]
    if not votes:
        return None, "none"
    cnt = Counter(v[0] for v in votes)
    top = max(cnt.values())
    tied = [o for o, c in cnt.items() if c == top]
    if len(tied) > 1:
        best = {o: max(c for d, c in votes if d == o) for o in tied}
        mx = max(best.values())
        tied2 = [o for o in tied if best[o] == mx]
        if len(tied2) > 1:
            seed = int(hashlib.sha256(("20261120|%s|indep" % pid).encode()).hexdigest()[:8], 16)
            win = random.Random(seed).choice(sorted(tied2))
        else:
            win = tied2[0]
    else:
        win = tied[0]
    if win == truth:
        return win, "ok"
    if truth not in cnt:
        return win, "none"
    if cnt[truth] < top:
        return win, "fewer"
    return win, "tie"


def vote_vec(team):
    cats = [indep_vote(team, p)[1] for p in PIDS]
    return np.array([1.0 if c == "ok" else 0.0 for c in cats]), Counter(cats)


# ---------- shapley ----------
def shapley_pp(team):
    """per-patient Shapley matrix (n, 4); empty coalition value 1/49."""
    n = len(pids_for(team))
    v = {"": np.full(n, 1.0 / 49.0)}
    for c in COALS:
        v[c] = cvec(team, c)
    out = np.zeros((n, 4))
    for i, m in enumerate(MEMBERS):
        others = [x for x in MEMBERS if x != m]
        for r in range(0, 4):
            w = math.factorial(r) * math.factorial(3 - r) / math.factorial(4)
            for S in itertools.combinations(others, r):
                s = "".join(S)
                si = "".join(sorted(S + (m,)))
                out[:, i] += w * (v[si] - v[s])
    return out


# ---------- retention ----------
def retention(team):
    eps = get(team, "ABCD")
    maxes = []
    link_n = link_d = 0  # previous ASKED question (skipping null rounds)
    adj_n = adj_d = 0    # strictly the preceding round, both asked
    votes = Counter()
    nullr = 0
    nrounds = Counter()
    for p in PIDS:
        ep = eps[p]
        nrounds[len(ep["rounds"])] += 1
        per_agent = Counter()
        prev = None
        prev_adj = None
        for k, r in enumerate(ep["rounds"]):
            ch = r.get("chosen")
            if ch is None:
                nullr += 1
                prev_adj = None
                continue
            votes[r.get("vote")] += 1
            props = set(i for i, pr in enumerate(r["proposals"])
                        if pr.get("valid") and pr.get("code") == ch)
            for i in props:
                per_agent[i] += 1
            if k >= 1:
                if prev is not None:
                    link_d += 1
                    link_n += 1 if (props & prev) else 0
                if prev_adj is not None:
                    adj_d += 1
                    adj_n += 1 if (props & prev_adj) else 0
            prev = props
            prev_adj = props
        maxes.append(max(per_agent.values()) if per_agent else 0)
    return dict(mean_max=float(np.mean(maxes)), link=link_n / link_d, link_n=link_n, link_d=link_d,
                adj=adj_n / adj_d if adj_d else float("nan"), votes=dict(votes), null_rounds=nullr,
                asked=sum(votes.values()), rounds_hist=dict(nrounds))


def n_trunc(ep):
    return sum(1 for u in ep.get("usage", []) if u.get("finish_reason") == "length")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, p)


# ---------- report helpers ----------
ROWS = []


def row(claim, paper, mine, ok):
    ROWS.append((claim, paper, mine, "MATCH" if ok else "MISMATCH"))


def r3(x):
    return "%.3f" % (x + 1e-12)


def pp(x):
    s = "%+.1f" % (x * 100 + (1e-9 if x >= 0 else -1e-9))
    return "0.0" if s in ("+0.0", "-0.0") else s


def chk_acc(claim, paper, val):
    row(claim, paper, r3(val)[1:] if 0 <= val < 1 else r3(val), abs(float(r3(val)) - float(paper)) < 1e-9)


def chk_diff(claim, p_pt, p_lo, p_hi, d):
    cis = ci3(d)
    pt = d.mean()
    ok_pt = abs(float(pp(pt)) - p_pt) < 1e-9
    worst = max(max(abs(lo * 100 - p_lo), abs(hi * 100 - p_hi)) for lo, hi in cis)
    mine = "%s; CI by seed: %s" % (pp(pt), " / ".join("[%+.2f, %+.2f]" % (lo * 100, hi * 100) for lo, hi in cis))
    mine += " (max endpoint gap %.2f pp)" % worst
    ok_ci = worst <= 0.2
    if not ok_ci:
        # differences live on a grid of 1/n, so a percentile can jump one grid step between seeds
        step = 100.0 / len(d)
        lo_g = [abs(lo * 100 - p_lo) for lo, hi in cis]
        hi_g = [abs(hi * 100 - p_hi) for lo, hi in cis]
        some = min(lo_g) <= 0.2 and min(hi_g) <= 0.2
        near = max(lo_g + hi_g) <= step + 0.06
        rlo, rhi = ci_ref(d)
        mine += "; lower within 0.2 pp in %d of 3 seeds, upper in %d of 3; B=200,000 reference [%+.2f, %+.2f]" % (
            sum(g <= 0.2 for g in lo_g), sum(g <= 0.2 for g in hi_g), rlo * 100, rhi * 100)
        if ok_pt and some and near:
            ROWS.append((claim, "%+.1f [%+.1f, %+.1f]" % (p_pt, p_lo, p_hi), mine,
                         "MATCH (seed-sensitive, one grid step of %.2f pp)" % step))
            return cis
    row(claim, "%+.1f [%+.1f, %+.1f]" % (p_pt, p_lo, p_hi), mine, ok_pt and ok_ci)
    return cis


def main():
    T5 = ["P0", "P1", "P0Q", "P1Q", "P2"]
    paper_team = dict(zip(T5, [".408", ".412", ".388", ".340", ".405"]))
    paper_mem = {"P0": [".367", ".408", ".388", ".388"], "P1": [".395", ".344", ".350", ".367"],
                 "P0Q": [".327", ".354", ".344", ".313"], "P1Q": [".313", ".310", ".320", ".327"],
                 "P2": [".367", ".344", ".327", ".357"]}
    paper_mean = dict(zip(T5, [".388", ".364", ".334", ".317", ".349"]))
    paper_tm = {"P0": (2.0, -0.5, 4.7), "P1": (4.8, 1.5, 8.0), "P0Q": (5.4, 2.0, 8.8),
                "P1Q": (2.3, -1.0, 5.8), "P2": (5.6, 2.1, 9.1)}
    paper_vote = dict(zip(T5, [".408", ".391", ".374", ".364", ".401"]))
    paper_tv = {"P0": (0.0, -2.7, 2.7), "P1": (2.0, -1.4, 5.4), "P0Q": (1.4, -3.1, 5.8),
                "P1Q": (-2.4, -6.5, 1.7), "P2": (0.3, -4.1, 4.8)}
    vm = {}
    cats = {}
    p0q_cis = None
    for t in T5:
        team = cvec(t, "ABCD")
        solo = [cvec(t, m) for m in MEMBERS]
        mm = np.mean(solo, axis=0)
        vote, cat = vote_vec(t)
        cats[t] = cat
        chk_acc("1 %s team accuracy" % t, paper_team[t], team.mean())
        for m, s, pv in zip(MEMBERS, solo, paper_mem[t]):
            chk_acc("1 %s member %s solo" % (t, m), pv, s.mean())
        chk_acc("1 %s mean member" % t, paper_mean[t], mm.mean())
        c = chk_diff("1 %s team - mean member" % t, *paper_tm[t], team - mm)
        if t == "P0Q":
            p0q_cis = c
        chk_acc("1 %s independent vote" % t, paper_vote[t], vote.mean())
        chk_diff("1 %s team - vote" % t, *paper_tv[t], team - vote)
        vm[t] = (vote.mean() - mm.mean()) * 100
    lo, hi = min(vm.values()), max(vm.values())
    row("1 vote - mean member, range over 5 teams", "2.0 to 5.3",
        "%.1f to %.1f (%s)" % (lo, hi, ", ".join("%s %.2f" % (t, vm[t]) for t in T5)),
        "%.1f" % lo == "2.0" and "%.1f" % hi == "5.3")

    # 2 Shapley
    sh = {t: shapley_pp(t) for t in ["P0", "P1", "P2", "CHECK"]}
    for t, plo, phi in [("P0", 0.083, 0.107), ("P2", 0.084, 0.105)]:
        v = sh[t].mean(axis=0)
        row("2 %s Shapley min..max" % t, "%.3f..%.3f" % (plo, phi),
            "%.3f..%.3f (%s)" % (v.min(), v.max(), ", ".join("%s %.4f" % (m, x) for m, x in zip(MEMBERS, v))),
            "%.3f" % v.min() == "%.3f" % plo and "%.3f" % v.max() == "%.3f" % phi)
    v = sh["P0"].mean(axis=0)
    row("2 P0 Shapley spread", "about 0.024", "%.4f" % (v.max() - v.min()), "%.3f" % (v.max() - v.min()) == "0.024")
    v = sh["P1"].mean(axis=0)
    for i, (ppt, plo, phi) in [(0, (0.125, 0.098, 0.151)), (1, (0.079, 0.057, 0.101))]:
        cis = ci3(sh["P1"][:, i])
        worst = max(max(abs(a - plo), abs(b - phi)) for a, b in cis)
        row("2 P1 Shapley member %s" % MEMBERS[i], "%.3f [%.3f, %.3f]" % (ppt, plo, phi),
            "%.3f; CI by seed: %s (max endpoint gap %.4f)" % (
                v[i], " / ".join("[%.4f, %.4f]" % c for c in cis), worst),
            "%.3f" % v[i] == "%.3f" % ppt and worst <= 0.002)
    row("2 P1 Shapley spread", "0.046", "%.4f (%s)" % (v.max() - v.min(), ", ".join("%s %.4f" % (m, x) for m, x in zip(MEMBERS, v))),
        "%.3f" % (v.max() - v.min()) == "0.046")
    v = sh["CHECK"].mean(axis=0)
    row("2 CHECK Shapley member D", "0.386", "%.3f" % v[3], "%.3f" % v[3] == "0.386")
    row("2 CHECK Shapley member C", "-0.032", "%.3f" % v[2], "%.3f" % v[2] == "-0.032")

    # 3 coalition size
    sz = {}
    for t in ["P0", "P1", "P1Q", "P2"]:
        three = np.mean([cvec(t, c).mean() for c in COALS if len(c) == 3])
        sz[t] = (cvec(t, "ABCD").mean() - three) * 100
    lo, hi = min(sz.values()), max(sz.values())
    row("3 full team - mean 3-agent coalition, range", "-4.3 to +1.4",
        "%+.1f to %+.1f (%s)" % (lo, hi, ", ".join("%s %+.2f" % (t, sz[t]) for t in sz)),
        "%+.1f" % lo == "-4.3" and "%+.1f" % hi == "+1.4")

    # 4 retention
    ret = {t: retention(t) for t in ["P0", "P2"]}
    for t, pv in [("P0", "4.97"), ("P2", "3.82")]:
        row("4 %s mean max questions by one agent" % t, pv, "%.2f" % ret[t]["mean_max"], "%.2f" % ret[t]["mean_max"] == pv)
    for t, pv in [("P0", "100"), ("P2", "79")]:
        r = ret[t]
        row("4 %s share linked to previous asked question" % t, pv + "%",
            "%.2f%% (%d/%d); strictly-adjacent-round variant %.2f%%" % (r["link"] * 100, r["link_n"], r["link_d"], r["adj"] * 100),
            "%.0f" % (r["link"] * 100) == pv)
    r = ret["P2"]
    nconf = r["votes"].get("confidence", 0)
    tot_all = r["asked"] + r["null_rounds"]
    row("4 P2 question rounds resolved by confidence", "32%",
        "%.2f%% (%d/%d asked rounds; %.2f%% of all %d rounds); votes %s" % (
            100.0 * nconf / r["asked"], nconf, r["asked"], 100.0 * nconf / tot_all, tot_all, r["votes"]),
        "%.0f" % (100.0 * nconf / r["asked"]) == "32")

    # 5 decomposition
    for t, pv in [("P0", (174, 156, 16, 2)), ("P2", (176, 127, 36, 13))]:
        c = cats[t]
        mine = (c["none"] + c["fewer"] + c["tie"], c["none"], c["fewer"], c["tie"])
        row("5 %s vote wrong = none / fewer / lost tie" % t, "%d = %d / %d / %d" % pv, "%d = %d / %d / %d" % mine, mine == pv)

    # 6 anchor
    anc = get("ANCHOR", "A")
    a = np.array([1.0 if anc[p]["correct"] else 0.0 for p in PIDS])
    q = cvec("P0Q", "ABCD")
    chk_diff("6 ANCHOR - P0Q team", 7.5, 2.0, 12.9, a - q)
    b = int(((a == 1) & (q == 0)).sum())
    c = int(((a == 0) & (q == 1)).sum())
    pval = mcnemar_exact(b, c)
    row("6 exact McNemar p", ".010", "%.4f (discordant %d vs %d; ANCHOR acc %.3f)" % (pval, b, c, a.mean()), "%.3f" % pval == "0.010")

    # 7 truncation
    e1, e0 = get("P1Q", "ABCD"), get("P0Q", "ABCD")
    keep = [i for i, p in enumerate(PIDS) if n_trunc(e1[p]) == 0 and n_trunc(e0[p]) == 0]
    d = (cvec("P1Q", "ABCD") - q)[keep]
    row("7 n patients with no truncated call (either team, ABCD)", "216", str(len(keep)), len(keep) == 216)
    chk_diff("7 P1Q - P0Q team on that subset", -4.2, -9.7, 1.4, d)
    for t, pv in [("P1Q", 102), ("P0Q", 123)]:
        per = {c: sum(n_trunc(get(t, c)[p]) for p in PIDS) for c in ["A", "B", "C", "D", "ABCD"]}
        row("7 %s truncated calls (A,B,C,D,ABCD)" % t, str(pv), "%d (%s)" % (sum(per.values()), per), sum(per.values()) == pv)

    # ---------- data checks ----------
    for t in T5 + ["CHECK"]:
        cl = ["A", "B", "C", "D", "ABCD"] if t == "P0Q" else COALS
        pl = set(pids_for(t))
        for c in cl:
            eps = get(t, c)
            if set(eps) != pl:
                NOTES.append("%s/%s: pid set differs from reference (%d extra, %d missing)" % (
                    t, c, len(set(eps) - pl), len(pl - set(eps))))
            bad = [p for p in pl if p in eps and len(eps[p]["rounds"]) != 5]
            if bad:
                NOTES.append("%s/%s: %d episodes with a round count other than 5" % (t, c, len(bad)))
            if len(c) == 1:
                nul = sum(1 for p in pl if not eps[p]["final"][0].get("dx"))
                if nul:
                    NOTES.append("%s/%s: %d solo episodes with empty final dx" % (t, c, nul))
                mism = sum(1 for p in pl if eps[p]["final"][0].get("dx")
                           and (eps[p]["final"][0]["dx"] == eps[p]["truth"]) != bool(eps[p]["correct"]))
                if mism:
                    NOTES.append("%s/%s: %d solo episodes where (final[0].dx == truth) differs from `correct`" % (t, c, mism))
                models = set(eps[p]["agents"][0]["model"] for p in pl)
                for m in models:
                    if not SCALES.get(m, {}).get("final"):
                        NOTES.append("%s/%s: model %s has no final confidence scale (conf/100 used)" % (t, c, m))
        truths = [tuple(get(t, c)[p]["truth"] for c in cl) for p in pl]
        if any(len(set(x)) != 1 for x in truths):
            NOTES.append("%s: truth differs across coalition files for some patient" % t)
    if set(anc) != set(PIDS):
        NOTES.append("ANCHOR/A: pid set differs from reference")
    NOTES.append("Claim 4, P2 linked share: 923 of 1176 = 78.49%, which rounds to 78%, not 79%. One more linked question (924) would give 78.57%. Counting proposers with or without the `valid` flag gives the same count, since invalid proposals have a null code.")
    NOTES.append("Claim 4, P0 linked share: 1173 of 1176 = 99.74%, so 100% only after rounding, as the paper says.")
    NOTES.append("Claim 4, confidence share: denominator is all 1470 P2 question rounds (none has a null `chosen`).")
    NOTES.append("Seed-sensitive CI rows (team - vote for P1, P1-Q, P2): the bootstrap mass at or below the grid point next to the endpoint is 0.975 to within about 0.001 (or 0.025 for lower endpoints), so either neighbouring grid value can come out. For P1-Q the B = 200,000 reference gives an upper endpoint of +1.36, one grid step below the paper's +1.7; two of three B = 10,000 seeds give +1.70.")
    for t in ["P0", "P2"]:
        r = ret[t]
        NOTES.append("%s/ABCD: %d rounds with null `chosen`; asked-round vote types %s" % (t, r["null_rounds"], r["votes"]))
    for t in T5:
        NOTES.append("%s independent vote categories: %s" % (t, dict(cats[t])))

    nm = sum(1 for r in ROWS if r[3] == "MISMATCH")
    nss = sum(1 for r in ROWS if r[3].startswith("MATCH (seed"))
    out = ["# Independent verification of v10 numbers", "",
           "Computed by `v_v10.py` from the raw episode logs and the confidence scales only.",
           "Bootstrap: percentile 95%% intervals over patients, B = %d, numpy default_rng seeds %s." % (B, ", ".join(map(str, SEEDS))),
           "Point estimates and counts must agree exactly after rounding. CI endpoints count as MATCH when every seed is within 0.2 pp (0.002 for Shapley) of the paper value.",
           "Some CI rows are marked MATCH (seed-sensitive): each paper endpoint is reproduced within 0.2 pp by at least one seed, and no seed is further away than one grid step. The paired differences take values on a grid of 1/n (0.34 pp for n = 294), and for these rows the 2.5% or 97.5% bootstrap quantile sits almost exactly on a grid boundary, so the endpoint moves by one step between seeds. A B = 200,000 reference run is listed for those rows.",
           "", "Rows: %d, MATCH: %d (of which seed-sensitive: %d), MISMATCH: %d" % (len(ROWS), len(ROWS) - nm, nss, nm), "",
           "| Claim | Paper | Recomputed | Result |", "|---|---|---|---|"]
    for r in ROWS:
        out.append("| %s | %s | %s | %s |" % tuple(str(x).replace("|", "/") for x in r))
    out += ["", "## P0-Q team minus mean member, CI by seed", ""]
    for s, (lo, hi) in zip(SEEDS, p0q_cis):
        out.append("- seed %d: [%+.2f, %+.2f] pp" % (s, lo * 100, hi * 100))
    out += ["", "## Notes from the logs", ""]
    out += ["- " + n for n in NOTES]
    out += [""] + POST_NOTE + [""]
    txt = "\n".join(out)
    with open(os.path.join(HERE, "V_V10.md"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
