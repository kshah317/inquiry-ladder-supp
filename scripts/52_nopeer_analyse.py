"""Analysis of the no-peer control. No API calls.
Output: results/analysis/nopeer.json and results/analysis/NOPEER_REPORT.md
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import stats as S  # noqa: E402
from ddx.game import norm_conf  # noqa: E402

PANELS = ["P0", "P1", "P1Q", "P2"]
LABEL = {"P0": "P0", "P1": "P1", "P1Q": "P1-Q", "P2": "P2"}


def coin_free(answers, models, truth):
    """Expected correctness of the game's vote with the final coin replaced by its expectation."""
    ch = [(dx, norm_conf(m, c, "final")) for (dx, c), m in zip(answers, models) if dx]
    if not ch:
        return 0.0
    from collections import Counter
    cnt = Counter(d for d, _ in ch)
    top = max(cnt.values())
    tied = [d for d, k in cnt.items() if k == top]
    best = {d: max(c for o, c in ch if o == d) for d in tied}
    m = max(best.values())
    tied = [d for d in tied if best[d] == m]
    return float(truth in tied) / len(tied)


def load(name):
    R = [json.loads(x) for x in (ROOT / "results" / "nopeer" / name / "ABCD.jsonl").open(encoding="utf-8")]
    return sorted(R, key=lambda r: r["pid"])


def per_answer(R, key):
    n = sw = w2r = r2w = w2w = peer = 0
    for r in R:
        for i in range(4):
            pre, post = r["final"][i][0], r[key][i][0]
            peers = {r["final"][j][0] for j in range(4) if j != i}
            n += 1
            ok0, ok1 = pre == r["truth"], post == r["truth"]
            if pre != post:
                sw += 1
                peer += post in peers
                w2w += (not ok0) and (not ok1)
            w2r += (not ok0) and ok1
            r2w += ok0 and (not ok1)
    return {"n_answers": n, "switches": sw, "switch_rate": sw / n, "wrong_to_right": w2r,
            "right_to_wrong": r2w, "wrong_to_other_wrong": w2w, "to_peer_share": peer / max(sw, 1)}


out, diffs = {}, {"peer": [], "second_look": [], "delib": [], "peer_cf": []}
pids0 = None
for name in PANELS:
    R = load(name)
    pids = [r["pid"] for r in R]
    pids0 = pids0 or pids
    assert pids == pids0 and len(R) == 294
    idx = S.boot_idx(len(R))
    pre = np.array([r["correct_pre"] for r in R], float)
    real = np.array([r["correct_real"] for r in R], float)
    np_ = np.array([r["correct_np"] for r in R], float)
    cf = {k: np.array([coin_free(r[src], r["models"], r["truth"]) for r in R])
          for k, src in (("pre", "final"), ("real", "revised"), ("np", "nopeer"))}

    def con(a, b):
        d = a - b
        lo, hi = S.ci(S.boot_means(d, idx))
        return {"diff_pts": 100 * d.mean(), "ci95": [100 * lo, 100 * hi]}
    out[name] = {
        "acc": {"pre": pre.mean(), "real": real.mean(), "nopeer": np_.mean()},
        "acc_coin_free": {k: float(v.mean()) for k, v in cf.items()},
        "delib_effect_real_minus_pre": con(real, pre),
        "second_look_np_minus_pre": con(np_, pre),
        "peer_effect_real_minus_np": con(real, np_) | {"mcnemar": S.mcnemar(real, np_)},
        "peer_effect_coin_free": con(cf["real"], cf["np"]),
        "answers_real": per_answer(R, "revised"), "answers_nopeer": per_answer(R, "nopeer"),
        "unanimous_after": {"real": float(np.mean([len({a[0] for a in r["revised"]}) == 1 for r in R])),
                            "nopeer": float(np.mean([len({a[0] for a in r["nopeer"]}) == 1 for r in R])),
                            "pre": float(np.mean([len({a[0] for a in r["final"]}) == 1 for r in R]))},
        "member_acc": {k: float(np.mean([a[0] == r["truth"] for r in R for a in r[src]]))
                       for k, src in (("pre", "final"), ("real", "revised"), ("nopeer", "nopeer"))},
        "kept_after_invalid": sum(len(r["kept_after_invalid"]) for r in R),
        "retries": sum(r["retries"] for r in R),
        "truncated": sum(u["finish_reason"] == "length" for r in R for u in r["usage"]),
        "cost_usd": sum((u["cost"] or 0) for r in R for u in r["usage"]),
    }
    diffs["peer"].append(real - np_)
    diffs["second_look"].append(np_ - pre)
    diffs["delib"].append(real - pre)
    diffs["peer_cf"].append(cf["real"] - cf["np"])

idx = S.boot_idx(294)
pooled = {}
for k, v in diffs.items():
    d = np.mean(v, axis=0)
    lo, hi = S.ci(S.boot_means(d, idx))
    pooled[k] = {"diff_pts": 100 * d.mean(), "ci95": [100 * lo, 100 * hi]}
out["pooled_mean_of_4"] = pooled
rule = pooled["peer"]["ci95"][0] > 0
out["reading_rule"] = ("peer influence (pooled CI excludes 0)" if rule else
                       "pooled peer-effect CI includes 0: not attributable to peers")
(ROOT / "results" / "analysis" / "nopeer.json").write_text(json.dumps(out, indent=1, default=float))

f = lambda c: f"{c['diff_pts']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"  # noqa: E731
L = ["# No-peer reconsideration control", "",
     "Full panels (ABCD), 294 patients each. Team top-1 accuracy; differences in pts [95% bootstrap CI].", "",
     "| Panel | Pre | Real delib | No-peer | Real − pre | No-peer − pre (second look) | Real − no-peer (peer effect) | McNemar real vs np | Coin-free peer effect |",
     "|---|---|---|---|---|---|---|---|---|"]
for n in PANELS:
    o = out[n]
    m = o["peer_effect_real_minus_np"]["mcnemar"]
    L.append(f"| {LABEL[n]} | {o['acc']['pre']:.3f} | {o['acc']['real']:.3f} | {o['acc']['nopeer']:.3f} | "
             f"{f(o['delib_effect_real_minus_pre'])} | {f(o['second_look_np_minus_pre'])} | {f(o['peer_effect_real_minus_np'])} | "
             f"{m['a_only']} vs {m['b_only']}, p={m.get('p', float('nan')):.3f} | {f(o['peer_effect_coin_free'])} |")
p = pooled
L += ["", f"**Pooled (mean of 4 panels):** deliberation {f(p['delib'])} = second look {f(p['second_look'])} + peer effect {f(p['peer'])}; "
      f"coin-free peer effect {f(p['peer_cf'])}.", f"Reading rule: {out['reading_rule']}.", "",
      "Per agent answer (1,176 answers per panel):", "",
      "| Panel | Condition | Switch rate | Wrong→right | Right→wrong | Wrong→other wrong | Switches to a peer's answer | Unanimous after |",
      "|---|---|---|---|---|---|---|---|"]
for n in PANELS:
    o = out[n]
    for lab, k, u in (("real", "answers_real", "real"), ("no-peer", "answers_nopeer", "nopeer")):
        a = o[k]
        L.append(f"| {LABEL[n]} | {lab} | {100*a['switch_rate']:.1f}% | {a['wrong_to_right']} | {a['right_to_wrong']} | "
                 f"{a['wrong_to_other_wrong']} | {100*a['to_peer_share']:.0f}% | {100*o['unanimous_after'][u]:.0f}% (pre {100*o['unanimous_after']['pre']:.0f}%) |")
L += ["", "Mean member accuracy (pre / real / no-peer): " + "; ".join(
    f"{LABEL[n]} {out[n]['member_acc']['pre']:.3f} / {out[n]['member_acc']['real']:.3f} / {out[n]['member_acc']['nopeer']:.3f}" for n in PANELS)]
L += ["", "Run health: " + "; ".join(f"{LABEL[n]} retries {out[n]['retries']}, kept-after-invalid {out[n]['kept_after_invalid']}, "
                                      f"truncated {out[n]['truncated']}, ${out[n]['cost_usd']:.2f}" for n in PANELS)]
(ROOT / "results" / "analysis" / "NOPEER_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
