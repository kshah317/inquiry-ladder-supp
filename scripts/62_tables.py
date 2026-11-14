"""Paper tables (no model calls): cost vs accuracy (incl. the anchor) and deliberation with the no-peer control.
Inputs: results/analysis/analysis.json, anchor.json, nopeer.json, results/main/ANCHOR/A.jsonl
Output: paper/tables/table_cost.{md,tex}, paper/tables/table_delib.{md,tex}"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import metrics as M  # noqa: E402
from ddx import stats as S  # noqa: E402

AN = ROOT / "results" / "analysis"
OUT = ROOT.parent / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((AN / "analysis.json").read_text())
A = json.loads((AN / "anchor.json").read_text())
N = json.loads((AN / "nopeer.json").read_text())
NAME = {"P0": "P0", "P1": "P1", "P1Q": "P1-Q", "P2": "P2"}


def write(stem, md, tex):
    (OUT / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md), "\n")


# ------------------------------------------------------------------ cost table
def cost_table():
    idx = S.boot_idx(294)
    rows = []   # (label, acc, ci, usd, calls, tok_in, tok_out)
    C = D["cost"]
    # solo Gemma generalist = mean of the four P0 seeds; independent vote = four solo games
    P0 = M.load_panel("P0")
    solo = np.stack([P0.correct((i,)) for i in M.FULL]).mean(0)
    sg = lambda P, k: [g for g in P.g(k)]  # noqa: E731
    def usage(P, keys):
        gs = [sg(P, k) for k in keys]
        calls = np.mean([sum(g[j]["calls"] for g in gs) for j in range(P.n)])
        ti = np.mean([sum(g[j]["tokens_in"] for g in gs) for j in range(P.n)])
        to = np.mean([sum(g[j]["tokens_out"] for g in gs) for j in range(P.n)])
        usd = np.mean([sum(g[j]["cost"] for g in gs) for j in range(P.n)])
        return usd, calls, ti, to
    u = tuple(np.mean([usage(P0, [k])[t] for k in "ABCD"]) for t in range(4))
    rows.append(("Solo Gemma 4 31B (mean of 4 seeds)", solo.mean(), S.ci(S.boot_means(solo, idx)), *u))
    for k in ["P0", "P2"]:
        P = M.load_panel(k)
        iv, _ = M.indep_vote(P, M.FULL)
        rows.append((f"{NAME[k]} independent vote (4 solo games)", iv.mean(), S.ci(S.boot_means(iv, idx)), *usage(P, list("ABCD"))))
    for k in ["P0", "P1", "P2", "P1Q"]:
        P = M.load_panel(k)
        t = P.correct(M.FULL)
        rows.append((f"{NAME[k]} team (4 agents, deliberation)", t.mean(), tuple(D["panels"][k]["team"]["ci95"]), *usage(P, ["ABCD"])))
    an = [json.loads(x) for x in (ROOT / "results/main/ANCHOR/A.jsonl").open(encoding="utf-8")]
    us = [[x for x in g["usage"]] for g in an]
    rows.append(("Anchor: Qwen3.8-Max alone (reasoning on)", A["acc"]["mean"], tuple(A["acc"]["ci95"]),
                 A["cost_per_consultation_usd"], np.mean([len(x) for x in us]),
                 np.mean([sum(u.get("prompt_tokens") or 0 for u in x) for x in us]),
                 np.mean([sum(u.get("completion_tokens") or 0 for u in x) for x in us])))
    base = rows[0][3]
    md = ["Table: Accuracy and cost per consultation (294 test patients, main run). Cost = provider price per call via OpenRouter, "
          "as if no call were cached. Attribution campaign (all 15 sub-teams of one panel): "
          + ", ".join(f"{NAME[k]} ${C[k]['campaign_per_patient_usd']:.2f}" for k in ["P0", "P1", "P2", "P1Q"]) + " per patient.", "",
          "| System | Top-1 [95% CI] | $ per consultation | × solo | Calls | Tokens in / out |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lcrrrr}", r"\toprule",
           r"System & Top-1 [95\% CI] & \$/consult. & $\times$ solo & Calls & Tokens in/out \\", r"\midrule"]
    for lab, acc, ci, usd, calls, ti, to in rows:
        md.append(f"| {lab} | {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {usd:.4f} | {usd/base:.1f} | {calls:.1f} | {ti/1000:.0f}k / {to/1000:.1f}k |")
        tex.append(f"{lab} & {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] & {usd:.4f} & {usd/base:.1f} & {calls:.1f} & {ti/1000:.0f}k / {to/1000:.1f}k \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Accuracy and cost per consultation (294 patients, main run). Cost is the provider price per call via OpenRouter, counted as if no call were cached. The anchor's accuracy is a lower bound: 20\% of its calls hit the 8{,}000-token limit.}",
            r"\label{tab:cost}", r"\end{table}"]
    write("table_cost", md, tex)


# ------------------------------------------------------------------ deliberation table
def delib_table():
    f = lambda c: f"{c['diff_pts']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"  # noqa: E731
    md = ["Table: Deliberation vs a no-peer second look (full panels, 294 patients). Real − pre = deliberation gain; "
          "no-peer − pre = gain from a second look without peers; real − no-peer = peer effect. Pts [95% bootstrap CI].", "",
          "| Panel | Pre | Real | No-peer | Real − pre | No-peer − pre | Peer effect | Switch rate real / no-peer | Unanimous after real / no-peer (pre) |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lccccccc}", r"\toprule",
           r"Panel & Pre & Real & No-peer & Real$-$pre & No-peer$-$pre & Peer effect & Switches real/no-peer \\", r"\midrule"]
    for k in ["P0", "P1", "P1Q", "P2"]:
        v = N[k]
        sw = f"{100*v['answers_real']['switch_rate']:.0f}% / {100*v['answers_nopeer']['switch_rate']:.0f}%"
        un = f"{100*v['unanimous_after']['real']:.0f}% / {100*v['unanimous_after']['nopeer']:.0f}% ({100*v['unanimous_after']['pre']:.0f}%)"
        cells = [NAME[k], f"{v['acc']['pre']:.3f}", f"{v['acc']['real']:.3f}", f"{v['acc']['nopeer']:.3f}",
                 f(v["delib_effect_real_minus_pre"]), f(v["second_look_np_minus_pre"]), f(v["peer_effect_real_minus_np"])]
        md.append("| " + " | ".join(cells + [sw, un]) + " |")
        tex.append(" & ".join(cells + [sw.replace("%", r"\%")]) + r" \\")
    p = N["pooled_mean_of_4"]
    md.append(f"| Mean of 4 | | | | {f(p['delib'])} | {f(p['second_look'])} | {f(p['peer'])} | | |")
    tex += [r"\midrule", f"Mean of 4 & & & & {f(p['delib'])} & {f(p['second_look'])} & {f(p['peer'])} & \\\\",
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Deliberation vs a no-peer second look (full panels, 294 patients). The no-peer call has the same context and seed as the real deliberation call; only the peers' answers are withheld. Differences in points with 95\% bootstrap CIs.}",
            r"\label{tab:delib}", r"\end{table*}"]
    write("table_delib", md, tex)


if __name__ == "__main__":
    cost_table()
    delib_table()
