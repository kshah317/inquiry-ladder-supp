"""Figures for draft v10 (labels only). Figures for draft v8 (labels and Fig 1 x-range only, same data as 63_figures_v2.py). Figures for draft v2: Fig 1 adds P0-Q and the anchor; Fig 2 uses the reader-free evidence value and the
decomposition against the mean Gemma seed; Fig 3 adds same-model Qwen pairs. Reads analysis.json, review.json, p0q.json,
anchor.json and the P0-Q logs. No model calls. Output: ../paper/figures/fig{1,2,3}_v2.{pdf,png}.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx import stats as S

OUT = ROOT.parent / "paper" / "figures"
A = ROOT / "results" / "analysis"
D = json.loads((A / "analysis.json").read_text())
RV = json.loads((A / "review.json").read_text())
Q = json.loads((A / "p0q.json").read_text())
AN = json.loads((A / "anchor.json").read_text())
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#6f6e69", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "lines.linewidth": 2})
LAB = {"P0": "P0 same model", "P1": "P1 personas", "P2": "P2 models", "P0Q": "P0-Q same model (Qwen)", "P1Q": "P1-Q personas (Qwen)"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


pids = sorted({json.loads(l)["pid"] for l in open(ROOT / "data/samples/main.csv").read().splitlines()[1:] for _ in [0]} if False else [])
team_q = {}
for line in open(ROOT / "results/main/P0Q/ABCD.jsonl", encoding="utf-8"):
    r = json.loads(line); team_q[r["pid"]] = float(r["correct"])
yq = np.array([team_q[p] for p in sorted(team_q)])
ciq = S.ci(S.boot_means(yq, S.boot_idx(len(yq))))
panels = {p: {"solo": D["panels"][p]["solo"], "vote": D["panels"][p]["indep_vote"], "team": D["panels"][p]["team"]["mean"],
              "ci": D["panels"][p]["team"]["ci95"]} for p in ["P0", "P1", "P2", "P1Q"]}
panels["P0Q"] = {"solo": Q["accuracy"]["solo"], "vote": Q["team_minus_indep_vote"]["indep_vote"], "team": Q["accuracy"]["team"], "ci": list(ciq)}

# ---------------------------------------------------------------- Fig 1
order = ["P0", "P1", "P2", "P0Q", "P1Q"]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for y, p in enumerate(order):
    P = panels[p]
    xs = [round(100 * s, 1) for s in P["solo"]]
    seen = {}
    for x in xs:
        k = seen.get(x, 0); seen[x] = k + 1
        ax.scatter([x], [y + 0.07 * k], s=22, facecolors="white", edgecolors=MUTED, linewidths=1, zorder=3,
                   label="members alone" if (y == 0 and x == xs[0]) else None)
    ax.scatter([100 * P["vote"]], [y + 0.22], marker="D", s=26, color=ORANGE, edgecolors="white", linewidths=0.8,
               zorder=4, label="independent vote" if y == 0 else None)
    lo, hi = (100 * v for v in P["ci"])
    ax.plot([lo, hi], [y - 0.2] * 2, color=BLUE, lw=1.4, solid_capstyle="round")
    ax.scatter([100 * P["team"]], [y - 0.2], s=30, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4,
               label="team (95% CI)" if y == 0 else None)
acc_anchor = 100 * AN["acc"]["mean"] if isinstance(AN["acc"], dict) else 100 * AN["acc"]
ax.axvline(acc_anchor, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
ax.text(acc_anchor + 0.4, 4.4, "single-agent\nreference", ha="left", va="bottom", fontsize=6.5, color=INK)
ax.set_yticks(range(len(order)), [LAB[p] for p in order])
ax.invert_yaxis()
ax.set_xlabel("Top-1 accuracy (%)")
ax.set_xlim(28, 50)
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=3, fontsize=7, handletextpad=0.3, columnspacing=0.9)
save(fig, "fig1_v4")

# ---------------------------------------------------------------- Fig 2
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1.1, 1]})
for y, p in enumerate(order):
    nbp = Q["nb"]["top1"] if p == "P0Q" else RV["nb"][p]["top1"]
    uni = Q["nb"]["union"] if p == "P0Q" else nbp["union"]
    mem = 100 * np.mean([nbp[k] for k in "ABCD"])
    joint, un = 100 * nbp["ABCD"], 100 * uni
    a.plot([min(mem, joint), un], [y, y], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    a.scatter([mem], [y], s=26, facecolors="white", edgecolors=MUTED, linewidths=1.2, zorder=3, label="one member alone (mean)" if y == 0 else None)
    a.scatter([joint], [y], s=30, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4, label="team, 5 shared questions" if y == 0 else None)
    a.scatter([un], [y], marker="D", s=26, color=ORANGE, edgecolors="white", linewidths=0.8, zorder=4, label="4 members pooled (20 questions)" if y == 0 else None)
a.set_yticks(range(len(order)), [LAB[p] for p in order])
a.invert_yaxis()
a.set_xlabel("Information ceiling (top-1 %)")
a.xaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
a.set_title("(a)", loc="left", fontsize=8, pad=24)
rows = ["P0", "P1", "P2"]
parts = [("asking_vs_mean_seed", "acquisition (language-model reader)", BLUE), ("after_split_reading_first_vote", "inference + first vote", MUTED),
         ("after_split_deliberation", "deliberation", ORANGE)]
for y, p in enumerate(rows):
    for j, (k, lab, col) in enumerate(parts):
        x = RV["decomp"][p][k]; yy = y + (j - 1) * 0.24
        b.plot(x["ci95"], [yy, yy], color=col, lw=1.2, alpha=0.8)
        b.scatter([x["diff"]], [yy], s=22, color=col, edgecolors="white", linewidths=0.6, zorder=4, label=lab if y == 0 else None)
b.axvline(0, color=MUTED, lw=0.8)
b.set_yticks(range(3), [LAB[p] for p in rows])
b.invert_yaxis()
b.set_xlabel("Points vs mean Gemma generalist")
b.xaxis.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True)
b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
b.set_title("(b)", loc="left", fontsize=8, pad=24)
fig.tight_layout(w_pad=2.0)
save(fig, "fig2_v4")

# ---------------------------------------------------------------- Fig 3
rows = [("P0", "same model", "Same model, Gemma (P0)"), ("P0Q", "same model", "Same model, Qwen (P0-Q)"),
        ("P1", "persona", "Personas, Gemma (P1)"), ("P1Q", "persona", "Personas, Qwen (P1-Q)"),
        ("P2", "same family", "Same family (P2)"), ("P2", "cross family", "Cross family (P2)")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for y, (p, t, lab) in enumerate(rows):
    x = (Q["error_overlap"] if p == "P0Q" else D["error_overlap"][p])["by_type"][t]["same_given_both"]
    lo, hi = (100 * v for v in x["null_95"])
    ax.plot([lo, hi], [y, y], color=MUTED, lw=5, alpha=0.35, solid_capstyle="butt", label="chance (95% band)" if y == 0 else None)
    ax.plot([100 * x["null_mean"], 100 * x["obs"]], [y, y], color=GRID, lw=1, zorder=1)
    ax.scatter([100 * x["obs"]], [y], s=30, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4, label="observed" if y == 0 else None)
    ax.text(100 * x["obs"] + 2, y, f"{100 * x['obs']:.0f}%", va="center", fontsize=7, color=INK)
ax.set_yticks(range(len(rows)), [r[2] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Same wrong answer, when both agents are wrong (%)")
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7)
save(fig, "fig3_v4")
print("ok", acc_anchor)
