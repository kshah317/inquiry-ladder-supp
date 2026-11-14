"""Paper figures from results/analysis/analysis.json and nopeer.json (no model calls).
Output: ../paper/figures/fig{1,2,3}.{pdf,png}. Colors: validated two-slot palette (blue, orange) + grays.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT / "results/analysis/analysis.json").read_text())
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#6f6e69", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "lines.linewidth": 2})
LAB = {"P0": "P0 same model", "P1": "P1 roles", "P2": "P2 models", "P1Q": "P1-Q roles (Qwen)"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1: team vs members vs independent vote
order = ["P0", "P1", "P2", "P1Q"]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for y, p in enumerate(order):
    P = D["panels"][p]
    ax.scatter([100 * s for s in P["solo"]], [y] * 4, s=22, facecolors="white", edgecolors=MUTED, linewidths=1, zorder=3,
               label="solo members" if y == 0 else None)
    ax.scatter([100 * P["indep_vote"]], [y + 0.18], marker="D", s=26, color=ORANGE, edgecolors="white", linewidths=0.8,
               zorder=4, label="vote of solo games" if y == 0 else None)
    lo, hi = (100 * v for v in P["team"]["ci95"])
    ax.plot([lo, hi], [y - 0.18] * 2, color=BLUE, lw=1.4, solid_capstyle="round")
    ax.scatter([100 * P["team"]["mean"]], [y - 0.18], s=30, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4,
               label="team (95% CI)" if y == 0 else None)
ax.set_yticks(range(len(order)), [LAB[p] for p in order])
ax.invert_yaxis()
ax.set_xlabel("Top-1 accuracy (%)")
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=7, handletextpad=0.3, columnspacing=0.9)
save(fig, "fig1_team_vs_vote")

# ---------------------------------------------------------------- Fig 2: asking vs reading (a) + answer vs evidence credit (b)
R = D["replay"]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
rows = ["P0", "P1", "P2"]
for y, p in enumerate(rows):
    dec = R[p]["decomposition_vs_gemma_generalist_solo"]
    ask, rd, tot = (100 * dec[k]["diff"] for k in ("asking", "reading_deliberation_vote", "total"))
    a.barh(y, ask, height=0.5, color=BLUE, edgecolor="white", linewidth=1, label="asking" if y == 0 else None)
    a.barh(y, rd, left=ask, height=0.5, color=ORANGE, edgecolor="white", linewidth=1,
           label="reading + deliberation + vote" if y == 0 else None)
    lo, hi = (100 * v for v in dec["total"]["ci95"])
    a.text(tot + 0.15, y, f"{tot:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=7, color=INK)
a.axvline(0, color=MUTED, lw=0.8)
a.set_xlim(0, 7.5)
a.set_yticks(range(3), [LAB[p] for p in rows])
a.invert_yaxis()
a.set_xlabel("Team − Gemma generalist solo (pts)")
a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=7)
a.set_title("(a)", loc="left", fontsize=8, pad=16)
names = ["Gemma 4 31B", "Gemma 4 26B", "Qwen3.8 27B", "Qwen3.6 27B"]
ans = R["P2"]["evidence_shapley_chance"]
shp = D["panels"]["P2"]["shapley_chance"]
for y in range(4):
    e, v = 100 * ans[y]["mean"], 100 * shp[y]["mean"]
    b.plot([e, v], [y, y], color=GRID, lw=2.5, solid_capstyle="round", zorder=1)
    b.scatter([e], [y], s=28, facecolors="white", edgecolors=BLUE, linewidths=1.4, zorder=3, label="evidence credit (asking)" if y == 0 else None)
    b.scatter([v], [y], s=28, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4, label="answer credit (team result)" if y == 0 else None)
b.set_yticks(range(4), names)
b.invert_yaxis()
b.set_xlabel("Shapley value in P2 (pts above chance)")
b.xaxis.grid(True, color=GRID, lw=0.6)
b.set_axisbelow(True)
b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7, handletextpad=0.2)
b.set_title("(b)", loc="left", fontsize=8, pad=16)
fig.tight_layout(w_pad=2.5)
save(fig, "fig2_asking_vs_reading")

# ---------------------------------------------------------------- Fig 3: error overlap vs null
rows = [("P0", "same model", "Same model (P0)"), ("P1", "persona", "Roles, Gemma (P1)"), ("P1Q", "persona", "Roles, Qwen (P1-Q)"),
        ("P2", "same family", "Same family (P2)"), ("P2", "cross family", "Cross family (P2)")]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for y, (p, t, lab) in enumerate(rows):
    x = D["error_overlap"][p]["by_type"][t]["same_given_both"]
    lo, hi = (100 * v for v in x["null_95"])
    ax.plot([lo, hi], [y, y], color=MUTED, lw=5, alpha=0.35, solid_capstyle="butt", label="chance (95% band)" if y == 0 else None)
    ax.plot([100 * x["null_mean"], 100 * x["obs"]], [y, y], color=GRID, lw=1, zorder=1)
    ax.scatter([100 * x["obs"]], [y], s=30, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4, label="observed" if y == 0 else None)
    ax.text(100 * x["obs"] + 2, y, f"{100 * x['obs']:.0f}%", va="center", fontsize=7, color=INK)
ax.set_yticks(range(len(rows)), [r[2] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Same wrong answer, when both agents are wrong (%)")
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7)
save(fig, "fig3_error_overlap")
print("written:", sorted(x.name for x in OUT.iterdir()))
