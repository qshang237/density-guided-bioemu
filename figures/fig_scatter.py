"""Main result figure: guided vs unguided, per target map (11 targets, 5 proteins).

Left: recovery (fraction within 2.5 A Calpha-RMSD of the target state), Wilson 95% CI.
Right: median density ratio against the target map (evaluation-project metric).
Data: results/figure_data.json (built from runs/ and results/density_ratio/).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "..", "results", "figure_data.json")))

SEL, GRAD = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated all-pairs)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SHORT = {"GltPh": "GltPh", "SLC37A4": "SLC37A4", "SPNS2": "SPNS2", "GPR4": "GPR4", "AuxinTransporter": "AUX1"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def label(r):
    return f"{SHORT[r['protein']]} {r['target'].split('_')[0]}"


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=200)

# left: recovery
ax = axes[0]
for r in rows:
    k0, n0 = r["recovery"]["unguided"]
    x, _, _ = wilson(k0, n0)
    for key, col in (("smc", SEL), ("grad", GRAD)):
        k, n = r["recovery"][key]
        y, lo, hi = wilson(k, n)
        ax.plot([x * 100, x * 100], [lo * 100, hi * 100], color=col, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(x * 100, y * 100, s=58, color=col, edgecolor="white", linewidth=1.6, zorder=3)
ax.plot([0, 100], [0, 100], ls=(0, (4, 4)), color=MUTED, lw=1, zorder=1)
ax.set_xlim(-3, 100); ax.set_ylim(-3, 100)
ax.set_xlabel("Unguided recovery (%)"); ax.set_ylabel("Guided recovery (%)")
ax.set_title("Recovery of the target state (Cα-RMSD < 2.5 Å)", color=INK, fontsize=11, loc="left")
# selective labels, placed next to the selection point of each labelled target
LEFT_LABELS = {("GltPh", "OFS_9bh2"): (2, 1), ("SLC37A4", "grpA_21nq"): (4, 4),
               ("SLC37A4", "grpB_9ugu"): (2.5, -1), ("SPNS2", "s1_8ex5"): (-14, 2), ("GltPh", "iOFS_9bh1"): (-16, 2)}
for r in rows:
    key = (r["protein"], r["target"])
    if key in LEFT_LABELS:
        x = wilson(*r["recovery"]["unguided"])[0] * 100
        y = wilson(*r["recovery"]["smc"])[0] * 100
        dx, dy = LEFT_LABELS[key]
        txt = "SLC37A4 grpA, AUX1 d1" if key == ("SLC37A4", "grpA_21nq") else label(r)
        ax.annotate(txt, xy=(x + dx, y + dy), fontsize=8.5, color=INK2, va="center")
ax.axvspan(-3, 3, color=GRID, alpha=0.6, zorder=0, lw=0)
ax.annotate("shaded: targets the model\nalmost never samples (< 3%)", xy=(4.5, 97), fontsize=8.5, color=MUTED, va="top")

# right: density ratio
ax = axes[1]
for r in rows:
    x = r["density_ratio"]["unguided"]["median"]
    for key, col in (("smc", SEL), ("grad", GRAD)):
        ax.scatter(x, r["density_ratio"][key]["median"], s=58, color=col, edgecolor="white", linewidth=1.6, zorder=3)
ax.plot([0.2, 0.85], [0.2, 0.85], ls=(0, (4, 4)), color=MUTED, lw=1, zorder=1)
ax.set_xlim(0.2, 0.85); ax.set_ylim(0.2, 0.85)
ax.set_xlabel("Unguided density ratio (median)"); ax.set_ylabel("Guided density ratio (median)")
ax.set_title("Fit to the target map (density ratio)", color=INK, fontsize=11, loc="left")
for r in rows:
    RIGHT_LABELS = {("SLC37A4", "grpB_9ugu"): ("smc", 0.012, 0), ("SLC37A4", "grpA_21nq"): ("grad", 0.012, 0),
                    ("SPNS2", "s2_8ex7"): ("smc", 0.012, 0), ("AuxinTransporter", "d2_9jdr"): ("grad", 0.012, 0.012)}
    key = (r["protein"], r["target"])
    if key in RIGHT_LABELS:
        k, dx, dy = RIGHT_LABELS[key]
        x = r["density_ratio"]["unguided"]["median"]
        y = r["density_ratio"][k]["median"]
        ax.annotate(label(r), xy=(x + dx, y + dy), fontsize=8.5, color=INK2, va="center", ha="left" if dx > 0 else "right")

for ax in axes:
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("equal")
h1 = plt.Line2D([], [], marker="o", ls="", color=SEL, markeredgecolor="white", markersize=8, label="SMC selection (w = 30)")
h2 = plt.Line2D([], [], marker="o", ls="", color=GRAD, markeredgecolor="white", markersize=8, label="Gradient guidance (w = 3000)")
h3 = plt.Line2D([], [], ls=(0, (4, 4)), color=MUTED, label="no change")
fig.legend(handles=[h1, h2, h3], loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.01, "Each point is one target map (11 maps, 5 proteins). Error bars: Wilson 95% CI.",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(os.path.join(HERE, "fig_main_scatter.png"), facecolor="white")
fig.savefig(os.path.join(HERE, "fig_main_scatter.pdf"), facecolor="white")
