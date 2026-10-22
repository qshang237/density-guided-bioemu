import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

U = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TARGETS = [("OFS_9bh2", "Outward-facing (9bh2)", "low"), ("iOFS_9bh1", "Intermediate (9bh1)", "medium"), ("IFS_6x12", "Inward-facing (6x12)", "higher")]
THR = 2.5
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [  # label, color, files (relative), linestyle
    ("Unguided", "#8a8985", ["OFS_9bh2/base_fkc", "OFS_9bh2/base_smc", "OFS_9bh2/base_smc_seed2"], (0, (4, 3))),
    ("Particle selection (SMC)", "#2a78d6", ["{t}/smc_density_w30", "{t}/smc_density_w30_seed2"], "-"),
    ("Gradient + reweighting (FKC)", "#eb6834", ["{t}/fkc_full_w30"], "-"),
    ("Gradient only", "#1baf7a", ["{t}/fkc_grad_only_w30"], "-"),
]
GROUP = 10  # BioEmu batch = one resampling group of 10; groups are the independent unit
GS = np.arange(1, 21)
rng = np.random.default_rng(0)

def load(t, files):
    return np.concatenate([np.load(f"{U}/{f.format(t=t)}_per_sample.npz")[f"rmsd_{t}"] for f in files])

def p_hit(r, g, reps=3000):
    """P(at least one hit) when generating g whole groups of 10, drawn without replacement."""
    hits = (r[: len(r) // GROUP * GROUP] < THR).reshape(-1, GROUP).any(1)
    if g > len(hits): return np.nan
    idx = np.argsort(rng.random((reps, len(hits))), axis=1)[:, :g]
    return hits[idx].any(1).mean()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), facecolor=SURF, gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.55, "wspace": 0.28})
table = []
for j, (t, title, support) in enumerate(TARGETS):
    axb, axc = axes[0, j], axes[1, j]
    for ax in (axb, axc):
        ax.set_facecolor(SURF); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for i, (lab, col, files, ls) in enumerate(SERIES):
        r = load(t, files); k = int((r < THR).sum()); n = len(r)
        ci = binomtest(k, n).proportion_ci(method="wilson")
        table.append((t, lab, k, n, k / n, ci.low, ci.high))
        axb.bar(i, k / n * 100, width=0.62, color=col, edgecolor=SURF, linewidth=2, zorder=2)
        axb.errorbar(i, k / n * 100, yerr=[[100 * (k / n - ci.low)], [100 * (ci.high - k / n)]], color=INK, lw=1, capsize=3, zorder=3)
        axb.text(i, ci.high * 100 + 1.5, f"{k}/{n}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        y = [p_hit(r, g) for g in GS]
        axc.plot(GS * GROUP, y, color=col, lw=2, ls=ls, label=lab, zorder=3, marker="o", ms=3.5, mec=SURF, mew=0.8)
    axb.set_xticks([]); axb.set_ylim(0, 55)
    axb.set_title(f"{title}\nprior support: {support}", fontsize=9.5, color=INK, loc="left")
    axc.set_ylim(0, 1.02); axc.set_xlim(0, 205); axc.set_xticks([10, 50, 100, 150, 200])
    axc.set_xlabel("Structures generated", fontsize=8.5)
    axc.axhline(0.9, color=GRID, lw=1, ls=":", zorder=1)
    if j == 0:
        axb.set_ylabel(f"Recovery rate (%)\nRMSD < {THR} Å to target")
        axc.set_ylabel("P(at least one structure\nwithin 2.5 Å of target)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _, _ in SERIES]
fig.legend(handles, [s[0] for s in SERIES], loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
fig.text(0.5, 0.5, "Discovery: chance that at least one generated structure matches the cryo-EM state (groups of 10 drawn as units)",
         ha="center", fontsize=9, color=INK)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_core_gltph.png"), dpi=220, bbox_inches="tight", facecolor=SURF)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_core_gltph.pdf"), bbox_inches="tight", facecolor=SURF)

# k needed for 90% chance of a hit
print("target, condition, hits/n, rate, 95% CI")
for row in table: print(f"{row[0]:10s} {row[1]:30s} {row[2]:3d}/{row[3]}  {row[4]:.3f}  [{row[5]:.3f}, {row[6]:.3f}]")
for t, _, _ in TARGETS:
    for lab, _, files, _ in SERIES:
        r = load(t, files)
        k90 = next((g * GROUP for g in range(1, len(r) // GROUP + 1) if p_hit(r, g, 2000) >= 0.9), None)
        print(f"k for 90% hit  {t:10s} {lab:30s} {k90}")
