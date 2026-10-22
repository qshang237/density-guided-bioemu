"""Density ratio (the evaluation project's metric) for the steering runs.

Uses shared_fitting.py from the model-comparing project unchanged: each ensemble is
superposed onto the deposited model (sequence-aligned common Calpha), the smoothed and
z-scored experimental map is interpolated at backbone atoms (N, CA, C, O), and the mean is
divided by the same quantity for the deposited model (density ratio, 1 = as good as deposited).

This is a second, partly independent judge: the guidance reward is a Calpha-Gaussian masked CC
on a cropped, binned map; the density ratio reads the smoothed full map at backbone atoms.

Usage (from the density-guided-bioemu folder, with model-comparing alongside it):
    python density_ratio_steer.py --mc_root ../model-comparing --runs runs --out results/density_ratio
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mdtraj as md
import numpy as np

from proteins import PROTEINS

GLTPH_FLAT = True  # GltPh runs live in runs/<target>/..., the other proteins in runs/<protein>/<target>/...


def run_dirs(runs: str, protein: str, orig_root: str) -> list[tuple[str, str, str | None]]:
    """(label, directory, aimed target or None) for the core conditions of one protein."""
    out = []
    orig = os.path.join(orig_root, protein)
    if os.path.exists(os.path.join(orig, "samples.xtc")):
        out.append(("original_v1.1", orig, None))
    base = os.path.join(runs, "base_fkc") if protein == "GltPh" else os.path.join(runs, protein, "base_fkc")
    out.append(("unguided", base, None))
    for t in PROTEINS[protein]["states"]:
        tdir = os.path.join(runs, t) if protein == "GltPh" else os.path.join(runs, protein, t)
        out.append((f"smc_w30_aim_{t}", os.path.join(tdir, "smc_density_w30"), t))
        out.append((f"grad_w3000_aim_{t}", os.path.join(tdir, "fkc_grad_only_w3000"), t))
    return [o for o in out if os.path.exists(os.path.join(o[1], "samples.xtc"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc_root", default="../model-comparing", help="model-comparing folder (shared_fitting.py, emdb_maps, pdb_structures)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--orig_root", default=None, help="default: <mc_root>/bioemu_results")
    ap.add_argument("--out", default="results/density_ratio")
    ap.add_argument("--protein", default="all")
    a = ap.parse_args()
    runs, out = os.path.abspath(a.runs), os.path.abspath(a.out)
    mc_root = os.path.abspath(a.mc_root)
    orig_root = os.path.abspath(a.orig_root or os.path.join(mc_root, "bioemu_results"))
    os.makedirs(out, exist_ok=True)

    sys.path.insert(0, mc_root)
    cwd = os.getcwd()
    os.chdir(mc_root)  # shared_fitting uses paths relative to the model-comparing folder
    from shared_fitting import compute_score_matrix, load_exp_data

    summary = {}
    for protein in (PROTEINS if a.protein == "all" else [a.protein]):
        states = PROTEINS[protein]["states"]
        print(f"\n===== {protein}", flush=True)
        emdb_list = [(f"EMD-{emd}", pdb, res) for emd, pdb, res in states.values()]
        exp = load_exp_data(emdb_list)
        ids = [f"EMD-{states[t][0]}" for t in states]
        summary[protein] = {}
        for label, d, aim in run_dirs(runs, protein, orig_root):
            traj = md.load(os.path.join(d, "samples.xtc"), top=os.path.join(d, "topology.pdb"))
            S = compute_score_matrix(traj, exp, ids)  # (frames, maps)
            np.savez(os.path.join(out, f"{protein}_{label}.npz"), ratio=S, maps=np.array(list(states)))
            row = {}
            for j, t in enumerate(states):
                r = S[:, j]
                row[t] = {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)), "best": float(r.max()),
                          "frac_ge_0.9": float((r >= 0.9).mean())}
            summary[protein][label] = {"n": int(S.shape[0]), "aim": aim, "per_map": row}
            print(f"{label:>28} n={S.shape[0]:3d} " + "  ".join(
                f"{t}: med {row[t]['median']:.3f} p90 {row[t]['p90']:.3f} best {row[t]['best']:.3f}" for t in states),
                flush=True)
    os.chdir(cwd)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
