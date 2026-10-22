"""Collect per-target recovery and density ratio into results/figure_data.json (input of fig_scatter.py).

Recovery is recomputed from the sampled structures with evaluate.py's core-posed RMSD, so all
five proteins use the same code path. Density ratios come from density_ratio_steer.py output.

    python figures/build_figure_data.py --runs runs
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from evaluate import load_run, posed_rmsd  # noqa: E402
from proteins import PROTEINS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
ap.add_argument("--threshold", type=float, default=2.5)
a = ap.parse_args()

rows = []
for P, v in PROTEINS.items():
    refs = torch.load(os.path.join(ROOT, "targets", f"{P}_references.pt"), weights_only=False)
    core = refs.pop("_core_idx")
    base = os.path.join(a.runs, "base_fkc") if P == "GltPh" else os.path.join(a.runs, P, "base_fkc")
    xb = torch.tensor(load_run(base))
    for t in v["states"]:
        td = os.path.join(a.runs, t) if P == "GltPh" else os.path.join(a.runs, P, t)
        rec = {}
        for lab, d in (("unguided", None), ("smc", "smc_density_w30"), ("grad", "fkc_grad_only_w3000")):
            x = xb if d is None else torch.tensor(load_run(os.path.join(td, d)))
            r = posed_rmsd(x, refs[t], core).numpy()
            rec[lab] = [int((r < a.threshold).sum()), len(r)]
        dr = {}
        for lab, f in (("unguided", "unguided"), ("smc", f"smc_w30_aim_{t}"), ("grad", f"grad_w3000_aim_{t}"),
                       ("original", "original_v1.1")):
            z = np.load(os.path.join(ROOT, "results", "density_ratio", f"{P}_{f}.npz"))
            j = list(z["maps"]).index(t)
            dr[lab] = {"median": float(np.median(z["ratio"][:, j])), "best": float(z["ratio"][:, j].max())}
        rows.append({"protein": P, "target": t, "recovery": rec, "density_ratio": dr})
        print(P, t, rec)
json.dump(rows, open(os.path.join(ROOT, "results", "figure_data.json"), "w"), indent=1)
