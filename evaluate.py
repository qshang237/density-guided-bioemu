"""Evaluate sampling runs against all deposited states of one protein, with metrics independent of the reward.

Usage
-----
python evaluate.py --targets targets --target_state IFS_6x12 \
    --run base=runs/base_fkc --run smc=runs/IFS_6x12/smc_density \
    --run grad=runs/IFS_6x12/fkc_grad_only --run fkc=runs/IFS_6x12/fkc_full \
    --out results/IFS_6x12

Other proteins: add --protein SLC37A4 (default GltPh); target_state is the label in proteins.py.

Each --run is label=directory holding topology.pdb and samples.xtc (BioEmu output).

Per sample: RMSD to every deposited state (pose on the state-invariant core), CC to every
map (reported, not used for success), Calpha chain breaks and clashes, nearest state.
Per run: recovery rate (RMSD to target state below threshold), best RMSD, best-of-k curves
(sample efficiency), unique fraction (resampling duplicates), diversity, cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mdtraj as md
import numpy as np
import torch

from density_bioemu import BioEmuDensityPotential
from density_potential import kabsch

KS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def load_run(path: str) -> np.ndarray:
    top = md.load(os.path.join(path, "topology.pdb")).topology
    traj = md.load(os.path.join(path, "samples.xtc"), top=top)
    ca = top.select("name CA")
    return traj.xyz[:, ca, :] * 10.0  # (S, L, 3) A


def posed_rmsd(x: torch.Tensor, ref: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
    ok = ~torch.isnan(ref[:, 0])
    c = core[ok[core]]
    r, t = kabsch(x[:, c], ref[c])
    y = x @ r.transpose(-1, -2) + t
    return ((y[:, ok] - ref[ok]) ** 2).sum(-1).mean(-1).sqrt()


def geometry(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bonds = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
    breaks = (bonds > 4.5).sum(1)
    d = torch.cdist(x, x)
    n = x.shape[1]
    sep = (torch.arange(n)[:, None] - torch.arange(n)[None]).abs() >= 3
    clashes = ((d < 3.4) & sep).sum((1, 2)) // 2
    return breaks, clashes


def best_of_k(values: np.ndarray, success: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> dict:
    out = {}
    for k in KS:
        if k > len(values):
            break
        idx = np.stack([rng.choice(len(values), k, replace=False) for _ in range(reps)])
        out[str(k)] = {"mean_best_rmsd": float(values[idx].min(1).mean()), "p_success": float(success[idx].any(1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="targets")
    ap.add_argument("--protein", default="GltPh")
    ap.add_argument("--target_state", required=True, help="e.g. IFS_6x12")
    ap.add_argument("--run", action="append", required=True, help="label=dir")
    ap.add_argument("--threshold", type=float, default=2.5, help="RMSD (A) counted as recovering the target state")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    refs = torch.load(os.path.join(a.targets, f"{a.protein}_references.pt"), weights_only=False)
    core = refs.pop("_core_idx")
    states = list(refs)
    pots = {s: BioEmuDensityPotential(p) for s in states for p in glob.glob(os.path.join(a.targets, f"{a.protein}_{s}.pt"))}
    rng = np.random.default_rng(0)

    summary = {}
    for item in a.run:
        label, path = item.split("=", 1)
        x = torch.tensor(load_run(path))
        rm = {s: posed_rmsd(x, refs[s], core).numpy() for s in states}
        cc = {s: torch.cat([pots[s].cc(c / 10.0) for c in torch.split(x, 25)]).detach().numpy() for s in pots}
        brk, cl = geometry(x)
        valid = ((brk == 0) & (cl == 0)).numpy()
        nearest = np.array(states)[np.argmin(np.stack([rm[s] for s in states], 1), 1)]
        tr = rm[a.target_state]
        succ = tr < a.threshold
        # resampling duplicates
        flat = x.reshape(len(x), -1)
        dup = torch.cdist(flat, flat) < 1e-3
        unique = int((~torch.triu(dup, 1).any(0)).sum())
        sub = x[rng.choice(len(x), min(len(x), 100), replace=False)]
        pair = torch.stack([posed_rmsd(sub, sub[i], torch.arange(x.shape[1])) for i in range(len(sub))])
        diversity = float(pair[torch.triu(torch.ones_like(pair, dtype=torch.bool), 1)].mean())
        info = {}
        if os.path.exists(os.path.join(path, "run_info.json")):
            info = json.load(open(os.path.join(path, "run_info.json")))

        summary[label] = {
            "n": len(x),
            "recovery_rate": float(succ.mean()),
            "recovery_rate_1.5A": float((tr < 1.5).mean()),
            "nearest_is_target_fraction": float((nearest == a.target_state).mean()),
            "best_rmsd_to_target": float(tr.min()),
            "median_rmsd_to_target": float(np.median(tr)),
            "cc_target_median": float(np.median(cc[a.target_state])),
            "cc_target_top10_mean": float(np.sort(cc[a.target_state])[-10:].mean()),
            "nearest_state_counts": {s: int((nearest == s).sum()) for s in states},
            "valid_fraction": float(valid.mean()),
            "unique_fraction": unique / len(x),
            "mean_pairwise_rmsd": diversity,
            "wall_seconds": info.get("wall_seconds"),
            "seconds_per_sample": None if not info else info["wall_seconds"] / len(x),
            "best_of_k": best_of_k(tr, succ, rng),
        }
        np.savez(os.path.join(a.out, f"{label}_per_sample.npz"), **{f"rmsd_{s}": rm[s] for s in states},
                 **{f"cc_{s}": cc[s] for s in pots}, breaks=brk.numpy(), clashes=cl.numpy())
        s = summary[label]
        print(f"{label:>10}: n={s['n']:4d} recovery={s['recovery_rate']:.3f} best={s['best_rmsd_to_target']:.2f}A "
              f"median={s['median_rmsd_to_target']:.2f}A ccTop10={s['cc_target_top10_mean']:.3f} valid={s['valid_fraction']:.2f} "
              f"unique={s['unique_fraction']:.2f} div={s['mean_pairwise_rmsd']:.2f}A nearest={s['nearest_state_counts']}")

    json.dump({"target_state": a.target_state, "threshold_A": a.threshold, "runs": summary},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
