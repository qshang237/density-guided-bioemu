"""Milestone 1 on real GltPh data: 3 maps x (3 deposited models + 500 BioEmu frames).

Pose: every model is superposed onto the map's own deposited model using only a
state-invariant scaffold core, found automatically by iterative outlier rejection
across the three deposited states. The pose therefore carries no information about
which state the model is in (Pilot A condition).

Protomer mask: map threshold intersected with a 6 A neighbourhood of the map's own
deposited chain A. This is a mild leak (model-derived region), stated in the report;
replace with map segmentation later.
"""

import json
import sys
import time

import mdtraj as md
import numpy as np
import torch

from density_potential import (
    DensityScorer, kabsch, load_mrc, preprocess_map, read_ca_pdb, refine_coordinates, rmsd,
)

torch.set_num_threads(2)
import os
# defaults assume model-comparing/ sits alongside this folder (override with MC_ROOT)
U = os.environ.get("MC_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model-comparing"))
W = os.path.join(U, "emdb_maps")
OUT = sys.argv[1] if len(sys.argv) > 1 else "gltph_m1"

STATES = {  # label: (emdb, pdb, resolution)
    "OFS_9bh2": ("44530", "9bh2", 2.70),
    "iOFS_9bh1": ("44529", "9bh1", 3.10),
    "IFS_6x12": ("21986", "6x12", 3.50),
}

# ---------------------------------------------------------------- models, by residue number
models = {}
for lab, (_, pdb, _) in STATES.items():
    xyz, nums = read_ca_pdb(f"{U}/pdb_structures/{pdb}.pdb", "A")
    models[lab] = dict(zip(nums, xyz))

top = md.load(f"{U}/bioemu_results/GltPh/topology.pdb").topology
traj = md.load(f"{U}/bioemu_results/GltPh/samples.xtc", top=top)
ca_idx = top.select("name CA")
bio_xyz = traj.xyz[:, ca_idx, :] * 10.0  # nm -> A
bio_nums = [top.atom(i).residue.resSeq + 1 for i in ca_idx]  # BioEmu is 0-based
bio_res = [top.atom(i).residue.name for i in ca_idx]

# check numbering agreement on residue names
def resnames(path):
    out = {}
    for line in open(path):
        if line.startswith("ATOM") and line[12:16] == " CA " and line[21] == "A":
            out[int(line[22:26])] = line[17:20]
    return out
dep_names = resnames(f"{U}/pdb_structures/9bh2.pdb")
agree = np.mean([dep_names[n] == r for n, r in zip(bio_nums, bio_res) if n in dep_names])
print(f"BioEmu frames {traj.n_frames}, CA {len(ca_idx)}, residue-name agreement with 9bh2: {agree:.3f}")
assert agree > 0.95, "residue numbering differs between BioEmu and deposited models"

common = sorted(set(bio_nums).intersection(*[set(m) for m in models.values()]))
print("residues common to all models:", len(common))
bio_pos = {n: i for i, n in enumerate(bio_nums)}
B = torch.tensor(bio_xyz[:, [bio_pos[n] for n in common]])  # (500, Nc, 3)
M = {lab: torch.tensor(np.array([models[lab][n] for n in common])) for lab in models}

# ---------------------------------------------------------------- scaffold core
def superpose(mob, tgt, idx):
    r, t = kabsch(mob[None, idx], tgt[idx])
    return (mob[None] @ r.transpose(-1, -2) + t)[0]

def inliers_from(seed, cut=2.0, iters=20):
    idx = seed
    for _ in range(iters):
        ref = M[labs[0]]
        dev = torch.zeros(len(common))
        for lab in labs[1:]:
            dev = torch.maximum(dev, (superpose(M[lab], ref, idx) - ref).norm(dim=-1))
        new = torch.nonzero(dev < cut).flatten()
        if len(new) < 10 or torch.equal(new, idx):
            return new
        idx = new
    return idx

labs = list(M)
best = torch.arange(0)
for start in range(0, len(common) - 30, 10):  # seed from every 30-residue window, keep the largest rigid core
    cand = inliers_from(torch.arange(start, start + 30))
    if len(cand) > len(best):
        best = cand
idx = best
core = idx
core_nums = [common[i] for i in core.tolist()]
print(f"scaffold core: {len(core)} residues, ranges:", end=" ")
runs, s = [], core_nums[0]
for a, b in zip(core_nums, core_nums[1:] + [None]):
    if b != a + 1:
        runs.append(f"{s}-{a}")
        s = b
print(", ".join(runs))

def pose_to(mob_batch, target):
    """Superpose a batch onto target using the scaffold core only."""
    r, t = kabsch(mob_batch[:, core], target[core])
    return mob_batch @ r.transpose(-1, -2) + t

# pairwise RMSD between deposited states (core-superposed)
pair = {f"{a}|{b}": float(rmsd(pose_to(M[a][None], M[b])[0], M[b])) for a in labs for b in labs if a < b}
print("deposited pairwise RMSD (core superposed):", {k: round(v, 2) for k, v in pair.items()})

# ---------------------------------------------------------------- per-map evaluation
report = {"core_residues": len(core), "core_ranges": runs, "deposited_pairwise_rmsd": pair, "maps": {}}
allcc = {}
for lab, (emd, pdb, res) in STATES.items():
    t0 = time.time()
    raw = load_mrc(next(f for f in (f"{W}/EMD-{emd}.map", f"{W}/EMD-{emd}.map.gz") if os.path.exists(f)))
    nat_full = torch.tensor(np.array(list(models[lab].values())))
    # protomer region: within 6 A of the deposited chain (stated leak)
    axes = raw.axes()
    lo = nat_full.min(0).values - 12
    hi = nat_full.max(0).values + 12
    sl = [slice(int(torch.searchsorted(axes[d], lo[d])), int(torch.searchsorted(axes[d], hi[d]))) for d in range(3)]
    sub = raw.data[sl[0], sl[1], sl[2]]
    origin = torch.stack([axes[d][sl[d].start] for d in range(3)])
    from density_potential import DensityMap
    sub_map = DensityMap(data=sub.clone(), origin=origin, voxel_size=raw.voxel_size)
    gx = [a for a in sub_map.axes()]
    grid = torch.stack(torch.meshgrid(*gx, indexing="ij"), -1).reshape(-1, 3)
    near = torch.zeros(grid.shape[0], dtype=torch.bool)
    for chunk in torch.split(nat_full, 64):
        near |= (torch.cdist(grid, chunk) < 6.0).any(1)
    region = near.reshape(sub.shape).numpy()
    vals = sub[torch.tensor(region)]
    thr = float(vals.mean() - 0.0 * vals.std())  # keep the protomer envelope
    thr = float(torch.quantile(vals[::7], 0.25))
    dmap = preprocess_map(sub_map, threshold=thr, dilate_A=2.0, pad_A=4.0, bin_factor=2, region_mask=region)
    scorer = DensityScorer(dmap, resolution=res)

    tgt = M[lab]
    # deposited models of every state, posed into this map
    dep = torch.stack([pose_to(M[o][None], tgt)[0] for o in labs])
    dep_cc = scorer(dep)
    # BioEmu frames
    Bp = pose_to(B, tgt)
    bio_cc = torch.cat([scorer(c) for c in torch.split(Bp, 25)])
    bio_rmsd = rmsd(Bp, tgt[None])
    # independent metric: RMSD of the non-core (moving) region
    move = torch.tensor([i for i in range(len(common)) if i not in set(core.tolist())])
    bio_rmsd_move = rmsd(Bp[:, move], tgt[None, move])
    rank = torch.argsort(bio_cc, descending=True)
    sp = float(np.corrcoef(np.argsort(np.argsort(-bio_cc.numpy())), np.argsort(np.argsort(bio_rmsd.numpy())))[0, 1])

    # noise smoothness on native
    g = torch.Generator().manual_seed(0)
    noise = {lv: float(scorer(tgt[None] + lv * torch.randn((4, *tgt.shape), generator=g)).mean()) for lv in [0, 0.5, 1, 2, 3, 4]}

    # gradient alignment at the best BioEmu frame and the other states, matched vs broad blur
    def cosine(start, sig_res):
        s = DensityScorer(dmap, resolution=res, sigma=0.225 * sig_res)
        x = start.clone().requires_grad_(True)
        gr = torch.autograd.grad(s(x[None])[0], x)[0]
        d = tgt - start
        return float((gr * d).sum() / (gr.norm() * d.norm() + 1e-12))
    cos = {}
    starts = {"best_bioemu": Bp[rank[0]], "bioemu_297": Bp[297]}
    starts.update({f"dep_{o}": dep[i] for i, o in enumerate(labs) if o != lab})
    for k, st in starts.items():
        cos[k] = {f"blur_{b:g}A": cosine(st, b) for b in [res, 8.0, 12.0]}
    # gradient finiteness
    x = Bp[:4].clone().requires_grad_(True)
    scorer(x).sum().backward()
    # timing (forward + backward) per structure
    xb = Bp[:8].clone().requires_grad_(True)
    t1 = time.time(); scorer(xb).sum().backward(); per = (time.time() - t1) / 8

    allcc[lab] = bio_cc
    report["maps"][lab] = {
        "emdb": emd, "resolution": res, "grid_after_bin": list(dmap.shape), "voxel_A": float(dmap.voxel_size[0]),
        "mask_voxels": int(dmap.mask.sum()),
        "deposited_cc": {o: float(c) for o, c in zip(labs, dep_cc)},
        "bioemu_cc": {"max": float(bio_cc.max()), "argmax_frame": int(rank[0]), "median": float(bio_cc.median()),
                      "frame_297": float(bio_cc[297]), "top10_frames": rank[:10].tolist()},
        "bioemu_rmsd_to_native": {"min": float(bio_rmsd.min()), "argmin_frame": int(bio_rmsd.argmin()),
                                  "of_cc_best": float(bio_rmsd[rank[0]]), "frame_297": float(bio_rmsd[297]),
                                  "moving_region_of_cc_best": float(bio_rmsd_move[rank[0]])},
        "spearman_cc_vs_rmsd": -sp,
        "cc_vs_noise_A": noise,
        "grad_finite": bool(torch.isfinite(x.grad).all()),
        "grad_cosine_to_native": cos,
        "seconds_per_structure_fwd_bwd_cpu": per,
        "seconds_total": time.time() - t0,
    }
    np.save(f"{OUT}_{lab}_bioemu_cc.npy", bio_cc.numpy())
    np.save(f"{OUT}_{lab}_bioemu_rmsd.npy", bio_rmsd.numpy())
    print(lab, json.dumps(report["maps"][lab], indent=1), flush=True)

# state assignment: which map does each BioEmu frame prefer (relative to each map's native CC)?
rel = torch.stack([allcc[l] / report["maps"][l]["deposited_cc"][l] for l in labs], 1)
pref = rel.argmax(1)
report["bioemu_preferred_map_counts"] = {l: int((pref == i).sum()) for i, l in enumerate(labs)}
json.dump(report, open(f"{OUT}_report.json", "w"), indent=1)
print("preferred map counts:", report["bioemu_preferred_map_counts"])
