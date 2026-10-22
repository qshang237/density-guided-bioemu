"""Milestone 1 checks on a real map. All models must already be in the map frame.

Example
-------
python milestone1_checks.py \
    --map emd_XXXX.map --resolution 3.9 --threshold 0.02 --chain A \
    --native gltph_ofs.pdb \
    --decoy near=bioemu_frame_123.pdb --decoy wrong=gltph_ifs_fitted.pdb \
    --out m1_gltph_ofs.json

Reports
-------
1. CC for native and each decoy (expect native > near > wrong).
2. Mean CC versus Gaussian coordinate noise (expect smooth decrease).
3. Gradient finiteness and magnitude, plus cosine between the gradient at each
   decoy and the direction (native - decoy), at several blur widths.
4. Free Cartesian ascent from each decoy: CC and RMSD traces (diagnostic; we
   expect overfitting without a structural prior).
5. Cost of one potential call (forward + backward) per batch size.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from density_potential import (
    CryoEMDensityPotential,
    DensityScorer,
    load_mrc,
    preprocess_map,
    read_ca_pdb,
    refine_coordinates,
    rmsd,
)


def common_residues(ref_nums, mob_nums):
    pos_r = {r: i for i, r in enumerate(ref_nums)}
    pairs = [(pos_r[r], j) for j, r in enumerate(mob_nums) if r in pos_r]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--resolution", type=float, required=True)
    ap.add_argument("--threshold", type=float, default=None, help="contour level; EMDB recommended level if known")
    ap.add_argument("--native", required=True)
    ap.add_argument("--decoy", action="append", default=[], help="label=path.pdb, repeatable")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--bin", type=int, default=1)
    ap.add_argument("--dilate", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="milestone1_report.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    raw = load_mrc(args.map)
    dmap = preprocess_map(raw, threshold=args.threshold, dilate_A=args.dilate, bin_factor=args.bin)
    scorer = DensityScorer(dmap, resolution=args.resolution).to(dev)
    report = {
        "map_grid": list(dmap.shape),
        "voxel_A": dmap.voxel_size.tolist(),
        "mask_voxels": int(dmap.mask.sum()),
        "sigma_A": scorer.sigma,
    }

    nat_xyz, nat_nums = read_ca_pdb(args.native, args.chain)
    native_full = torch.tensor(nat_xyz, device=dev)
    frame_check = float(raw.to(dev).value_at(native_full).mean() / (raw.data.std() + 1e-8))
    report["native_density_in_raw_std_units"] = frame_check  # well above 1 means the frame is right

    # 1. discrimination (each decoy scored on its own residues; native on the same subset)
    report["scores"] = {"native": float(scorer(native_full[None])[0])}
    decoys = {}
    for item in args.decoy:
        label, path = item.split("=", 1)
        xyz, nums = read_ca_pdb(path, args.chain)
        ir, im = common_residues(nat_nums, nums)
        d = torch.tensor(xyz[im], device=dev)
        n = native_full[torch.as_tensor(ir, device=dev)]
        decoys[label] = (d, n)
        report["scores"][label] = {
            "cc": float(scorer(d[None])[0]),
            "native_cc_same_residues": float(scorer(n[None])[0]),
            "rmsd_to_native_A": float(rmsd(d, n)),
            "n_residues": int(len(ir)),
        }

    # 2. smoothness under noise
    g = torch.Generator(device=dev).manual_seed(0)
    noise = {}
    for lv in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        batch = native_full[None] + lv * torch.randn((8, *native_full.shape), generator=g, device=dev)
        noise[str(lv)] = float(scorer(batch).mean())
    report["cc_vs_noise_A"] = noise

    # 3. gradients and their alignment with the true direction, versus blur
    pot = CryoEMDensityPotential(scorer, weight=1.0)
    x = (native_full[None] / 10.0).clone().requires_grad_(True)
    pot(x).sum().backward()
    report["grad_finite"] = bool(torch.isfinite(x.grad).all())
    report["grad_max_abs"] = float(x.grad.abs().max())
    align = {}
    for label, (d, n) in decoys.items():
        row = {}
        for blur_res in [args.resolution, 8.0, 12.0, 16.0]:
            s = DensityScorer(dmap, resolution=args.resolution, sigma=0.225 * blur_res).to(dev)
            xx = d.clone().requires_grad_(True)
            gr = torch.autograd.grad(s(xx[None])[0], xx)[0]
            dirv = n - d
            row[f"blur_{blur_res:g}A"] = float((gr * dirv).sum() / (gr.norm() * dirv.norm() + 1e-12))
        align[label] = row
    report["grad_cosine_to_native"] = align

    # 4. free ascent diagnostic
    asc = {}
    for label, (d, n) in decoys.items():
        tr = refine_coordinates(scorer, d, steps=150, lr=0.05, reference_A=n)
        asc[label] = {
            "cc_start": tr["cc"][0], "cc_end": tr["cc"][-1],
            "rmsd_start": tr["rmsd"][0], "rmsd_min": min(tr["rmsd"]), "rmsd_end": tr["rmsd"][-1],
        }
    report["free_ascent"] = asc

    # 5. cost per potential call (forward + backward)
    timing = {}
    for b in [1, 4, 16]:
        xb = (native_full[None].repeat(b, 1, 1) / 10.0).requires_grad_(True)
        for _ in range(2):
            pot(xb).sum().backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            xb.grad = None
            pot(xb).sum().backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        timing[str(b)] = (time.perf_counter() - t0) / reps
    report["seconds_per_call_by_batch"] = timing
    report["note"] = "Compare seconds_per_call with one BioEmu score-network call at the same batch size."

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
