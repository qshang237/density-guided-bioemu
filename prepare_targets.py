"""Precompute guidance targets for BioEmu: one small .pt file per cryo-EM map (any protein in proteins.py).

Each target holds the preprocessed map (protomer region, cropped, binned, normalised,
fixed mask), its resolution, the deposited Calpha model indexed by BioEmu residue
index (0-based), and the state-invariant core used for the oracle pose.

Run once (CPU is fine), from the model-comparing folder layout:
    python prepare_targets.py --root /path/to/model-comparing --out targets/ --protein SLC37A4   (or all)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from density_potential import DensityMap, kabsch, load_mrc, preprocess_map, read_ca_pdb

from proteins import GLTPH_SEQ, PROTEINS  # noqa: F401  (GLTPH_SEQ kept for older imports)


def find_map(root: str, emd: str) -> str:
    for name in (f"EMD-{emd}.map", f"EMD-{emd}.map.gz", f"emd_{emd}.map", f"emd_{emd}.map.gz"):
        p = os.path.join(root, "emdb_maps", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no map for EMD-{emd} under {root}/emdb_maps")


def state_invariant_core(models: dict[str, np.ndarray], cut: float = 2.0) -> np.ndarray:
    """Largest set of residues that superpose within `cut` A across all states.

    models: label -> (L, 3) array with NaN for missing residues (BioEmu indexing).
    Seeds from every 30-residue window and grows by iterative refitting.
    """
    labs = list(models)
    present = np.all([~np.isnan(m[:, 0]) for m in models.values()], axis=0)
    common = np.nonzero(present)[0]
    M = {k: torch.tensor(v[common]) for k, v in models.items()}

    def sup(mob, tgt, idx):
        r, t = kabsch(mob[None, idx], tgt[idx])
        return (mob[None] @ r.transpose(-1, -2) + t)[0]

    def grow(seed):
        idx = seed
        for _ in range(20):
            ref = M[labs[0]]
            dev = torch.zeros(len(common))
            for lab in labs[1:]:
                dev = torch.maximum(dev, (sup(M[lab], ref, idx) - ref).norm(dim=-1))
            new = torch.nonzero(dev < cut).flatten()
            if len(new) < 10 or torch.equal(new, idx):
                return new
            idx = new
        return idx

    best = torch.arange(0)
    for s in range(0, len(common) - 30, 10):
        cand = grow(torch.arange(s, s + 30))
        if len(cand) > len(best):
            best = cand
    return common[best.numpy()]


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}


def read_ca_with_names(path: str, chain: str) -> tuple[np.ndarray, str]:
    """Calpha coordinates and one-letter sequence of one chain, in file order (first model, first altloc)."""
    xyz, seq = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and line[21] == chain:
                if line[16] not in (" ", "A"):
                    continue
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                seq.append(AA3TO1.get(line[17:20], "X"))
    return np.asarray(xyz, dtype=np.float32), "".join(seq)


def model_on_sequence(pdb_path: str, chain: str, sequence: str) -> np.ndarray:
    """Deposited Calpha model indexed by BioEmu residue (0-based), NaN where unmodelled.

    Residues are matched by global sequence alignment, not by author numbering, so numbering
    offsets and constructs with deleted segments (SPNS2) are handled.
    """
    from Bio import Align

    xyz, mseq = read_ca_with_names(pdb_path, chain)
    al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)
    al.end_insertion_score = 0
    al.end_deletion_score = 0
    aln = al.align(sequence, mseq)[0]
    arr = np.full((len(sequence), 3), np.nan, dtype=np.float32)
    n_pairs = n_same = 0
    for (s0, s1), (q0, q1) in zip(*aln.aligned):
        for i, j in zip(range(s0, s1), range(q0, q1)):
            arr[i] = xyz[j]
            n_pairs += 1
            n_same += sequence[i] == mseq[j]
    if n_pairs < 0.9 * len(mseq) or n_same < 0.95 * n_pairs:
        raise ValueError(f"{pdb_path}: poor alignment ({n_same}/{n_pairs} identical of {len(mseq)} modelled)")
    print(f"  {os.path.basename(pdb_path)} chain {chain}: {n_pairs} residues placed, {n_same} identical")
    return arr


def build(root: str, out: str, protein: str = "GltPh", bin_factor: int | None = None, region_A: float = 6.0,
          target_voxel_A: float = 1.7) -> None:
    """bin_factor None: choose per map so the binned voxel is close to target_voxel_A (GltPh gives 2, as before)."""
    os.makedirs(out, exist_ok=True)
    P = PROTEINS[protein]
    seq, chain = P["sequence"], P["chain"]
    pdb_dir = os.path.join(root, "pdb_structures")
    models = {lab: model_on_sequence(os.path.join(pdb_dir, f"{pdb}.pdb"), chain, seq) for lab, (_, pdb, _) in P["states"].items()}
    extra = {lab: model_on_sequence(os.path.join(pdb_dir, f"{pdb}.pdb"), chain, seq) for lab, pdb in P["extra_refs"].items()}

    core = state_invariant_core(models | extra)
    print(f"{protein} state-invariant core: {len(core)} residues (0-based), e.g. {core[:5]} ... {core[-5:]}")

    for lab, (emd, pdb, res) in P["states"].items():
        raw = load_mrc(find_map(root, emd))
        ref = torch.tensor(models[lab])
        nat = ref[~torch.isnan(ref[:, 0])]
        axes = raw.axes()
        lo, hi = nat.min(0).values - 12, nat.max(0).values + 12
        sl = [slice(int(torch.searchsorted(axes[d], lo[d])), int(torch.searchsorted(axes[d], hi[d]))) for d in range(3)]
        sub = raw.data[sl[0], sl[1], sl[2]].clone()
        origin = torch.stack([axes[d][sl[d].start] for d in range(3)])
        sub_map = DensityMap(data=sub, origin=origin, voxel_size=raw.voxel_size)
        grid = torch.stack(torch.meshgrid(*sub_map.axes(), indexing="ij"), -1).reshape(-1, 3)
        near = torch.zeros(grid.shape[0], dtype=torch.bool)
        for chunk in torch.split(nat, 64):
            near |= (torch.cdist(grid, chunk) < region_A).any(1)
        region = near.reshape(sub.shape).numpy()
        thr = float(torch.quantile(sub[torch.tensor(region)][::7], 0.25))
        bf = bin_factor or max(1, round(target_voxel_A / float(raw.voxel_size[0])))
        dmap = preprocess_map(sub_map, threshold=thr, dilate_A=2.0, pad_A=4.0, bin_factor=bf, region_mask=region)

        core_here = np.array([i for i in core if not np.isnan(models[lab][i, 0])])
        path = os.path.join(out, f"{protein}_{lab}.pt")
        torch.save(
            {
                "protein": protein, "label": lab, "emdb": emd, "pdb": pdb, "resolution": res, "sequence": seq,
                "map_data": dmap.data, "map_origin": dmap.origin, "map_voxel": dmap.voxel_size, "map_mask": dmap.mask,
                "reference_ca_A": ref,  # (L, 3), NaN where unmodelled
                "core_idx": torch.tensor(core_here, dtype=torch.long),
                "note": "protomer region and mask derived using the deposited model (mild leak, stated)",
            },
            path,
        )
        print(f"wrote {path}: raw voxel {float(raw.voxel_size[0]):.2f} A, bin {bf}, grid {tuple(dmap.shape)}, "
              f"voxel {float(dmap.voxel_size[0]):.2f} A, mask {int(dmap.mask.sum())}")

    # every deposited model (targets and evaluation-only refs) in one file, for evaluation
    refs = {k: torch.tensor(v) for k, v in (models | extra).items()}
    torch.save(refs | {"_core_idx": torch.tensor(core)}, os.path.join(out, f"{protein}_references.pt"))
    labs = list(refs)
    print("pairwise RMSD between deposited models after core pose (A):")
    for i, a_ in enumerate(labs):
        for b_ in labs[i + 1:]:
            ok = ~(torch.isnan(refs[a_][:, 0]) | torch.isnan(refs[b_][:, 0]))
            c = torch.tensor([k for k in core if ok[k]])
            r, t = kabsch(refs[a_][None, c], refs[b_][c])
            y = refs[a_][None] @ r.transpose(-1, -2) + t
            print(f"  {a_:>12} vs {b_:<12} {float(((y[0, ok] - refs[b_][ok]) ** 2).sum(-1).mean().sqrt()):6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="model-comparing folder")
    ap.add_argument("--out", default="targets")
    ap.add_argument("--protein", default="GltPh", choices=list(PROTEINS) + ["all"])
    ap.add_argument("--bin", type=int, default=None, help="default: per map, binned voxel close to 1.7 A")
    a = ap.parse_args()
    for p in (PROTEINS if a.protein == "all" else [a.protein]):
        build(a.root, a.out, p, a.bin)
