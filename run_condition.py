"""Run one BioEmu sampling condition, optionally steered by a density target.

Examples
--------
# unguided reference (target-independent)
python run_condition.py --condition base_fkc --num_samples 200 --out runs/base_fkc

# guided, target = inward-facing map
python run_condition.py --condition fkc_full --target targets/GltPh_IFS_6x12.pt \
    --weight 30 --num_samples 200 --out runs/IFS_6x12/fkc_full_w30

# quick smoke test (2 batches, 20 denoising steps)
python run_condition.py --condition fkc_full --target targets/GltPh_IFS_6x12.pt --smoke --out runs/smoke

# another protein: the protein comes from the target file, or --protein for unguided runs;
# --batch sets the exact batch size (10 = one SMC group of 10 particles per batch)
python run_condition.py --condition base_fkc --protein SLC37A4 --batch 10 --out runs/SLC37A4/base_fkc
python run_condition.py --condition smc_density --target targets/SLC37A4_grpB_9ugu.pt --batch 10 \
    --out runs/SLC37A4/grpB_9ugu/smc_density_w30

Same --seed across conditions gives the same initial noise per batch.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so Hydra can import density_bioemu

from proteins import PROTEINS, protein_of_target  # noqa: E402

CONDITIONS = ["base_fkc", "base_smc", "smc_density", "fkc_grad_only", "fkc_full"]


def build_config(cond: str, target: str | None, weight: float, t_on: float, blur_res: float | None,
                 num_particles: int | None, N: int | None) -> dict:
    cfg = yaml.safe_load(open(HERE / "configs" / f"{cond}.yaml"))
    cfg = copy.deepcopy(cfg)
    if cfg.get("fk_potentials"):
        if target is None:
            raise SystemExit(f"--target is required for {cond}")
        for p in cfg["fk_potentials"]:
            p["target_path"] = str(Path(target).resolve())
            p["weight"] = weight
            p["t_on"] = t_on
            if blur_res is not None:
                p["blur_res"] = blur_res
    if num_particles is not None and cfg.get("steering_config"):
        cfg["steering_config"]["num_particles"] = num_particles
    if N is not None:
        cfg["N"] = N
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--weight", type=float, default=30.0)
    ap.add_argument("--t_on", type=float, default=0.5)
    ap.add_argument("--blur_res", type=float, default=None, help="broader Gaussian, e.g. 8 for a wider capture radius")
    ap.add_argument("--num_particles", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--protein", default=None, help="for unguided runs; guided runs take it from --target")
    ap.add_argument("--batch_size_100", type=int, default=180, help="180 gives batch 10 for GltPh (422 aa)")
    ap.add_argument("--batch", type=int, default=None, help="exact batch size; overrides --batch_size_100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_name", default="bioemu-v1.1", help="v1.1 matches the earlier unguided ensemble")
    ap.add_argument("--cache_embeds_dir", default=str(HERE / "cache_embeds"))
    ap.add_argument("--N", type=int, default=None, help="override denoising steps")
    ap.add_argument("--smoke", action="store_true", help="20 samples, 20 steps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpcore", "httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.smoke:
        a.num_samples, a.N = 20, 20

    protein = protein_of_target(a.target) if a.target else (a.protein or "GltPh")
    if a.protein and a.target and a.protein != protein:
        raise SystemExit(f"--protein {a.protein} does not match target {a.target}")
    sequence = PROTEINS[protein]["sequence"]
    if a.target:
        tseq = torch.load(a.target, map_location="cpu", weights_only=False)["sequence"]
        if tseq != sequence:
            raise SystemExit(f"target sequence does not match {protein}")
    if a.batch is not None:  # BioEmu uses batch = int(batch_size_100 * (100 / L) ** 2)
        L = len(sequence)
        b100 = math.ceil(a.batch * (L / 100) ** 2)
        while int(b100 * (100 / L) ** 2) > a.batch:
            b100 -= 1
        while int(b100 * (100 / L) ** 2) < a.batch:
            b100 += 1
        a.batch_size_100 = b100
    a.protein = protein

    cfg = build_config(a.condition, a.target, a.weight, a.t_on, a.blur_res, a.num_particles, a.N)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "denoiser_config.yaml", "w"))

    from bioemu.sample import main as bioemu_sample

    t0 = time.time()
    bioemu_sample(
        sequence=sequence,
        num_samples=a.num_samples,
        output_dir=str(out),
        batch_size_100=a.batch_size_100,
        model_name=a.model_name,
        denoiser_config=cfg,
        cache_embeds_dir=a.cache_embeds_dir,
        filter_samples=False,  # keep everything; physical validity is measured in evaluation
        base_seed=a.seed,
    )
    info = vars(a) | {
        "wall_seconds": time.time() - t0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
    }
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
