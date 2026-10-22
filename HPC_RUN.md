# Running Milestone 2 on Apocrita

## What gets run

Five sampling conditions, all BioEmu v1.1, 100 denoising steps, noise 1.0, batch 10, same seeds:

| Condition | Density gradient | Resampling | Config |
|---|---|---|---|
| base_fkc | no | no | `configs/base_fkc.yaml` (unguided, FKC solver) |
| base_smc | no | no | `configs/base_smc.yaml` (unguided, SMC solver) |
| smc_density | no | yes | `configs/smc_density.yaml` |
| fkc_grad_only | yes | no | `configs/fkc_grad_only.yaml` |
| fkc_full | yes | yes | `configs/fkc_full.yaml` |

Guided conditions use groups of 10 particles, guidance switched on smoothly from t = 0.5, strength (weight) 30.
Each guided condition is run against each of the three GltPh maps, which differ in how well the unguided prior covers them:

| Target | Unguided recovery (RMSD < 2.5 A, 486 frames) | Prior support |
|---|---|---|
| OFS_9bh2 | 2.5% | low |
| iOFS_9bh1 | 7.4% | medium |
| IFS_6x12 | 11.9% | higher |

## Steps

```bash
# 1. copy this folder to the HPC (from the Mac)
scp -r ~/density-guided-bioemu bty189@login.hpc.qmul.ac.uk:/data/home/bty189/

# 2. on the HPC
cd /data/home/bty189/density-guided-bioemu
bash submit_jobs.sh setup      # checks bioemu steering, runs CPU tests (about 2 min)
bash submit_jobs.sh smoke      # one short GPU job; read logs/smoke_*.out
bash submit_jobs.sh main       # 11 jobs
squeue -u bty189
bash submit_jobs.sh eval       # after all main jobs finish
```

## What to check in the smoke log before submitting `main`

* It finishes and prints `run_info.json` with `max_gpu_mem_GB`. If it ran out of memory, rerun with fewer particles: edit `submit_jobs.sh` to pass `--num_particles 5 --batch_size_100 90`.
* `wall_seconds` for 20 samples at 20 steps. Main jobs use 200 samples at 100 steps, so expect roughly 50 times longer.
* Lines like `ESS=0.xx`. If ESS stays near 1.00 the guidance is too weak to select anything (try weight 100); if it collapses to about 0.1 at every step it is too strong (try 10). The `sweep` mode runs both.

## Bring the results back

```bash
scp -r bty189@login.hpc.qmul.ac.uk:/data/home/bty189/density-guided-bioemu/results ~/density-guided-bioemu/
scp -r bty189@login.hpc.qmul.ac.uk:/data/home/bty189/density-guided-bioemu/logs ~/density-guided-bioemu/
```

`results/<target>/summary.json` holds recovery rate, best RMSD, best-of-k sample-efficiency curves, validity, duplicates and diversity for every condition.
