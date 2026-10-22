#!/bin/bash
# Density-guided BioEmu on Apocrita (QMUL). Usage, from this folder on the HPC:
#   bash submit_jobs.sh setup    # once: check the bioemu env has steering, run the CPU tests
#   bash submit_jobs.sh smoke    # one short GPU job: memory, speed, MSA/embedding cache
#   bash submit_jobs.sh main     # 11 GPU jobs: 2 unguided references + 3 targets x 3 guided conditions
#   bash submit_jobs.sh sweep    # optional, after main: guidance strength 10 and 100 on all targets
#   bash submit_jobs.sh eval     # one CPU job: evaluate everything into results/
#   bash submit_jobs.sh multi    # 24 GPU jobs: the other four proteins (SLC37A4, SPNS2, GPR4, AuxinTransporter)
#   bash submit_jobs.sh evalmulti   # run on the login node after multi: evaluate the four proteins into results/<protein>/
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_RUN="/data/home/bty189/bioemu-run/bioemu-run-2/GltPh"   # earlier unguided v1.1 ensemble (486 frames)
ACCOUNT="pilot_sae_gpu"; PARTITION="sae"
NUM_SAMPLES=200
WEIGHT=30
TARGETS="OFS_9bh2 iOFS_9bh1 IFS_6x12"
mkdir -p "$BASE/logs" "$BASE/runs" "$BASE/results"

ENV_SETUP='module load miniforge
mamba activate bioemu
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH='"$BASE"':${PYTHONPATH:-}'

submit_gpu () {  # name, hours, command, [extra sbatch option]; prints the job id into LAST_JOB
  LAST_JOB=$(sbatch --parsable ${4:-} <<SLURM
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=${2}:00:00
#SBATCH --job-name=dg_${1}
#SBATCH --output=${BASE}/logs/${1}_%j.out
${ENV_SETUP}
cd ${BASE}
nvidia-smi --query-gpu=name,memory.total --format=csv
${3}
SLURM
)
  echo "submitted ${1} (job ${LAST_JOB})"
}

# targets per protein for the multi mode (labels as in proteins.py)
declare -A MULTI_TARGETS=(
  [SLC37A4]="grpA_21nq grpB_9ugu"
  [SPNS2]="s1_8ex5 s2_8ex7"
  [GPR4]="pHa_8zfd pHb_9jvh"
  [AuxinTransporter]="d1_9ute d2_9jdr"
)
# exact batch size for gradient runs (gradient memory grows with length; AUX1 has 485 residues)
declare -A GRAD_BATCH=([SLC37A4]=5 [SPNS2]=5 [GPR4]=5 [AuxinTransporter]=4)

check_multi_targets () {  # stop before submitting anything if a target file is missing (16 Oct: upload had silently failed)
  local missing=0
  for p in "${!MULTI_TARGETS[@]}"; do
    for f in targets/${p}_references.pt $(for t in ${MULTI_TARGETS[$p]}; do echo targets/${p}_${t}.pt; done); do
      [ -f "${BASE}/${f}" ] || { echo "missing ${f}"; missing=1; }
    done
  done
  [ $missing -eq 0 ] || { echo "upload the missing target files first; nothing submitted"; exit 1; }
}

case "${1:-}" in
setup)
  set +u   # conda/mamba activation scripts reference unset variables
  eval "$ENV_SETUP"
  cd "$BASE"
  if ! python -c "import bioemu.steering.dpm_fkc" 2>/dev/null; then
    echo "This bioemu version has no FKC steering. Upgrading inside the bioemu env..."
    pip install -U "git+https://github.com/microsoft/bioemu.git"
  fi
  python -c "import bioemu, bioemu.steering.dpm_fkc, bioemu.steering.dpm_smc; print('bioemu', getattr(bioemu, '__version__', '?'), 'steering OK')"
  pip install -q pytest mdtraj "numpy<2"   # BioEmu embeddings use AlphaFold/TensorFlow code that needs numpy 1.x
  python -m pytest -q tests/test_density_potential.py -k "not roundtrip"
  python -m pytest -q tests/test_bioemu_loop.py
  ;;
smoke)
  submit_gpu smoke 1 "python run_condition.py --condition fkc_full --target targets/GltPh_IFS_6x12.pt --weight ${WEIGHT} --smoke --out runs/smoke_fkc_full"
  ;;
diagnose)
  # 4 short jobs, 10 samples each, to locate an instability: solver steps vs guidance
  T=targets/GltPh_IFS_6x12.pt
  submit_gpu diag_base_N20  1 "python run_condition.py --condition base_fkc --num_samples 10 --N 20  --out runs/diag_base_N20"
  submit_gpu diag_base_N100 1 "python run_condition.py --condition base_fkc --num_samples 10 --N 100 --out runs/diag_base_N100"
  submit_gpu diag_smc_N100  1 "python run_condition.py --condition smc_density --target ${T} --weight ${WEIGHT} --num_samples 10 --N 100 --out runs/diag_smc_N100"
  submit_gpu diag_fkc_N100  1 "python run_condition.py --condition fkc_full --target ${T} --weight ${WEIGHT} --num_samples 10 --N 100 --out runs/diag_fkc_N100"
  ;;
main)
  for c in base_fkc base_smc; do
    submit_gpu "${c}" 6 "python run_condition.py --condition ${c} --num_samples ${NUM_SAMPLES} --out runs/${c}"
  done
  for t in $TARGETS; do
    for c in smc_density fkc_grad_only fkc_full; do
      submit_gpu "${t}_${c}_w${WEIGHT}" 8 "python run_condition.py --condition ${c} --target targets/GltPh_${t}.pt --weight ${WEIGHT} --num_samples ${NUM_SAMPLES} --out runs/${t}/${c}_w${WEIGHT}"
    done
  done
  ;;
missing)
  # resubmit main runs that have no samples.xtc and are not queued or running (finished batches are kept)
  QUEUED=$(squeue -h -u "$USER" -o %j || true)
  todo=()
  for c in base_fkc base_smc; do todo+=("${c}|${c}|runs/${c}|6|"); done
  for t in $TARGETS; do for c in smc_density fkc_grad_only fkc_full; do
    todo+=("${t}_${c}_w${WEIGHT}|${c}|runs/${t}/${c}_w${WEIGHT}|8|--target targets/GltPh_${t}.pt --weight ${WEIGHT}")
  done; done
  for item in "${todo[@]}"; do
    IFS='|' read -r name cond out hours extra <<< "$item"
    if [ -f "${out}/samples.xtc" ]; then continue; fi
    if grep -qx "dg_${name}" <<< "$QUEUED"; then echo "still queued/running: ${name}"; continue; fi
    submit_gpu "${name}" "${hours}" "python run_condition.py --condition ${cond} ${extra} --num_samples ${NUM_SAMPLES} --out ${out}"
  done
  ;;
blur)
  # gradient conditions with a broader Gaussian (wider capture radius), tests the capture-radius explanation
  for t in $TARGETS; do
    for c in fkc_grad_only fkc_full; do
      submit_gpu "${t}_${c}_w${WEIGHT}_blur8" 8 "python run_condition.py --condition ${c} --target targets/GltPh_${t}.pt --weight ${WEIGHT} --blur_res 8 --num_samples ${NUM_SAMPLES} --out runs/${t}/${c}_w${WEIGHT}_blur8"
    done
  done
  ;;
seed2)
  # independent replicate of the cheap conditions (different initial noise)
  submit_gpu base_smc_seed2 6 "python run_condition.py --condition base_smc --seed 100000 --num_samples ${NUM_SAMPLES} --out runs/base_smc_seed2"
  for t in $TARGETS; do
    submit_gpu "${t}_smc_density_w${WEIGHT}_seed2" 6 "python run_condition.py --condition smc_density --target targets/GltPh_${t}.pt --weight ${WEIGHT} --seed 100000 --num_samples ${NUM_SAMPLES} --out runs/${t}/smc_density_w${WEIGHT}_seed2"
  done
  ;;
gradsweep)
  # dose response of the gradient alone: is there a strength where it moves samples before it breaks them?
  for w in 300 3000; do
    submit_gpu "IFS_6x12_fkc_grad_only_w${w}" 8 "python run_condition.py --condition fkc_grad_only --target targets/GltPh_IFS_6x12.pt --weight ${w} --num_samples 100 --out runs/IFS_6x12/fkc_grad_only_w${w}"
  done
  ;;
gradstrong)
  # follow-up to the w3000 result: replicate it, test the other targets, and bracket the strength.
  # batch 5 (--batch_size_100 90) avoids out-of-memory on 44 GB GPUs; gradient-only has no particle interaction
  G="--condition fkc_grad_only --num_samples 100 --batch_size_100 90"
  submit_gpu OFS_9bh2_grad_w3000   8 "python run_condition.py ${G} --target targets/GltPh_OFS_9bh2.pt --weight 3000 --out runs/OFS_9bh2/fkc_grad_only_w3000"
  submit_gpu iOFS_9bh1_grad_w3000  8 "python run_condition.py ${G} --target targets/GltPh_iOFS_9bh1.pt --weight 3000 --out runs/iOFS_9bh1/fkc_grad_only_w3000"
  submit_gpu IFS_6x12_grad_w3000_s2 8 "python run_condition.py ${G} --target targets/GltPh_IFS_6x12.pt --weight 3000 --seed 100000 --out runs/IFS_6x12/fkc_grad_only_w3000_seed2"
  submit_gpu IFS_6x12_grad_w1000   8 "python run_condition.py ${G} --target targets/GltPh_IFS_6x12.pt --weight 1000 --out runs/IFS_6x12/fkc_grad_only_w1000"
  submit_gpu IFS_6x12_grad_w10000  8 "python run_condition.py ${G} --target targets/GltPh_IFS_6x12.pt --weight 10000 --out runs/IFS_6x12/fkc_grad_only_w10000"
  ;;
sweep)
  for t in $TARGETS; do
    for w in 10 100; do
      for c in smc_density fkc_full; do
        submit_gpu "${t}_${c}_w${w}" 8 "python run_condition.py --condition ${c} --target targets/GltPh_${t}.pt --weight ${w} --num_samples ${NUM_SAMPLES} --out runs/${t}/${c}_w${w}"
      done
    done
  done
  ;;
eval)
  sbatch <<SLURM
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=dg_eval
#SBATCH --output=${BASE}/logs/eval_%j.out
${ENV_SETUP}
cd ${BASE}
for t in ${TARGETS}; do
  RUNS="--run original_v1.1=${ORIGINAL_RUN} --run base_fkc=runs/base_fkc --run base_smc=runs/base_smc"
  for d in runs/\${t}/*/; do
    [ -f "\${d}samples.xtc" ] && RUNS="\${RUNS} --run \$(basename \${d})=\${d}"
  done
  python evaluate.py --targets targets --target_state \${t} \${RUNS} --out results/\${t}
done
SLURM
  echo "submitted eval"
  ;;
multi)
  # Per protein: unguided references (FKC and SMC solvers, 200 each), then per target map
  # SMC selection at weight 30 (200) and gradient-only at weight 3000 (100): the two settings that
  # worked on GltPh. The first job of each protein computes the MSA/embedding; the rest wait for it.
  check_multi_targets
  for p in "${!MULTI_TARGETS[@]}"; do
    submit_gpu "${p}_base_fkc" 6 "python run_condition.py --condition base_fkc --protein ${p} --batch 10 --num_samples ${NUM_SAMPLES} --out runs/${p}/base_fkc"
    DEP="--dependency=afterany:${LAST_JOB}"
    submit_gpu "${p}_base_smc" 6 "python run_condition.py --condition base_smc --protein ${p} --batch 10 --num_samples ${NUM_SAMPLES} --out runs/${p}/base_smc" "$DEP"
    for t in ${MULTI_TARGETS[$p]}; do
      submit_gpu "${p}_${t}_smc_w${WEIGHT}" 6 "python run_condition.py --condition smc_density --target targets/${p}_${t}.pt --weight ${WEIGHT} --batch 10 --num_samples ${NUM_SAMPLES} --out runs/${p}/${t}/smc_density_w${WEIGHT}" "$DEP"
      submit_gpu "${p}_${t}_grad_w3000" 8 "python run_condition.py --condition fkc_grad_only --target targets/${p}_${t}.pt --weight 3000 --batch ${GRAD_BATCH[$p]} --num_samples 100 --out runs/${p}/${t}/fkc_grad_only_w3000" "$DEP"
    done
  done
  ;;
multimissing)
  # resubmit multi runs that have no samples.xtc and are not queued or running
  check_multi_targets
  QUEUED=$(squeue -h -u "$USER" -o %j || true)
  for p in "${!MULTI_TARGETS[@]}"; do
    todo=("${p}_base_fkc|--condition base_fkc --protein ${p} --batch 10 --num_samples ${NUM_SAMPLES}|runs/${p}/base_fkc"
          "${p}_base_smc|--condition base_smc --protein ${p} --batch 10 --num_samples ${NUM_SAMPLES}|runs/${p}/base_smc")
    for t in ${MULTI_TARGETS[$p]}; do
      todo+=("${p}_${t}_smc_w${WEIGHT}|--condition smc_density --target targets/${p}_${t}.pt --weight ${WEIGHT} --batch 10 --num_samples ${NUM_SAMPLES}|runs/${p}/${t}/smc_density_w${WEIGHT}"
             "${p}_${t}_grad_w3000|--condition fkc_grad_only --target targets/${p}_${t}.pt --weight 3000 --batch ${GRAD_BATCH[$p]} --num_samples 100|runs/${p}/${t}/fkc_grad_only_w3000")
    done
    for item in "${todo[@]}"; do
      IFS='|' read -r name args out <<< "$item"
      if [ -f "${out}/samples.xtc" ]; then continue; fi
      if grep -qx "dg_${name}" <<< "$QUEUED"; then echo "still queued/running: ${name}"; continue; fi
      submit_gpu "${name}" 8 "python run_condition.py ${args} --out ${out}"
    done
  done
  ;;
evalmulti)
  # CPU only, a few minutes per protein; run directly on the login node
  set +u
  eval "$ENV_SETUP"
  cd "$BASE"
  for p in "${!MULTI_TARGETS[@]}"; do
    ORIG="/data/home/bty189/bioemu-run/bioemu-run-2/${p}"
    for t in ${MULTI_TARGETS[$p]}; do
      RUNS=""
      [ -f "${ORIG}/samples.xtc" ] && RUNS="--run original_v1.1=${ORIG}"
      for d in runs/${p}/base_fkc runs/${p}/base_smc runs/${p}/${t}/*; do
        [ -f "${d}/samples.xtc" ] && RUNS="${RUNS} --run $(basename ${d})=${d}"
      done
      echo "=== ${p} target ${t}"
      python -u evaluate.py --protein ${p} --targets targets --target_state ${t} ${RUNS} --out results/${p}/${t}
    done
  done
  ;;
*)
  echo "usage: bash submit_jobs.sh {setup|smoke|diagnose|main|missing|blur|seed2|gradsweep|gradstrong|sweep|eval|multi|multimissing|evalmulti}"; exit 1 ;;
esac
