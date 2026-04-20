#!/bin/bash
#SBATCH -J cwm-probes
#SBATCH -A berzelius-2026-111
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --partition=berzelius-cpu
#SBATCH -o logs/probes_%j.out
#SBATCH -e logs/probes_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
WORKDIR=/proj/assert-berzelius/users/x_andaf/cwm

# Pass via: sbatch --export=ALL,OUT_DIR=figures_500_capped,RUN_NAME=capped-500,MAX_BUCKET=50000
OUT_DIR=${OUT_DIR:-interp-swerl-extract/figures}
RUN_NAME=${RUN_NAME:-probes}
MAX_BUCKET=${MAX_BUCKET:-50000}
RUN_GLOBAL_PROBE=${RUN_GLOBAL_PROBE:-True}
MAX_GLOBAL_PROBE=${MAX_GLOBAL_PROBE:-200000}

cd ${WORKDIR}

echo "=== Probe run: out_dir=${OUT_DIR}, max_samples_per_bucket=${MAX_BUCKET} ==="
echo "=== Job started: $(date) ==="

${PYTHON} -m interp.swerl.analysis.run_probes \
  extract_dir=interp-swerl-extract \
  layer=32 \
  n_bins=10 \
  max_captures_per_file=5000 \
  max_samples_per_bucket=${MAX_BUCKET} \
  out_dir=${OUT_DIR} \
  run_global_probe=${RUN_GLOBAL_PROBE} \
  max_global_probe_samples=${MAX_GLOBAL_PROBE} \
  wandb_project=cwm-interp \
  wandb_run_name=${RUN_NAME}

echo "=== Probe run finished: $(date) ==="
