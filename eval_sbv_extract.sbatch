#!/bin/bash
#SBATCH -J cwm-sbv-extract
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/sbv_extract_%j.out
#SBATCH -e logs/sbv_extract_%j.err

set -euo pipefail
mkdir -p logs

export MODAL_PROFILE=assert-kth

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
WORKDIR=/proj/assert-berzelius/users/x_andaf/cwm

cd ${WORKDIR}

echo "=== SWEbench activation extraction: 500 instances, 8xA100-80GB (TP=2, DP=4) ==="
echo "=== Resume-safe: already-extracted .pt files are skipped automatically ==="
echo "=== Resubmit this job to continue after a wall-clock timeout ==="
echo "=== Job started: $(date) ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=8 \
  -m interp.swerl.run_swerl_extract -- \
  config=eval_sbv_extract.yaml \
  dump_dir=interp-swerl-extract \
  checkpoint_dir=./model_weights/cwm \
  "layers=[32]" \
  num_rollout_threads=2

echo "=== Extraction finished: $(date) ==="
