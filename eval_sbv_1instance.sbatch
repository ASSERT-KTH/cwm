#!/bin/bash
#SBATCH -J cwm-sbv-1instance
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH -C "fat"
#SBATCH -o logs/sbv_1instance_%j.out
#SBATCH -e logs/sbv_1instance_%j.err

set -euo pipefail
mkdir -p logs

export MODAL_PROFILE=assert-kth

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
WORKDIR=/proj/assert-berzelius/users/x_andaf/cwm

cd ${WORKDIR}

echo "=== SWEbench 1-instance eval: instance=django__django-11119, GPUs=2 (TP=2, DP=1) ==="
echo "=== Job started: $(date) ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=2 \
  -m evals.main -- \
  config=eval_sbv_1instance.yaml \
  dump_dir=eval-cwm-sbv-1instance \
  checkpoint_dir=./model_weights/cwm

echo "=== Job finished: $(date) ==="
