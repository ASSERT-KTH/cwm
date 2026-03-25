#!/bin/bash
#
# Quick smoke test: can CWM load and generate with TP=4 on standard (non-fat) nodes?
#
#SBATCH -J cwm-tp4-test
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -o logs/tp4_test_%j.out
#SBATCH -e logs/tp4_test_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
CKPT=${CKPT:-./model_weights/cwm}

echo "=== TP=4 smoke test ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

${PYTHON} -m torch.distributed.run --nproc_per_node=4 \
    -m interp.scripts._tp4_smoke \
    checkpoint_dir=${CKPT} \
    tp_size=4

echo "=== Done ==="
