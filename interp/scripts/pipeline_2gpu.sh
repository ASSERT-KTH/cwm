#!/bin/bash
#
# Phase 4 only: steering sweep on 2× A100 80GB (TP=2, DP=1).
#
# Prerequisites: run phases123_cpu.sh first to produce index.jsonl,
# logit_lens.pt and probe outputs.
#
#SBATCH -J cwm-interp-steer
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH -C "fat"
#SBATCH -o logs/pipeline_%j.out
#SBATCH -e logs/pipeline_%j.err

set -euo pipefail
mkdir -p logs

EXTRACT_DIR=./interp-extract-trace_full
CKPT=./model_weights/cwm
PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

# Sanity-check that phases 1-3 have been run
if [ ! -s "${EXTRACT_DIR}/index.jsonl" ]; then
    echo "ERROR: ${EXTRACT_DIR}/index.jsonl is missing or empty."
    echo "Run phases123_cpu.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 4: Steering sweep — pure GPU work from minute 0
# 3 layers × 5 alphas × 50 samples, pre-tokenised to minimise inter-sample gaps
# ---------------------------------------------------------------------------
echo "=== Phase 4: Steering sweep ==="
$PYTHON -m torch.distributed.run --nproc_per_node=2 \
    -m interp.steering.run_steering \
    checkpoint_dir=${CKPT} \
    dump_dir=./interp-steer-correct_vs_incorrect \
    extract_dir=${EXTRACT_DIR} \
    condition=correct_vs_incorrect \
    "target_layers=[32, 48, 63]" \
    "alphas=[-2.0, -1.0, 0.0, 1.0, 2.0]" \
    n_samples=50 \
    mode=trace_full \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0

echo "=== Steering complete ==="
