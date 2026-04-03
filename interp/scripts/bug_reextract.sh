#!/bin/bash
#
# Re-extract decode-step activation trajectories using prefill-based approach.
# Fixes empty trajectories caused by CUDA graph replay bypassing Python hooks.
#
# Root cause: FastGen uses CUDA graphs for decode (env FG_NO_CUDA_GRAPHS not set).
# CUDA graph replay bypasses Python _forward → hooks never fire → empty trajectories.
# Fix: pass prompt_tokens + generated_tokens as a single prefill → Python path always
# used for prefill → hooks fire → slice h[n_prompt:] for decode trajectory.
#
#SBATCH -J cwm-bug-reextract
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_reextract_%j.out
#SBATCH -e logs/bug_reextract_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Bug re-extraction: track=${TRACK}, N_GPUS=${N_GPUS} (TP=${TP_SIZE}, DP=${DP_SIZE}), seed=${SEED} ==="
echo "=== Writing to: ${TRAJ_DIR} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.run_bug_reextract \
    checkpoint_dir=${CKPT} \
    traj_dir=${TRAJ_DIR} \
    pairs_path=${PAIRS} \
    track=${TRACK} \
    seed=${SEED} \
    "layers=[16, 32, 48, 63]" \
    stride=5 \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Re-extraction complete: ${TRAJ_DIR} ==="
