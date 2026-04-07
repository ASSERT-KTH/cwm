#!/bin/bash
#
# Re-extract activation trajectories for bug-only dataset using prefill approach.
# Runs after bug_bugonly_extract.sh has completed generation.
#
#SBATCH -J cwm-bug-bugonly-reextract
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_bugonly_reextract_%j.out
#SBATCH -e logs/bug_bugonly_reextract_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-track_a-bugonly}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
CKPT=${CKPT:-./model_weights/cwm}
SEED=${SEED:-42}

TP_SIZE=2

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Bug-only RE-extraction (prefill-based): traj_dir=${TRAJ_DIR} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.run_bug_reextract \
    traj_dir=${TRAJ_DIR} \
    pairs_path=${PAIRS} \
    checkpoint_dir=${CKPT} \
    track=track_a \
    seed=${SEED} \
    "layers=[16, 32, 48, 63]" \
    stride=5 \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Bug-only RE-extraction complete ==="
