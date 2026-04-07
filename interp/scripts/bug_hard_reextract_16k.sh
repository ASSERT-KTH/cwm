#!/bin/bash
#
# Prefill-based re-extraction for 16k hard-mutation trajectories.
# Run after bug_hard_extract_16k.sh — fills empty trajectory dicts
# caused by CUDA graph bypass during decode.
#
#SBATCH -J cwm-bug-hard-reextract-16k
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_hard_reextract_16k_%j.out
#SBATCH -e logs/bug_hard_reextract_16k_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-hard-16k}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs_hard.json}
CKPT=${CKPT:-./model_weights/cwm}
SEED=${SEED:-42}

TP_SIZE=4

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Hard-mutation 16k RE-extraction (prefill-based): traj_dir=${TRAJ_DIR} ==="

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

echo "=== Hard-mutation 16k RE-extraction complete ==="
