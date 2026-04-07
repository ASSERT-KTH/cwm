#!/bin/bash
#
# Re-extract activation trajectories for hard-mutation dataset using prefill approach.
# num_cuda_graphs=0 does NOT disable CUDA graphs in FastGen (only controls count).
# The real bypass is prefill-based re-extraction in run_bug_reextract.py.
#
# Expects: interp-bug-trajectories-hard/ already has .pt files with generated_text
# but empty trajectory dicts from the prior run_bug_extract.py run.
#
#SBATCH -J cwm-bug-hard-reextract
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_hard_reextract_%j.out
#SBATCH -e logs/bug_hard_reextract_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-hard}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs_hard.json}
CKPT=${CKPT:-./model_weights/cwm}
SEED=${SEED:-42}

TP_SIZE=2

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Hard-mutation RE-extraction (prefill-based): traj_dir=${TRAJ_DIR} ==="
echo "=== Fixes empty trajectory dicts caused by CUDA graph bypass ==="

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

echo "=== Hard RE-extraction complete ==="
