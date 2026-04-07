#!/bin/bash
#
# Extract decode-step activation trajectories for hard-mutation bug experiments.
# Uses 8 GPUs (TP=2, DP=4). Hard mutations go to a separate trajectory dir.
#
#SBATCH -J cwm-bug-hard-extract
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_hard_extract_%j.out
#SBATCH -e logs/bug_hard_extract_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs_hard.json}
DUMP_DIR=${DUMP_DIR:-./interp-bug-trajectories-hard}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Hard-mutation extraction: track=${TRACK}, N_GPUS=${N_GPUS} (TP=${TP_SIZE}, DP=${DP_SIZE}), seed=${SEED} ==="
echo "=== pairs=${PAIRS}, dump_dir=${DUMP_DIR} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.run_bug_extract \
    checkpoint_dir=${CKPT} \
    dump_dir=${DUMP_DIR} \
    pairs_path=${PAIRS} \
    track=${TRACK} \
    seed=${SEED} \
    "layers=[16, 32, 48, 63]" \
    stride=5 \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Extraction complete: ${DUMP_DIR} ==="
