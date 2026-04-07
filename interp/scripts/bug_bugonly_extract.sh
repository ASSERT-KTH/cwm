#!/bin/bash
#
# Extract activation trajectories for buggy samples only (no originals).
# Avoids the contradictory-prompt confound: original samples get a prompt that says
# "this function has a bug" but has identical wrong/correct output.
# Writes to a separate dump dir so existing results are not overwritten.
#
#SBATCH -J cwm-bug-bugonly-extract
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_bugonly_extract_%j.out
#SBATCH -e logs/bug_bugonly_extract_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
DUMP_DIR=${DUMP_DIR:-./interp-bug-trajectories-track_a-bugonly}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Bug-only extraction: track=${TRACK}, N_GPUS=${N_GPUS} (TP=${TP_SIZE}, DP=${DP_SIZE}), seed=${SEED} ==="
echo "=== include_originals=False: only genuinely buggy samples will be extracted ==="
echo "=== dump_dir=${DUMP_DIR} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.run_bug_extract \
    checkpoint_dir=${CKPT} \
    dump_dir=${DUMP_DIR} \
    pairs_path=${PAIRS} \
    track=${TRACK} \
    seed=${SEED} \
    include_originals=False \
    "layers=[16, 32, 48, 63]" \
    stride=5 \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Bug-only extraction complete: ${DUMP_DIR} ==="
