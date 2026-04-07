#!/bin/bash
#
# Extract hard-mutation trajectories with 16k context (4× the default 4096).
# ~48% of samples hit the 4096-token limit in the original run; extending to
# 16384 allows full reasoning chains to complete.
#
# Uses TP=4 to distribute the larger KV cache across 4 GPUs per rank,
# DP=2 for 2× throughput (8 GPUs total: 2 TP-groups of 4).
#
#SBATCH -J cwm-bug-hard-extract-16k
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_hard_extract_16k_%j.out
#SBATCH -e logs/bug_hard_extract_16k_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs_hard.json}
CKPT=${CKPT:-./model_weights/cwm}
DUMP_DIR=${DUMP_DIR:-./interp-bug-trajectories-hard-16k}
SEED=${SEED:-42}

TP_SIZE=4
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

echo "=== Hard-mutation extraction (16k context): N_GPUS=${N_GPUS} (TP=${TP_SIZE}, DP=${DP_SIZE}) ==="
echo "=== max_gen=16384, dump_dir=${DUMP_DIR} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.run_bug_extract \
    checkpoint_dir=${CKPT} \
    dump_dir=${DUMP_DIR} \
    pairs_path=${PAIRS} \
    track=track_a \
    seed=${SEED} \
    include_originals=False \
    max_gen=16384 \
    "layers=[16, 32, 48, 63]" \
    stride=5 \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Hard-mutation 16k extraction complete: ${DUMP_DIR} ==="
