#!/bin/bash
#
# Steering sweep using CCS directions from bug-trace analysis.
# Tests whether the CCS direction is causally relevant to bug fixing.
#
#SBATCH -J cwm-bug-steer
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_steer_%j.out
#SBATCH -e logs/bug_steer_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
DUMP_DIR=${DUMP_DIR:-./interp-bug-steer-${TRACK}}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

if [ ! -s "${TRAJ_DIR}/ccs_mean.pt" ]; then
    echo "ERROR: ${TRAJ_DIR}/ccs_mean.pt missing. Run bug_analysis_cpu.sh first."
    exit 1
fi

echo "=== Steering sweep: track=${TRACK}, N_GPUS=${N_GPUS}, seed=${SEED} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.analysis.activation_patch \
    checkpoint_dir=${CKPT} \
    traj_dir=${TRAJ_DIR} \
    pairs_path=${PAIRS} \
    dump_dir=${DUMP_DIR} \
    track=${TRACK} \
    "layers_to_patch=[16, 32, 48, 63]" \
    "time_positions=[0.0, 0.25, 0.5, 1.0]" \
    n_pairs=100 \
    seed=${SEED} \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Steering complete: ${DUMP_DIR} ==="
