#!/bin/bash
#
# Activation patching experiment (GPU, requires pre-computed CCS directions).
# Run bug_analysis_cpu.sh first.
#
#SBATCH -J cwm-bug-patch
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -C "fat"
#SBATCH -o logs/bug_patch_%j.out
#SBATCH -e logs/bug_patch_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-4}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
DUMP_DIR=${DUMP_DIR:-./interp-bug-patch-${TRACK}}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

if [ ! -s "${TRAJ_DIR}/ccs_mean.pt" ]; then
    echo "ERROR: ${TRAJ_DIR}/ccs_mean.pt missing. Run bug_analysis_cpu.sh first."
    exit 1
fi

echo "=== Activation patching: track=${TRACK}, N_GPUS=${N_GPUS}, seed=${SEED} ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.analysis.activation_patch \
    checkpoint_dir=${CKPT} \
    traj_dir=${TRAJ_DIR} \
    pairs_path=${PAIRS} \
    dump_dir=${DUMP_DIR} \
    track=${TRACK} \
    "layers_to_patch=[16, 32, 48, 63]" \
    "time_positions=[0.1, 0.25, 0.5, 0.75, 1.0]" \
    n_pairs=50 \
    seed=${SEED} \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Patching complete: ${DUMP_DIR} ==="
