#!/bin/bash
#
# Fast focused steering check: layer 32 only, 30 pairs, 3 alpha values.
# Estimates ~30 jobs/rank × ~120s/gen = ~60 min on 2 DP ranks (4 GPUs).
#
# Full 8-GPU version (bug_steer.sh) crashed at 54 min due to moodist TCP errors
# and would have timed out anyway (22h estimated). This script fits in 2h wall.
#
#SBATCH -J cwm-bug-steer-fast
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -C "fat"
#SBATCH -o logs/bug_steer_fast_%j.out
#SBATCH -e logs/bug_steer_fast_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-4}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
DUMP_DIR=${DUMP_DIR:-./interp-bug-steer-fast-${TRACK}}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

if [ ! -s "${TRAJ_DIR}/ccs_mean.pt" ]; then
    echo "ERROR: ${TRAJ_DIR}/ccs_mean.pt missing. Run bug_analysis_cpu.sh first."
    exit 1
fi

echo "=== Steering sweep (fast): track=${TRACK}, N_GPUS=${N_GPUS}, seed=${SEED} ==="
echo "=== Layer 32 only, 30 pairs, alpha in [0.0, 0.5, 1.0] ==="

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.analysis.activation_patch \
    checkpoint_dir=${CKPT} \
    traj_dir=${TRAJ_DIR} \
    pairs_path=${PAIRS} \
    dump_dir=${DUMP_DIR} \
    track=${TRACK} \
    "layers_to_patch=[32]" \
    "time_positions=[0.0, 0.5, 1.0]" \
    n_pairs=30 \
    seed=${SEED} \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Steering complete (fast): ${DUMP_DIR} ==="
