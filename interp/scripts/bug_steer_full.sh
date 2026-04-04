#!/bin/bash
#
# Full steering sweep (S1): all test-split pairs, alpha in {0.0, 0.5, 1.0}.
#
# Sizing: 249 test pairs × 3 alphas / 4 DP ranks = ~187 per rank × 150s ≈ 7.8h
# Request 12h wall. Evaluates only layer 32 (the program-fate hub).
#
# IMPORTANT: ccs_mean.pt and ccs_split.json must exist in TRAJ_DIR.
# Run bug_iter2_cpu.sh (which runs ccs.py with test_fraction=0.3) first.
#
#SBATCH -J cwm-bug-steer-full
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/bug_steer_full_%j.out
#SBATCH -e logs/bug_steer_full_%j.err

set -euo pipefail
mkdir -p logs

N_GPUS=${N_GPUS:-8}
TRACK=${TRACK:-track_a}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=${PAIRS:-./interp/bug_trace/data/pairs.json}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
DUMP_DIR=${DUMP_DIR:-./interp-steer-correct_vs_incorrect}

TP_SIZE=2
DP_SIZE=$((N_GPUS / TP_SIZE))

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

if [ ! -s "${TRAJ_DIR}/ccs_mean.pt" ]; then
    echo "ERROR: ${TRAJ_DIR}/ccs_mean.pt missing. Run bug_iter2_cpu.sh first."
    exit 1
fi

if [ ! -s "${TRAJ_DIR}/ccs_split.json" ]; then
    echo "ERROR: ${TRAJ_DIR}/ccs_split.json missing. Run bug_iter2_cpu.sh (which runs ccs.py with test_fraction=0.3)."
    exit 1
fi

echo "=== Full steering sweep: track=${TRACK}, N_GPUS=${N_GPUS}, seed=${SEED} ==="
echo "    Using CCS test split from ${TRAJ_DIR}/ccs_split.json"
echo "    DP_SIZE=${DP_SIZE}, TP_SIZE=${TP_SIZE}"
date

${PYTHON} -m torch.distributed.run --nproc_per_node=${N_GPUS} \
    -m interp.bug_trace.analysis.activation_patch \
    checkpoint_dir=${CKPT} \
    traj_dir=${TRAJ_DIR} \
    pairs_path=${PAIRS} \
    dump_dir=${DUMP_DIR} \
    track=${TRACK} \
    "layers_to_patch=[32]" \
    "time_positions=[0.0, 0.5, 1.0]" \
    n_pairs=0 \
    ccs_split_path=${TRAJ_DIR}/ccs_split.json \
    seed=${SEED} \
    gen_args.tp_size=${TP_SIZE} \
    gen_args.num_cuda_graphs=0

echo "=== Steering complete: ${DUMP_DIR} ==="
date
