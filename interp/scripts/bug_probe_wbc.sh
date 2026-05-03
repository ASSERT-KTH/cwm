#!/bin/bash
#
# Train will_be_correct temporal probe for a given split strategy.
#
# Single run:
#   SPLIT_BY=original sbatch interp/scripts/bug_probe_wbc.sh
#
# Sweep (create):
#   SPLIT_BY=original N_RUNS=30 sbatch interp/scripts/bug_probe_wbc.sh
#
# Sweep (join existing):
#   SPLIT_BY=original N_RUNS=20 SWEEP_ID=<id> sbatch interp/scripts/bug_probe_wbc.sh
#
#SBATCH -J cwm-probe-wbc
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p berzelius-cpu
#SBATCH -o logs/probe_wbc_%j.out
#SBATCH -e logs/probe_wbc_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-track_a}
SPLIT_BY=${SPLIT_BY:-sample}
N_RUNS=${N_RUNS:-0}
SWEEP_ID=${SWEEP_ID:-""}
SEED=${SEED:-42}
LAYERS="[16, 32, 48, 63]"
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-512}
EPOCHS=${EPOCHS:-30}
PATIENCE=${PATIENCE:-5}

echo "=== Probe WBC: traj_dir=${TRAJ_DIR} split_by=${SPLIT_BY} n_runs=${N_RUNS} ==="

if [ ! -s "${TRAJ_DIR}/index.jsonl" ]; then
    echo "ERROR: ${TRAJ_DIR}/index.jsonl missing."
    exit 1
fi

SWEEP_ARG=${SWEEP_ID:+sweep_id=${SWEEP_ID}}

${PYTHON} -m interp.bug_trace.analysis.probe_temporal \
    traj_dir=${TRAJ_DIR} \
    target=will_be_correct \
    "layers=${LAYERS}" \
    split_by=${SPLIT_BY} \
    n_runs=${N_RUNS} \
    ${SWEEP_ARG} \
    seed=${SEED} \
    lr=${LR} \
    weight_decay=${WEIGHT_DECAY} \
    batch_size=${BATCH_SIZE} \
    epochs=${EPOCHS} \
    patience=${PATIENCE} \
    wandb_run_name="probe_wbc_${SPLIT_BY}"

echo "=== Done ==="
