#!/bin/bash
#
# W&B hyperparameter sweep for the global SWEbench outcome probe (CPU only).
#
#SBATCH -J cwm-swerl-probe-sweep
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=256G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/swerl_probe_sweep_%j.out
#SBATCH -e logs/swerl_probe_sweep_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
EXTRACT_DIR=${EXTRACT_DIR:-./interp-swerl-extract}
N_RUNS=${N_RUNS:-60}
SWEEP_ID=${SWEEP_ID:-""}

CMD="$PYTHON -m interp.swerl.analysis.sweep_global_probe \
    extract_dir=${EXTRACT_DIR} \
    layer=32 \
    n_runs=${N_RUNS} \
    wandb_project=cwm-interp"

if [ -n "${SWEEP_ID}" ]; then
    CMD="${CMD} sweep_id=${SWEEP_ID}"
fi

echo "=== Launching sweep: n_runs=${N_RUNS} ==="
$CMD
echo "=== Sweep complete ==="
