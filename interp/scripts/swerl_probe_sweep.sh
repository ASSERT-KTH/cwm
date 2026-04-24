#!/bin/bash
#
# W&B hyperparameter sweep for the global SWEbench outcome probe (CPU only).
#
#SBATCH -J cwm-swerl-probe-sweep
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/swerl_probe_sweep_%j.out
#SBATCH -e logs/swerl_probe_sweep_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
EXTRACT_DIR=${EXTRACT_DIR:-./interp-swerl-extract}
CACHE_PATH=${CACHE_PATH:-${EXTRACT_DIR}/probe_cache_layer32.pt}
N_RUNS=${N_RUNS:-60}
SWEEP_ID=${SWEEP_ID:-""}

CMD="$PYTHON -m interp.swerl.analysis.sweep_global_probe \
    extract_dir=${EXTRACT_DIR} \
    layer=32 \
    n_runs=${N_RUNS} \
    cache_path=${CACHE_PATH} \
    wandb_project=cwm-interp"

if [ -n "${SWEEP_ID}" ]; then
    CMD="${CMD} sweep_id=${SWEEP_ID}"
fi

echo "=== Launching sweep: n_runs=${N_RUNS} ==="
$CMD
echo "=== Sweep complete ==="
