#!/bin/bash
#
# W&B hyperparameter sweep agents for the global SWEbench outcome probe (GPU).
# Requires the cache to exist first — run swerl_probe_cache.sh before this.
#
#SBATCH -J cwm-swerl-probe-sweep-gpu
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p berzelius
#SBATCH -o logs/swerl_probe_sweep_gpu_%j.out
#SBATCH -e logs/swerl_probe_sweep_gpu_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
EXTRACT_DIR=${EXTRACT_DIR:-./interp-swerl-extract}
CACHE_PATH=${CACHE_PATH:-${EXTRACT_DIR}/probe_cache_layer32.pt}
N_RUNS=${N_RUNS:-25}
SWEEP_ID=${SWEEP_ID:-""}

if [ ! -f "${CACHE_PATH}" ]; then
    echo "ERROR: cache not found at ${CACHE_PATH}. Run swerl_probe_cache.sh first."
    exit 1
fi

CMD="$PYTHON -m interp.swerl.analysis.sweep_global_probe \
    extract_dir=${EXTRACT_DIR} \
    cache_path=${CACHE_PATH} \
    layer=32 \
    n_runs=${N_RUNS} \
    wandb_project=cwm-interp"

if [ -n "${SWEEP_ID}" ]; then
    CMD="${CMD} sweep_id=${SWEEP_ID}"
fi

echo "=== Launching GPU sweep agent: n_runs=${N_RUNS} ==="
$CMD
echo "=== Agent complete ==="
