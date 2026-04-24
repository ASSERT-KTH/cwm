#!/bin/bash
#
# Build the probe cache: loads 485 .pt files, stacks into a single tensor, saves to disk.
# Run this once before launching GPU sweep agents.
#
#SBATCH -J cwm-swerl-probe-cache
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=256G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/swerl_probe_cache_%j.out
#SBATCH -e logs/swerl_probe_cache_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
EXTRACT_DIR=${EXTRACT_DIR:-./interp-swerl-extract}
CACHE_PATH=${CACHE_PATH:-${EXTRACT_DIR}/probe_cache_layer32.pt}

echo "=== Building probe cache ==="
$PYTHON -m interp.swerl.analysis.sweep_global_probe \
    extract_dir=${EXTRACT_DIR} \
    layer=32 \
    cache_path=${CACHE_PATH} \
    n_runs=0
echo "=== Cache built: ${CACHE_PATH} ==="
