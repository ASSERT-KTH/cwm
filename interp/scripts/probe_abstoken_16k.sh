#!/bin/bash
#SBATCH -J cwm-probe-abstoken-hard-16k
#SBATCH -t 03:00:00
#SBATCH -N 1 -n 8 --mem=64G -p berzelius-cpu
#SBATCH -o logs/probe_abstoken_hard_16k_%j.out
#SBATCH -e logs/probe_abstoken_hard_16k_%j.err
set -euo pipefail
mkdir -p logs
PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
echo "=== Absolute-token probe on hard 16k mutations (buggy-only, 200-token buckets) ==="
${PYTHON} -m interp.bug_trace.analysis.probe_abstoken \
    traj_dir=./interp-bug-trajectories-hard-16k \
    bucket_width=200 \
    "layers=[16, 32, 48, 63]" \
    epochs=30 \
    seed=42
echo "=== Done ==="
