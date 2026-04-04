#!/bin/bash
#
# Finish iter2 CPU analysis: DMD (fixed: randomised SVD) + visualize.
# Run after bug_iter2_cpu.sh has completed CCS, T*, probe_content.
#
#SBATCH -J cwm-bug-iter2-finish
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_iter2_finish_%j.out
#SBATCH -e logs/bug_iter2_finish_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
TRACK=${TRACK:-track_a}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
LAYERS="[16, 32, 48, 63]"
SEED=${SEED:-42}

echo "=== Iter2 finish: DMD + visualize ==="
date

echo "--- A3: DMD (randomised SVD) ---"
${PYTHON} -m interp.bug_trace.analysis.dmd \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    seed=${SEED}

echo "--- Visualize ---"
${PYTHON} -m interp.bug_trace.analysis.visualize \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}"

echo "=== Finish complete ==="
date
