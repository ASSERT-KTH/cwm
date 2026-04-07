#!/bin/bash
#
# Analysis for bug-only trajectories: will_be_correct probe only.
# No CCS, no is_buggy probe — both require original/buggy contrast which is
# methodologically unsound when originals are excluded.
#
# Key question: can a linear probe on decode-step hidden states predict early in
# generation whether the model will successfully fix the bug?
#
#SBATCH -J cwm-bug-bugonly-analysis
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_bugonly_analysis_%j.out
#SBATCH -e logs/bug_bugonly_analysis_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-track_a-bugonly}
SEED=${SEED:-42}
LAYERS="[16, 32, 48, 63]"

echo "=== Bug-only analysis: traj_dir=${TRAJ_DIR} ==="

if [ ! -s "${TRAJ_DIR}/index.jsonl" ]; then
    echo "ERROR: ${TRAJ_DIR}/index.jsonl missing. Run bug_bugonly_extract.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Primary question: will_be_correct probe — does the model "know" early?
# ---------------------------------------------------------------------------
echo "--- Probe temporal: will_be_correct ---"
${PYTHON} -m interp.bug_trace.analysis.probe_temporal \
    traj_dir=${TRAJ_DIR} \
    target=will_be_correct \
    "layers=${LAYERS}" \
    seed=${SEED}

# ---------------------------------------------------------------------------
# T* text correlation: when does the representation shift?
# ---------------------------------------------------------------------------
echo "--- T* text correlation ---"
${PYTHON} -m interp.bug_trace.analysis.t_star_text \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}"

# ---------------------------------------------------------------------------
# Change point detection
# ---------------------------------------------------------------------------
echo "--- Change point (cosine) ---"
${PYTHON} -m interp.bug_trace.analysis.change_point \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    method=cosine

# ---------------------------------------------------------------------------
# DMD: persistent modes in bug-only trajectories
# ---------------------------------------------------------------------------
echo "--- DMD (randomised SVD) ---"
${PYTHON} -m interp.bug_trace.analysis.dmd \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    seed=${SEED}

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
echo "--- Visualize ---"
${PYTHON} -m interp.bug_trace.analysis.visualize \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}"

echo "=== Bug-only analysis complete: results in ${TRAJ_DIR} ==="
