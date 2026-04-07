#!/bin/bash
#
# Analysis pipeline for hard-mutation trajectories.
# Runs: probe_content (mutation_type, 3-class) + t_star_text + CCS + visualize + compare_profiles.
# Depends on: bug_hard_extract.sh having completed (DUMP_DIR must exist with index.jsonl).
#
#SBATCH -J cwm-bug-hard-analysis
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_hard_analysis_%j.out
#SBATCH -e logs/bug_hard_analysis_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
TRAJ_HARD=${TRAJ_HARD:-./interp-bug-trajectories-hard}
TRAJ_EASY=${TRAJ_EASY:-./interp-bug-trajectories-track_a}
SEED=${SEED:-42}
LAYERS="[16, 32, 48, 63]"

echo "=== Hard-mutation analysis: traj_dir=${TRAJ_HARD} ==="

# Sanity check
if [ ! -s "${TRAJ_HARD}/index.jsonl" ]; then
    echo "ERROR: ${TRAJ_HARD}/index.jsonl missing. Run bug_hard_extract.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Probe: mutation_type classification (3-class for hard mutations)
# ---------------------------------------------------------------------------
echo "--- Probe: mutation_type (hard, 3-class) ---"
${PYTHON} -m interp.bug_trace.analysis.probe_content \
    traj_dir=${TRAJ_HARD} \
    "layers=${LAYERS}" \
    seed=${SEED}

# ---------------------------------------------------------------------------
# T* text correlation
# ---------------------------------------------------------------------------
echo "--- T* text correlation ---"
${PYTHON} -m interp.bug_trace.analysis.t_star_text \
    traj_dir=${TRAJ_HARD} \
    "layers=${LAYERS}"

# ---------------------------------------------------------------------------
# CCS (train/test split)
# ---------------------------------------------------------------------------
echo "--- CCS (mean representation) ---"
${PYTHON} -m interp.bug_trace.analysis.ccs \
    traj_dir=${TRAJ_HARD} \
    "layers=${LAYERS}" \
    time_bin=mean \
    seed=${SEED}

echo "--- CCS (last_quarter) ---"
${PYTHON} -m interp.bug_trace.analysis.ccs \
    traj_dir=${TRAJ_HARD} \
    "layers=${LAYERS}" \
    time_bin=last_quarter \
    seed=${SEED}

# ---------------------------------------------------------------------------
# DMD
# ---------------------------------------------------------------------------
echo "--- DMD (randomised SVD) ---"
${PYTHON} -m interp.bug_trace.analysis.dmd \
    traj_dir=${TRAJ_HARD} \
    "layers=${LAYERS}" \
    seed=${SEED}

# ---------------------------------------------------------------------------
# Visualize hard trajectories
# ---------------------------------------------------------------------------
echo "--- Visualize (hard) ---"
${PYTHON} -m interp.bug_trace.analysis.visualize \
    traj_dir=${TRAJ_HARD} \
    "layers=${LAYERS}"

# ---------------------------------------------------------------------------
# Cross-experiment comparison: easy vs hard temporal profiles
# ---------------------------------------------------------------------------
echo "--- Compare profiles: easy vs hard ---"
if [ -s "${TRAJ_EASY}/probe_content_mutation_type.pt" ]; then
    ${PYTHON} -m interp.bug_trace.analysis.compare_profiles \
        easy_traj_dir=${TRAJ_EASY} \
        hard_traj_dir=${TRAJ_HARD} \
        "layers=${LAYERS}"
else
    echo "  SKIP compare_profiles: ${TRAJ_EASY}/probe_content_mutation_type.pt not found"
fi

echo "=== Hard analysis complete: results in ${TRAJ_HARD} ==="
