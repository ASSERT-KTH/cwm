#!/bin/bash
#
# Tier 1+2+3 analysis WITHOUT logit lens (which is compute-prohibitive on CPU
# due to 3.15 GB output_weight cache thrashing — 35s/sample × 1280 = 12.5 hours).
# Runs: probes → PCA → CCS → change point → RSA → DMD → visualize → decide.
#
#SBATCH -J cwm-bug-analysis-noll
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_analysis_noll_%j.out
#SBATCH -e logs/bug_analysis_noll_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
TRACK=${TRACK:-track_a}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
SEED=${SEED:-42}
LAYERS="[16, 32, 48, 63]"

echo "=== Analysis (no logit lens): traj_dir=${TRAJ_DIR} ==="

if [ ! -s "${TRAJ_DIR}/index.jsonl" ]; then
    echo "ERROR: ${TRAJ_DIR}/index.jsonl missing."
    exit 1
fi

# ---------------------------------------------------------------------------
# Tier 1: Probes (most important — skip logit lens)
# ---------------------------------------------------------------------------
echo "--- Tier 1: Probe Temporal (is_buggy) ---"
${PYTHON} -m interp.bug_trace.analysis.probe_temporal \
    traj_dir=${TRAJ_DIR} \
    target=is_buggy \
    "layers=${LAYERS}" \
    seed=${SEED}

echo "--- Tier 1: Probe Temporal (will_be_correct) ---"
${PYTHON} -m interp.bug_trace.analysis.probe_temporal \
    traj_dir=${TRAJ_DIR} \
    target=will_be_correct \
    "layers=${LAYERS}" \
    seed=${SEED}

# ---------------------------------------------------------------------------
# Tier 2: PCA, CCS, change point
# ---------------------------------------------------------------------------
echo "--- Tier 2: PCA Trajectory ---"
${PYTHON} -m interp.bug_trace.analysis.pca_trajectory \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    seed=${SEED}

echo "--- Tier 2: CCS (mean) ---"
${PYTHON} -m interp.bug_trace.analysis.ccs \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    time_bin=mean \
    seed=${SEED}

echo "--- Tier 2: CCS (last_quarter) ---"
${PYTHON} -m interp.bug_trace.analysis.ccs \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    time_bin=last_quarter \
    seed=${SEED}

echo "--- Tier 2: Change Point (cosine) ---"
${PYTHON} -m interp.bug_trace.analysis.change_point \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    method=cosine

echo "--- Tier 2: Change Point (l2) ---"
${PYTHON} -m interp.bug_trace.analysis.change_point \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    method=l2

# ---------------------------------------------------------------------------
# Tier 3: RSA, DMD
# ---------------------------------------------------------------------------
echo "--- Tier 3: RSA ---"
${PYTHON} -m interp.bug_trace.analysis.rsa \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    seed=${SEED}

echo "--- Tier 3: DMD ---"
${PYTHON} -m interp.bug_trace.analysis.dmd \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}"

# ---------------------------------------------------------------------------
# Visualise (matplotlib not available — will skip gracefully)
# ---------------------------------------------------------------------------
echo "--- Visualise ---"
${PYTHON} -m interp.bug_trace.analysis.visualize \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}"

# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------
echo "--- Decision Engine ---"
${PYTHON} -m interp.bug_trace.decide_next \
    traj_dir=${TRAJ_DIR} \
    current_tier=3

# ---------------------------------------------------------------------------
# Fill report
# ---------------------------------------------------------------------------
echo "--- Fill Report ---"
${PYTHON} -m interp.bug_trace.fill_report \
    traj_dir=${TRAJ_DIR} \
    report_path=experiments/02_bug_trace/REPORT.md

echo "=== Analysis complete (no logit lens): results in ${TRAJ_DIR} ==="
