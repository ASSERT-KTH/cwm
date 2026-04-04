#!/bin/bash
#
# Iteration 2 CPU analysis chain:
#   1. mutation_type probe (probe_content)
#   2. CCS with 70/30 train/test split (saves ccs_split.json)
#   3. T* text correlation (t_star_text)
#   4. DMD (fixed: one layer at a time)
#   5. Visualize (all figures including mutation_type PCA)
#
#SBATCH -J cwm-bug-iter2-cpu
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_iter2_cpu_%j.out
#SBATCH -e logs/bug_iter2_cpu_%j.err

set -euo pipefail
mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
TRACK=${TRACK:-track_a}
TRAJ_DIR=${TRAJ_DIR:-./interp-bug-trajectories-${TRACK}}
CKPT=${CKPT:-./model_weights/cwm}
SEED=${SEED:-42}
LAYERS="[16, 32, 48, 63]"

echo "=== Iteration 2 CPU analysis: traj_dir=${TRAJ_DIR} ==="
date

if [ ! -s "${TRAJ_DIR}/index.jsonl" ]; then
    echo "ERROR: ${TRAJ_DIR}/index.jsonl missing. Run bug_extract.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# M1: mutation_type probe (5-class, only on buggy samples)
# ---------------------------------------------------------------------------
echo "--- M1: Probe Content (mutation_type, 5-class) ---"
${PYTHON} -m interp.bug_trace.analysis.probe_content \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    n_time_bins=10 \
    epochs=30 \
    n_perm=5 \
    seed=${SEED}

# ---------------------------------------------------------------------------
# M2: CCS with 70/30 train/test split (creates ccs_split.json)
# ---------------------------------------------------------------------------
echo "--- M2: CCS (mean, train/test split) ---"
${PYTHON} -m interp.bug_trace.analysis.ccs \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    time_bin=mean \
    test_fraction=0.3 \
    seed=${SEED}

echo "--- M2: CCS (last_quarter, train/test split) ---"
${PYTHON} -m interp.bug_trace.analysis.ccs \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    time_bin=last_quarter \
    test_fraction=0.3 \
    seed=${SEED}

# ---------------------------------------------------------------------------
# A1: T* text correlation (per-sample + aggregate figures)
# ---------------------------------------------------------------------------
echo "--- A1: T* Text Correlation ---"
${PYTHON} -m interp.bug_trace.analysis.t_star_text \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    checkpoint_dir=${CKPT} \
    n_sample_figures=100 \
    seed=${SEED}

# ---------------------------------------------------------------------------
# A3: DMD (fixed: one layer at a time, no OOM)
# ---------------------------------------------------------------------------
echo "--- A3: DMD (per-layer, memory-safe) ---"
${PYTHON} -m interp.bug_trace.analysis.dmd \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}" \
    seed=${SEED}

# ---------------------------------------------------------------------------
# Visualize: all figures including mutation_type PCA + probe_content heatmap
# ---------------------------------------------------------------------------
echo "--- Visualize ---"
${PYTHON} -m interp.bug_trace.analysis.visualize \
    traj_dir=${TRAJ_DIR} \
    "layers=${LAYERS}"

echo "=== Iteration 2 CPU analysis complete: ${TRAJ_DIR} ==="
date
