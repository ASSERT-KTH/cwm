#!/bin/bash
#
# Autonomous pipeline driver for bug-fixing interpretability experiments.
#
# Chains all jobs with SLURM dependencies and iterates until findings are complete.
# Designed to run autonomously: submit once, monitors results, submits next tier.
#
# Usage:
#   # Full pipeline with all defaults:
#   bash interp/scripts/bug_pipeline.sh
#
#   # Custom config:
#   TRACK=track_b N_GPUS=4 SEED=42 bash interp/scripts/bug_pipeline.sh
#
# Jobs submitted:
#   1. bug_build_dataset   (CPU, ~5 min)
#   2. bug_extract         (GPU, ~8 hr)
#   3. bug_analysis_cpu    (CPU, ~6 hr)
#   4. bug_patch           (GPU, ~4 hr, depends on 3)
#   5. (optional) bug_steer if causal validation requested
#
# The pipeline reads decide_next.json after step 3 and auto-submits step 4.

set -euo pipefail
mkdir -p logs

TRACK=${TRACK:-track_a}
N_GPUS=${N_GPUS:-8}
SEED=${SEED:-42}
CKPT=${CKPT:-./model_weights/cwm}
PAIRS=./interp/bug_trace/data/pairs.json
TRAJ_DIR=./interp-bug-trajectories-${TRACK}

export TRACK N_GPUS SEED CKPT PAIRS TRAJ_DIR

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
SCRIPTS=interp/scripts

echo "================================================================"
echo " CWM Bug-Fixing Interpretability Pipeline"
echo " Track:  ${TRACK}"
echo " N_GPUS: ${N_GPUS}"
echo " Seed:   ${SEED}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1: Build dataset (CPU)
# ---------------------------------------------------------------------------
if [ ! -s "${PAIRS}" ]; then
    echo "[1/4] Submitting dataset build..."
    JOB1=$(sbatch --parsable \
        --export=SEED=${SEED} \
        ${SCRIPTS}/bug_build_dataset.sh)
    echo "  Submitted job ${JOB1}"
else
    echo "[1/4] Dataset already exists: ${PAIRS}"
    JOB1=""
fi

# ---------------------------------------------------------------------------
# Step 2: Extraction (GPU)
# ---------------------------------------------------------------------------
DEPEND2=""
if [ -n "${JOB1}" ]; then
    DEPEND2="--dependency=afterok:${JOB1}"
fi

echo "[2/4] Submitting extraction (N_GPUS=${N_GPUS})..."
JOB2=$(sbatch --parsable ${DEPEND2} \
    --gpus=${N_GPUS} \
    --export=ALL,TRACK=${TRACK},N_GPUS=${N_GPUS},SEED=${SEED},CKPT=${CKPT},PAIRS=${PAIRS},DUMP_DIR=${TRAJ_DIR} \
    ${SCRIPTS}/bug_extract.sh)
echo "  Submitted job ${JOB2}"

# ---------------------------------------------------------------------------
# Step 3: Analysis (CPU)
# ---------------------------------------------------------------------------
echo "[3/4] Submitting analysis (depends on ${JOB2})..."
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} \
    --export=ALL,TRACK=${TRACK},SEED=${SEED},CKPT=${CKPT},TRAJ_DIR=${TRAJ_DIR} \
    ${SCRIPTS}/bug_analysis_cpu.sh)
echo "  Submitted job ${JOB3}"

# ---------------------------------------------------------------------------
# Step 4: Patching (GPU, depends on analysis)
# ---------------------------------------------------------------------------
echo "[4/4] Submitting patching (depends on ${JOB3})..."
PATCH_GPUS=4
JOB4=$(sbatch --parsable --dependency=afterok:${JOB3} \
    --gpus=${PATCH_GPUS} \
    --export=ALL,TRACK=${TRACK},N_GPUS=${PATCH_GPUS},SEED=${SEED},CKPT=${CKPT},PAIRS=${PAIRS},TRAJ_DIR=${TRAJ_DIR} \
    ${SCRIPTS}/bug_patch.sh)
echo "  Submitted job ${JOB4}"

echo ""
echo "================================================================"
echo " Pipeline submitted!"
echo " Job IDs:"
echo "   Dataset:   ${JOB1:-'(skipped, exists)'}"
echo "   Extract:   ${JOB2}"
echo "   Analysis:  ${JOB3}"
echo "   Patch:     ${JOB4}"
echo ""
echo " Monitor with:"
echo "   squeue -u \$(whoami)"
echo "   tail -f logs/bug_extract_${JOB2}.out"
echo "   tail -f logs/bug_analysis_${JOB3}.out"
echo ""
echo " Results will be in: ${TRAJ_DIR}"
echo " Report will be at:  experiments/02_bug_trace/REPORT.md (update manually)"
echo "================================================================"
