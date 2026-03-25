#!/bin/bash
#SBATCH -J cwm-intervene
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH -C "fat"
#SBATCH -o logs/intervene_%j.out
#SBATCH -e logs/intervene_%j.err
set -euo pipefail

mkdir -p logs

CKPT=./model_weights/cwm
PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
N_SAMPLES=${N_SAMPLES:-200}

# ── Intervention type ─────────────────────────────────────────────────────────
# Set INTERVENTION to one of: variable | return | branch | corrupt
# Adjust type-specific params below.
INTERVENTION=${INTERVENTION:-variable}

# variable: which variable to target and what to replace it with
VARNAME=${VARNAME:-n}
NEW_VALUE=${NEW_VALUE:-999}

# return: replace the inner function return value with this
RETURN_VALUE=${RETURN_VALUE:-0}

echo "=== Token intervention: ${INTERVENTION} ==="

if [ "${INTERVENTION}" = "variable" ]; then
    DUMP_DIR=./interp-intervene-variable-${VARNAME}
    EXTRA="varname=${VARNAME} new_value=${NEW_VALUE}"

elif [ "${INTERVENTION}" = "return" ]; then
    DUMP_DIR=./interp-intervene-return
    EXTRA="new_value=${RETURN_VALUE} which_return=inner"

elif [ "${INTERVENTION}" = "branch" ]; then
    DUMP_DIR=./interp-intervene-branch
    EXTRA=""

elif [ "${INTERVENTION}" = "corrupt" ]; then
    DUMP_DIR=./interp-intervene-corrupt
    EXTRA="truncate_corrupt=True"

else
    echo "ERROR: unknown INTERVENTION=${INTERVENTION}"
    exit 1
fi

# Use a port derived from job ID to avoid collisions when multiple jobs share a node
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

$PYTHON -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} \
    -m interp.token_intervention.run_intervention \
    checkpoint_dir=${CKPT} \
    dump_dir=${DUMP_DIR} \
    intervention=${INTERVENTION} \
    n_samples=${N_SAMPLES} \
    ${EXTRA} \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0

echo "=== Done: results in ${DUMP_DIR}/summary.json ==="
