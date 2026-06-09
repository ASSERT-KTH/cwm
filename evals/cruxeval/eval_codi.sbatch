#!/bin/bash
#
#SBATCH -J cwm-codi-cruxeval-thin
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-40GB:8
#SBATCH -A berzelius-2026-167
#SBATCH -o logs/codi_cruxeval_thin_%j.out
#SBATCH -e logs/codi_cruxeval_thin_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

# run_eval_codi.py arguments.
ADAPTER_DIR=./codi-training-output
BASE_MODEL=./model_weights/cwm_hf
LATENT_STEPS=2
DUMP_DIR=./eval-codi-cruxeval
MODE=trace_full
N_SAMPLES=-1
N_GENERATIONS=10
TEMPERATURE=0.6
TOP_P=0.95
SEED=42
DATA_SPLIT=val
MAX_GEN=0
BATCH_SIZE=8
DIST_TIMEOUT_MINUTES=360
DEVICE_MAP=auto
LOAD_4BIT=1

# Launcher controls.
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}

EXTRA_ARGS=()
if [[ "${LOAD_4BIT}" == "1" ]]; then
    EXTRA_ARGS+=(--load_4bit)
fi
echo "launching ${NPROC}-way data-parallel eval"

python -m torch.distributed.run --nproc_per_node=${NPROC} \
    -m evals.cruxeval.run_eval_codi \
    --adapter_dir "${ADAPTER_DIR}" \
    --base_model "${BASE_MODEL}" \
    --latent_steps "${LATENT_STEPS}" \
    --dump_dir "${DUMP_DIR}" \
    --mode "${MODE}" \
    --n_samples "${N_SAMPLES}" \
    --n_generations "${N_GENERATIONS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
    --data_split "${DATA_SPLIT}" \
    --max_gen "${MAX_GEN}" \
    --batch_size "${BATCH_SIZE}" \
    --dist_timeout_minutes "${DIST_TIMEOUT_MINUTES}" \
    --device_map "${DEVICE_MAP}" \
    "${EXTRA_ARGS[@]}"
