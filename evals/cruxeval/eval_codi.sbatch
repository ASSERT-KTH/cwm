#!/bin/bash
#SBATCH -J codi_cruxeval_val
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -C "thin"
#SBATCH -A berzelius-2026-167
#SBATCH --array=0-3%4
#SBATCH -o logs/codi_cruxeval_val_%A_%a.out
#SBATCH -e logs/codi_cruxeval_val_%A_%a.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

MODES=(direct reasoning trace_full trace_single_step)
MAX_GENS=(512 3072 3072 512)
RUN_IDS=(16807668)
ADAPTER_DIRS=(
    /proj/assert-berzelius/users/x_sirli/distill_adapter/codi_qlora_l1_n0_s1280_b32x1_tw1.0_lmw1.0_kdw1.0_e5_06101018
)

RUN_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
MODE_INDEX=$((SLURM_ARRAY_TASK_ID % 4))

RUN_ID=${RUN_IDS[$RUN_INDEX]}
MODE=${MODES[$MODE_INDEX]}
MAX_GEN=${MAX_GENS[$MODE_INDEX]}
ADAPTER_DIR=${ADAPTER_DIRS[$RUN_INDEX]}

BASE_MODEL=./model_weights/cwm_hf
LATENT_STEPS=1
DUMP_ROOT=./eval-codi-cruxeval-${RUN_ID}
DUMP_DIR=${DUMP_ROOT}/${MODE}
N_SAMPLES=-1
N_GENERATIONS=10
TEMPERATURE=0.6
TOP_P=0.95
SEED=42
DATA_SOURCE=cruxeval_o
BATCH_SIZE=8
DIST_TIMEOUT_MINUTES=360
DEVICE_MAP=auto
LOAD_4BIT=1

NPROC=$(nvidia-smi -L | wc -l)

EXTRA_ARGS=()
if [[ "${LOAD_4BIT}" == "1" ]]; then
    EXTRA_ARGS+=(--load_4bit)
fi

echo "run_id=${RUN_ID}"
echo "adapter=${ADAPTER_DIR}"
echo "mode=${MODE}"
echo "dump_dir=${DUMP_DIR}"
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
    --data_source "${DATA_SOURCE}" \
    --max_gen "${MAX_GEN}" \
    --batch_size "${BATCH_SIZE}" \
    --dist_timeout_minutes "${DIST_TIMEOUT_MINUTES}" \
    --device_map "${DEVICE_MAP}" \
    "${EXTRA_ARGS[@]}"
