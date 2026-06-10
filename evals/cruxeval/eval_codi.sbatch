#!/bin/bash
#
# Evaluate completed CODI QLoRA runs 904/905/906 on all CRUXEval CODI modes.
#
# Task mapping:
#   0..3   -> run 904, direct/reasoning/trace_full/trace_single_step
#   4..7   -> run 905, direct/reasoning/trace_full/trace_single_step
#   8..11  -> run 906, direct/reasoning/trace_full/trace_single_step
#
# Limit to four concurrent tasks so each wave matches the 907 sweep footprint.
#SBATCH -J codi904-906-cruxeval
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-40GB:8
#SBATCH -A berzelius-2026-167
#SBATCH --array=0-11%4
#SBATCH -o logs/codi904_906_cruxeval_%A_%a.out
#SBATCH -e logs/codi904_906_cruxeval_%A_%a.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

MODES=(direct reasoning trace_full trace_single_step)
MAX_GENS=(512 3072 3072 512)
RUN_IDS=(904 905 906)
ADAPTER_DIRS=(
    /proj/assert-berzelius/users/x_sirli/distill_adapter/codi_qlora_l1_n0_s1536_b32x1_lmw1.0_kdw1.0_e5_06081405
    /proj/assert-berzelius/users/x_sirli/distill_adapter/codi_qlora_l1_n0_s1536_b32x1_lmw1.0_kdw2.0_e5_06081405
    /proj/assert-berzelius/users/x_sirli/distill_adapter/codi_qlora_l1_n0_s1536_b32x1_lmw2.0_kdw1.0_e5_06081405
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
DATA_SPLIT=val
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
    --data_split "${DATA_SPLIT}" \
    --max_gen "${MAX_GEN}" \
    --batch_size "${BATCH_SIZE}" \
    --dist_timeout_minutes "${DIST_TIMEOUT_MINUTES}" \
    --device_map "${DEVICE_MAP}" \
    "${EXTRA_ARGS[@]}"
