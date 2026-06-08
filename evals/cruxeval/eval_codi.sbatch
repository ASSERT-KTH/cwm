#!/bin/bash
#
#SBATCH -J cwm-codi-cruxeval
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-80GB:8
#SBATCH -A berzelius-2026-167
#SBATCH --exclude=node061
#SBATCH -o logs/codi_cruxeval_%j.out
#SBATCH -e logs/codi_cruxeval_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

# CRUXEval-O output prediction (pass@1) for a CODI-distilled model:
# bf16 base + CODI LoRA adapter + thought_projector, greedy trace_full decoding.
# Override ADAPTER_DIR with the training output_dir.
ADAPTER_DIR=${ADAPTER_DIR:?set ADAPTER_DIR to the CODI training output_dir}
LATENT_STEPS=${LATENT_STEPS:-2}
DATA_SPLIT=${DATA_SPLIT:-val}        # sweep evaluates the held-out split
DUMP_DIR=${DUMP_DIR:-./eval-codi-cruxeval}  # per-config dir; sweep gives each config its own

python -m torch.distributed.run --nproc_per_node=8 \
    -m evals.cruxeval.run_eval_codi \
    --adapter_dir "${ADAPTER_DIR}" \
    --base_model ./model_weights/cwm_hf \
    --latent_steps "${LATENT_STEPS}" \
    --data_split "${DATA_SPLIT}" \
    --dump_dir "${DUMP_DIR}"
