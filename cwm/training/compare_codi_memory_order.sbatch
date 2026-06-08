#!/bin/bash
#SBATCH -J codi-mem-order
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -C "fat"
#SBATCH -o logs/codi_mem_order_%j.out
#SBATCH -e logs/codi_mem_order_%j.err

LATENT_STEPS=1
N_SAMPLES=0
MAX_SEQ_LEN=1536
MAX_BATCH_TOKENS=1536
BATCH_SIZE=32
GRAD_ACCUM_STEPS=1
LM_LOSS_WEIGHT=1.0
KD_LOSS_WEIGHT=1.0
NUM_BATCHES=1
SKIP_BATCHES=0

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

cd /home/x_sirli/distill_cwm/cwm_andre
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CRUXEVAL_DIR=/proj/assert-berzelius/users/x_sirli/datasets/cruxeval_test
export HF_DATASETS_OFFLINE=1

COMMON_ARGS=(
    model_name_or_path=model_weights/cwm_hf
    qlora=true
    latent_steps=${LATENT_STEPS}
    n_samples=${N_SAMPLES}
    lm_loss_weight=${LM_LOSS_WEIGHT}
    kd_loss_weight=${KD_LOSS_WEIGHT}
    max_seq_len=${MAX_SEQ_LEN}
    max_batch_tokens=${MAX_BATCH_TOKENS}
    batch_size=${BATCH_SIZE}
    grad_accum_steps=${GRAD_ACCUM_STEPS}
    tp_size=1
    dp_size=4
    flash_attn=true
    num_workers=0
    num_batches=${NUM_BATCHES}
    skip_batches=${SKIP_BATCHES}
    run_backward=true
    log_all_ranks=true
)

for ORDER in student_first teacher_first; do
    echo "===== ${ORDER} $(date) ====="
    torchrun --standalone --nproc_per_node=4 -m cwm.training.compare_codi_memory_order         order=${ORDER}         "${COMMON_ARGS[@]}"         "$@"
    echo "===== ${ORDER} done $(date) ====="
done
