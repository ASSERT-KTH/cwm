#!/bin/bash
#SBATCH -J cwm-codi-qlora
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -C "thin"
#SBATCH -o logs/codi_qlora_%j.out
#SBATCH -e logs/codi_qlora_%j.err

# Task params (single source of truth; reused below for the job name and torchrun).
# 
# max_seq_len and max_batch_tokens caps peak VRAM (build_example drops longer traces; peak tracks the longest row). 
#
# CRUXEval-O 800: min=115 p50=302 p90=988 p99=3649 max=28488.
#   cutoff | coverage     |
#   1024   | 729 (91.1%)  |
#   1280   | 755 (94.4%)  |
#   1536   | 763 (95.4%)  |
#   2048   | 778 (97.2%)  |
#   3072   | 789 (98.6%)  |
LATENT_STEPS=2
N_SAMPLES=16
MAX_SEQ_LEN=1000
MAX_BATCH_TOKENS=1000
EPOCHS=1
BATCH_SIZE=4
GRAD_ACCUM_STEPS=1

# Name encodes the params so squeue / dirs / logs are self-describing.
NAME=codi_qlora_l${LATENT_STEPS}_n${N_SAMPLES}_s${MAX_SEQ_LEN}_b${BATCH_SIZE}x${GRAD_ACCUM_STEPS}_e${EPOCHS}
# #SBATCH -J is static (parsed before the shell runs), so rename the live job.
scontrol update JobId=${SLURM_JOB_ID} JobName=${NAME} 2>/dev/null

OUT=/proj/assert-berzelius/users/x_sirli/distill_adapter/${NAME}_$(date +%m%d%H%M)

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

cd /home/x_sirli/distill_cwm/cwm_andre

# Compute nodes are online: wandb syncs to the cloud directly; logs also kept under $OUT/wandb.
export WANDB_DIR=${OUT}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# on collective timeout, abort comms and crash the rank (don't dead-loop)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p ${OUT} logs

torchrun --standalone --nproc_per_node=4 -m cwm.training.train \
    model_name_or_path=model_weights/cwm_hf \
    output_dir=${OUT} \
    qlora=true \
    latent_steps=${LATENT_STEPS} n_samples=${N_SAMPLES} \
    max_seq_len=${MAX_SEQ_LEN} max_batch_tokens=${MAX_BATCH_TOKENS} epochs=${EPOCHS} \
    batch_size=${BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS} \
    tp_size=1 dp_size=4 \
    flash_attn=true wandb_log=true timing_log_every=1
