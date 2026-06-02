#!/bin/bash
#SBATCH -J cwm-codi-train
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "thin"
#SBATCH -o logs/codi_train_%j.out
#SBATCH -e logs/codi_train_%j.err

# Teacher-forcing CODI distillation on CRUXEval-O traces.
# Shorter formal run: 200 of 800 samples, latent_steps=6, seq<=4096, 1 epoch.
OUT=/proj/assert-berzelius/users/x_sirli/distill_adapter/fixed_teacher_06021510

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

# Compute nodes are offline: local weights, pre-cached dataset, wandb logs to $OUT/wandb.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR=${OUT}
mkdir -p ${OUT} logs

# device_map=auto shards teacher+student in one process -> plain python, not torchrun.
python -m cwm.training.train \
    model_name_or_path=model_weights/cwm_hf \
    output_dir=${OUT} \
    latent_steps=6 n_samples=200 max_seq_len=4096 epochs=1 \
    wandb_log=true
