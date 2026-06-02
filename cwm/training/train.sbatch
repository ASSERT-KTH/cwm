#!/bin/bash
#
#SBATCH -J cwm-codi-train
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/codi_train_%j.out
#SBATCH -e logs/codi_train_%j.err

# Teacher-forcing CODI distillation on CRUXEval-O execution traces.
# Override on submit, e.g.:
#   sbatch --export=ALL,MODEL=facebook/cwm,LATENT_STEPS=6 cwm/training/train.sbatch

MODEL=${MODEL:-facebook/cwm}
OUT=${OUT:-./codi-cruxeval}
LATENT_STEPS=${LATENT_STEPS:-6}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

# CodiModel shards teacher+student across the node via device_map=auto (single
# process), and the trace dataset is built in-memory, so this is a plain python
# launch -- not torchrun.
python -m cwm.training.train \
    model_name_or_path=${MODEL} \
    output_dir=${OUT} \
    latent_steps=${LATENT_STEPS} \
    wandb_log=true
