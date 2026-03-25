#!/bin/bash
#
#SBATCH -J cwm-logit-lens
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH -o logs/logit_lens_%j.out
#SBATCH -e logs/logit_lens_%j.err

EXTRACT_DIR=${EXTRACT_DIR:-./interp-extract-trace_full}

mkdir -p logs




/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python -m interp.logit_lens.run_logit_lens \
    extract_dir=${EXTRACT_DIR} \
    checkpoint_dir=./model_weights/cwm
