#!/bin/bash
#
#SBATCH -J cwm-interp-test
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --gpus=0
#SBATCH -o logs/interp_test_%j.out
#SBATCH -e logs/interp_test_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

pytest tests/interp/ -v --tb=short
