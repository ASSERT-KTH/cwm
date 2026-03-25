#!/bin/bash
#
#SBATCH -J cwm-interp-test
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --gpus=0
#SBATCH -o logs/interp_test_%j.out
#SBATCH -e logs/interp_test_%j.err

mkdir -p logs




pytest tests/interp/ -v --tb=short
