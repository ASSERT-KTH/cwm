#!/bin/bash
#SBATCH -J cwm-smoke
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-80GB:4
#SBATCH -A berzelius-2026-167
#SBATCH --exclude=node061
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

export MOODIST_LOG_LEVEL=1

python -m torch.distributed.run --nproc_per_node=4 \
    -m evals.cruxeval.run_eval \
    checkpoint_dir=./model_weights/cwm \
    gen_args.tp_size=2 \
    n_samples=4 \
    n_generations=1 \
    dump_dir=./eval-smoke \
    mode=direct
