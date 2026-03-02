#!/bin/bash
#
#SBATCH -J cwm-cruxeval
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/cruxeval_%j.out
#SBATCH -e logs/cruxeval_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

python -m torch.distributed.run --nproc_per_node=8 \
    -m evals.cruxeval.run_eval \
    checkpoint_dir=./model_weights/cwm \
    dump_dir=./eval-cwm-cruxeval \
    gen_args.tp_size=2
