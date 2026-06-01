#!/bin/bash
#
#SBATCH -J cwm-table9
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-80GB:8
#SBATCH -A berzelius-2026-167
#SBATCH --exclude=node061
#SBATCH -o logs/table9_%j.out
#SBATCH -e logs/table9_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

export MOODIST_LOG_LEVEL=1

# Table 9 uses greedy decoding (gen_args.use_sampling defaults to False).
python -m torch.distributed.run --nproc_per_node=8 \
    -m evals.trace_analysis.run_eval \
    checkpoint_dir=./model_weights/cwm \
    gen_args.tp_size=2 \
    dump_dir=./eval-cwm-table9
