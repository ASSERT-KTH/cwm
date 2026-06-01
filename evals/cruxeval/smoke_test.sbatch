#!/bin/bash
#SBATCH -J cwm-smoke
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100fat:2
#SBATCH -A naiss2026-3-261
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err

mkdir -p logs

module load Miniforge3/24.1.2-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mimer/NOBACKUP/groups/assert-kth/siruil/venv_cwm

python -m torch.distributed.run --nproc_per_node=2 \
    -m evals.cruxeval.run_eval \
    checkpoint_dir=./model_weights/cwm \
    gen_args.tp_size=2 \
    n_samples=2 \
    n_generations=1 \
    dump_dir=./eval-smoke \
    mode=direct
