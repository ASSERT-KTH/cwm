#!/bin/bash
#
#SBATCH -J cwm-steer
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/steer_%j.out
#SBATCH -e logs/steer_%j.err

VECTOR_DIR=${VECTOR_DIR:?"Set VECTOR_DIR"}
CONDITION=${CONDITION:-correct_vs_incorrect}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

python -m torch.distributed.run --nproc_per_node=8 \
    -m interp.steering.run_steering \
    checkpoint_dir=./model_weights/cwm \
    dump_dir=./interp-steer-${CONDITION} \
    vector_dir=${VECTOR_DIR} \
    condition=${CONDITION} \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0
