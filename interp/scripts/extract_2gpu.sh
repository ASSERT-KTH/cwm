#!/bin/bash
#
#SBATCH -J cwm-extract
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH -C "fat"
#SBATCH -o logs/extract_%j.out
#SBATCH -e logs/extract_%j.err

MODE=${MODE:-trace_full}
N_SAMPLES=${N_SAMPLES:-800}
DUMP_DIR=${DUMP_DIR:-./interp-extract-${MODE}}

mkdir -p logs

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

$PYTHON -m torch.distributed.run --nproc_per_node=2 \
    -m interp.extract.run_extract \
    checkpoint_dir=./model_weights/cwm \
    dump_dir=${DUMP_DIR} \
    mode=${MODE} \
    n_samples=${N_SAMPLES} \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0
