#!/bin/bash
#
#SBATCH -J cwm-extract
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/extract_%j.out
#SBATCH -e logs/extract_%j.err

MODE=${MODE:-trace_full}
N_SAMPLES=${N_SAMPLES:-800}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

python -m torch.distributed.run --nproc_per_node=8 \
    -m interp.extract.run_extract \
    checkpoint_dir=./model_weights/cwm \
    dump_dir=./interp-extract-${MODE} \
    mode=${MODE} \
    n_samples=${N_SAMPLES} \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0
