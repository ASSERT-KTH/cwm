#!/bin/bash
#
#SBATCH -J cwm-probes
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH -o logs/probes_%j.out
#SBATCH -e logs/probes_%j.err

EXTRACT_DIR=${EXTRACT_DIR:-./interp-extract-trace_full}

mkdir -p logs




# Train all property × probe_type combinations
for prop in will_be_correct return_type return_sign return_truthy; do
    for probe in linear mlp1; do
        echo "=== Training $probe probe for $prop ==="
        python -m interp.probes.train_probe \
            extract_dir=${EXTRACT_DIR} \
            target_property=$prop \
            probe_type=$probe
    done
done
