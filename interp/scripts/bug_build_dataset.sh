#!/bin/bash
#
# Build bug-injection dataset from CRUXEval (CPU only, ~5 min).
#
#SBATCH -J cwm-bug-dataset
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_dataset_%j.out
#SBATCH -e logs/bug_dataset_%j.err

set -euo pipefail
mkdir -p logs interp/bug_trace/data

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
SEED=${SEED:-42}

echo "=== Building bug-injection dataset (seed=${SEED}) ==="
${PYTHON} -m interp.bug_trace.mutate \
    --output_path interp/bug_trace/data/pairs.json \
    --n_samples 800 \
    --max_mutations_per_sample 3 \
    --seed ${SEED}

echo "=== Dataset ready: interp/bug_trace/data/pairs.json ==="
wc -l interp/bug_trace/data/pairs.json || true
