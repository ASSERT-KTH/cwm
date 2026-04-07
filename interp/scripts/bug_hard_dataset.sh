#!/bin/bash
#
# Build hard bug-injection dataset from CRUXEval (CPU only, ~5 min).
# Uses only hard mutation types: wrong_variable, deleted_accumulator, swapped_arguments.
#
#SBATCH -J cwm-bug-hard-dataset
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p berzelius-cpu
#SBATCH -o logs/bug_hard_dataset_%j.out
#SBATCH -e logs/bug_hard_dataset_%j.err

set -euo pipefail
mkdir -p logs interp/bug_trace/data

PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python
SEED=${SEED:-42}

echo "=== Building hard bug-injection dataset (seed=${SEED}) ==="
${PYTHON} -m interp.bug_trace.mutate \
    --output_path interp/bug_trace/data/pairs_hard.json \
    --n_samples 800 \
    --max_mutations_per_sample 3 \
    --seed ${SEED} \
    --hard

echo "=== Dataset ready: interp/bug_trace/data/pairs_hard.json ==="
${PYTHON} -c "
import json
pairs = json.load(open('interp/bug_trace/data/pairs_hard.json'))
from collections import Counter
counts = Counter(p['mutation_type'] for p in pairs)
print(f'Total pairs: {len(pairs)}')
for k, v in sorted(counts.items()):
    print(f'  {k}: {v}')
"
