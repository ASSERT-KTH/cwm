#!/bin/bash
#
# Phases 1-3 of the interp pipeline — CPU only (no GPU required).
#   Phase 1: rebuild index.jsonl from existing .pt files (or run extraction if missing)
#   Phase 2: logit lens
#   Phase 3: probe training
#
# Run this first, then submit pipeline_2gpu.sh for the steering sweep.
#
#SBATCH -J cwm-interp-cpu
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p berzelius-cpu
#SBATCH -o logs/phases123_%j.out
#SBATCH -e logs/phases123_%j.err

set -euo pipefail
mkdir -p logs

EXTRACT_DIR=./interp-extract-trace_full
CKPT=./model_weights/cwm
PYTHON=/proj/assert-berzelius/users/x_andaf/.conda/envs/CWM/bin/python

# ---------------------------------------------------------------------------
# Phase 1: rebuild index.jsonl from existing .pt files
# (If fewer than 800 files exist, abort — run extraction on a GPU node first)
# ---------------------------------------------------------------------------
echo "=== Phase 1: Index rebuild ==="
N_EXISTING=$(ls ${EXTRACT_DIR}/activations/*.pt 2>/dev/null | wc -l)
if [ "${N_EXISTING}" -lt 800 ]; then
    echo "ERROR: Only ${N_EXISTING}/800 activation files found."
    echo "Run the extraction step on a GPU node first."
    exit 1
fi

$PYTHON - << 'PYEOF'
import json, torch
from pathlib import Path
d = Path("interp-extract-trace_full")
entries = []
for pt in sorted((d/"activations").glob("*.pt")):
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
        entries.append({k: v for k, v in data.items() if k != "activations"})
    except Exception as e:
        print(f"WARNING: {pt.name}: {e}")
with (d/"index.jsonl").open("w") as f:
    [f.write(json.dumps(e, default=str)+"\n") for e in entries]
n_c = sum(e.get("correct", False) for e in entries)
print(f"index.jsonl rebuilt: {len(entries)} samples, pass@1={n_c/len(entries):.4f}")
PYEOF

# ---------------------------------------------------------------------------
# Phase 2: Logit lens (CPU fallback via torch.cuda.is_available() == False)
# ---------------------------------------------------------------------------
echo "=== Phase 2: Logit lens ==="
$PYTHON -m interp.logit_lens.run_logit_lens \
    extract_dir=${EXTRACT_DIR} \
    checkpoint_dir=${CKPT}

# ---------------------------------------------------------------------------
# Phase 3: Probe training (CPU fallback)
# ---------------------------------------------------------------------------
echo "=== Phase 3: Probe training ==="
for prop in will_be_correct return_type return_sign return_truthy; do
    for probe in linear mlp1; do
        echo "--- $probe / $prop ---"
        $PYTHON -m interp.probes.train_probe \
            extract_dir=${EXTRACT_DIR} \
            target_property=$prop \
            probe_type=$probe
    done
done

echo "=== Phases 1-3 complete. Submit pipeline_2gpu.sh for steering. ==="
