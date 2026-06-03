#!/bin/bash
#
#SBATCH -J cwm-table9-awq
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-80GB:1
#SBATCH -A berzelius-2026-167
#SBATCH --exclude=node061
#SBATCH -o logs/table9_awq_%j.out
#SBATCH -e logs/table9_awq_%j.err

mkdir -p logs

# vLLM lives in its own env (torch 2.7.0); the cwm repo is imported via PYTHONPATH
# so we don't need the cwm package installed there.
VLLM_PY=/proj/assert-berzelius/users/x_sirli/conda/envs/vllm/bin/python
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Compute nodes are offline; model + dataset are local/cached.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# True 4-bit AWQ (compressed-tensors) on one GPU, greedy decoding.
"$VLLM_PY" -m evals.trace_analysis.run_eval_vllm \
    --model ./model_weights/cwm-awq-4bit \
    --dump_dir ./eval-cwm-table9-awq-4bit \
    --tp_size 1
