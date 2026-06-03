#!/bin/bash
#SBATCH -J cwm-table9-awq-smoke
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=A100-SXM4-80GB:1
#SBATCH -A berzelius-2026-167
#SBATCH --exclude=node061
#SBATCH -o logs/table9_awq_smoke_%j.out
#SBATCH -e logs/table9_awq_smoke_%j.err

mkdir -p logs

VLLM_PY=/proj/assert-berzelius/users/x_sirli/conda/envs/vllm/bin/python
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"$VLLM_PY" -m evals.trace_analysis.run_eval_vllm \
    --model ./model_weights/cwm-awq-4bit \
    --n_samples 8 \
    --dump_dir ./eval-table9-awq-4bit-smoke \
    --tp_size 1
