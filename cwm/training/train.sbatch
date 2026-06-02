#!/bin/bash
#SBATCH -J cwm-codi-train
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "thin"
#SBATCH -o logs/codi_train_%j.out
#SBATCH -e logs/codi_train_%j.err

# Teacher-forcing CODI distillation on CRUXEval-O traces.
# Pure tensor parallelism: 8 ranks form one TP group and process the same batch.
# This replaces device_map=auto layer sharding, which behaves like pipeline-style
# serial execution for batch_size=1 and leaves most GPUs idle.
OUT=/proj/assert-berzelius/users/x_sirli/distill_adapter/codi_$(date +%m%d%H%M)

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

cd /home/x_sirli/distill_cwm/cwm_andre

# Compute nodes are offline: local weights, pre-cached dataset, wandb logs to $OUT/wandb.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR=${OUT}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Distributed debug: if a subset of ranks dies and the rest hang in NCCL (the
# "8 -> 3 procs, rank0 at 100%" symptom), these surface the root cause in the
# logs instead of a silent stall. No collective timeout is set, so a genuine
# hang still waits indefinitely; cancel manually if needed.
export NCCL_DEBUG=WARN                      # print NCCL errors / abort reasons
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # raise+exit on NCCL error, don't hang silently
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # report collective shape/arg mismatches
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 # flight recorder: dump where each rank is stuck

mkdir -p ${OUT} logs

# torchrun sets WORLD_SIZE/RANK/LOCAL_RANK. With tp_size=8, dp_size=1, every
# rank consumes the same batch and participates in each tensor-parallel layer.
torchrun --standalone --nproc_per_node=8 -m cwm.training.train \
    model_name_or_path=model_weights/cwm_hf \
    output_dir=${OUT} \
    latent_steps=2 n_samples=-1 max_seq_len=2048 epochs=1 \
    batch_size=4 grad_accum_steps=1 \
    device_map=none tp_size=8 dp_size=1 pp_size=1 tp_plan=auto \
    flash_attn=true wandb_log=true
