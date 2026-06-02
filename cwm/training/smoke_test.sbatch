#!/bin/bash
#SBATCH -J cwm-smoke
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err

OUT=/proj/assert-berzelius/users/x_sirli/distill_adapter/smoke_$(date +%m%d%H%M)

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

cd /home/x_sirli/distill_cwm/cwm_andre

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
mkdir -p ${OUT} logs

# n_samples=16 -> 8 samples/rank; grad_accum_steps=8 keeps global batch 16.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
GPU_PAIRS=("0,1,2,3" "4,5,6,7")
export WORLD_SIZE=${#GPU_PAIRS[@]}

pids=()

launch_rank() {
    local rank=$1
    local visible_gpus=$2
    CUDA_VISIBLE_DEVICES=${visible_gpus} \
    RANK=${rank} LOCAL_RANK=0 WORLD_SIZE=${WORLD_SIZE} \
    MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
    python -m cwm.training.train \
        model_name_or_path=model_weights/cwm_hf \
        output_dir=${OUT} \
        latent_steps=3 n_samples=16 max_seq_len=512 \
        batch_size=1 grad_accum_steps=8 epochs=1 \
        device_map=auto flash_attn=true \
        wandb_log=false &
    pids+=("$!")
}

for rank in "${!GPU_PAIRS[@]}"; do
    launch_rank "${rank}" "${GPU_PAIRS[${rank}]}"
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
done
exit ${status}
