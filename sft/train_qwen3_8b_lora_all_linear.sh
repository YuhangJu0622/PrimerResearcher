#!/usr/bin/env bash
# Single-GPU (96 GB) Qwen3-8B DeepResearch SFT with all-linear LoRA.
# Usage (after the GPU is assigned):
#   WANDB_API_KEY=... bash sft/shell/train_qwen3_8b_lora_all_linear.sh
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# W&B can also authenticate through a prior `wandb login`.
export WANDB_PROJECT="${WANDB_PROJECT:-qqr-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-8b-deepresearch-all-linear-lora-r16}"

PROJECT_DIR="/root/Project/qqr"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-8B}"
TRAIN_FILE="${PROJECT_DIR}/sft/data/ms_swift_train.jsonl"
VAL_FILE="${PROJECT_DIR}/sft/data/ms_swift_val.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/output/qqr}"

[[ -f "${TRAIN_FILE}" ]] || { echo "Missing training data: ${TRAIN_FILE}" >&2; exit 1; }
[[ -f "${VAL_FILE}" ]] || { echo "Missing validation data: ${VAL_FILE}" >&2; exit 1; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "Missing model: ${MODEL_DIR}" >&2; exit 1; }
command -v swift >/dev/null || { echo "MS-Swift command not found" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA driver not found" >&2; exit 1; }

echo "Visible GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "W&B: project=${WANDB_PROJECT}, mode=${WANDB_MODE}, run=${WANDB_RUN_NAME}"

# `packing=true` activates padding-free mode in MS-Swift and requires FlashAttention.
# 32,768 is Qwen3-8B's max_position_embeddings and keeps every current sample intact.
swift sft \
  --model "${MODEL_DIR}" \
  --template qwen3 \
  --template_backend swift \
  --tuner_type lora \
  --target_modules all-linear \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --dataset "${TRAIN_FILE}" \
  --val_dataset "${VAL_FILE}" \
  --dataset_num_proc 4 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --use_liger_kernel true \
  --max_length 32768 \
  --packing true \
  --lazy_tokenize false \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 3 \
  --num_train_epochs 3 \
  --learning_rate 3e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --loss_scale default+ignore_empty_think \
  --eval_strategy steps \
  --eval_steps 20 \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 20 \
  --logging_steps 1 \
  --report_to wandb \
  --run_name "${WANDB_RUN_NAME}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}"
