#!/usr/bin/env bash
set -euxo pipefail
# Verified runnable on GPU memory of 8x80GB

WANDB_API_KEY=${WANDB_API_KEY:-""}
DATASET_ID=${DATASET_ID:-"demystify-long-cot/math-train-qwq-rs-n32"}
MODEL_NAME=${MODEL_NAME:-"llama-3.1-8b-math-qwq-n32-rft"}

deepspeed --module openrlhf.cli.train_sft \
    --pretrain "meta-llama/Meta-Llama-3.1-8B" \
    --dataset "${DATASET_ID}" --input_key messages --apply_chat_template \
    --max_len $((128 * 1024)) \
    --train_batch_size 256 --micro_train_batch_size 1 \
    --learning_rate "5e-6" --max_epochs 2 \
    --bf16 --zero_stage 2 --flash_attn --gradient_checkpointing \
    --save_path "ckpts/${MODEL_NAME}" \
    --ckpt_path "ckpts/${MODEL_NAME}-ckpts-sft" \
    --save_steps -1 --eval_steps -1 --logging_steps 1 \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "demystify-long-cot" --wandb_run_name "${MODEL_NAME}" 