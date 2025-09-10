#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=0,2
export NVIDIA_TF32_OVERRIDE=1

export INFONCE_USE_BATCH=True
export INFONCE_HARD_NEGATIVES=5
export INFONCE_MASK_FAKE_NEGATIVE=True

NPROC_PER_NODE=2 \
swift sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type full \
  --dataset ../data/train_with_hard_negatives.jsonl \
  --val_dataset ../data/dev_with_hard_negatives.jsonl \
  --output_dir ../output \
  --eval_strategy steps --eval_steps 100 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6e-6 \
  --loss_type infonce \
  --dataloader_drop_last true \
  --dataloader_num_workers 16 \
  --dataloader_persistent_workers true \
  --attn_impl flash_attn \
  --bf16 true \
  --use_hf true \
  --seed 42