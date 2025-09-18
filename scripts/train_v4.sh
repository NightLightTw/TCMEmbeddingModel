#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

export INFONCE_USE_BATCH=True
export INFONCE_HARD_NEGATIVES=5
export INFONCE_MASK_FAKE_NEGATIVE=True

NPROC_PER_NODE=8 \
swift sft \
  --model Qwen/Qwen3-Embedding-4B \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type lora \
  --dataset ../data/train_with_hard_negatives.jsonl \
  --val_dataset ../data/dev_with_hard_negatives.jsonl \
  --output_dir ../output \
  --eval_strategy steps --eval_steps 100 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2\
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6e-6 \
  --loss_type infonce \
  --dataloader_num_workers 16 \
  --dataloader_prefetch_factor 4 \
  --dataloader_drop_last true \
  --dataloader_persistent_workers true \
  --dataloader_pin_memory true \
  --max_length 1024 \
  --attn_impl sdpa \
  --fp16 true --bf16 false \
  --use_hf true \
  --seed 42 \
  --gradient_checkpointing true \
  --deepspeed zero3