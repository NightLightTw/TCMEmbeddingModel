#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
unset VLLM_USE_XFORMERS

swift deploy \
  --ckpt_dir ../output/v0-20250918-104115/checkpoint-2665 \
  --served_model_name Qwen3-Embedding-4B-finetuned-v0-2665 \
  --task_type embedding \
  --infer_backend vllm \
  --torch_dtype float16 \
  --attn_impl sdpa \
  --port 5001
  