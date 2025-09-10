#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=1

swift deploy \
  --ckpt_dir ../output/v3-20250908-085903/checkpoint-3375 \
  --served_model_name Qwen3-Embedding-0.6B-finetuned \
  --task_type embedding \
  --infer_backend vllm \
  --torch_dtype float16