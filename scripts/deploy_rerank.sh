
export CUDA_VISIBLE_DEVICES=1

# there is bug in swift<3.9, so we use vllm==0.10.2 to deploy
# swift deploy \
#   --model Qwen/Qwen3-Reranker-0.6B \
#   --infer_backend vllm \
#   --torch_dtype float16 \
#   --port 8001

# base model
# vllm serve Qwen/Qwen3-Reranker-0.6B \
#    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
#    --port 8001 \
#    --host 0.0.0.0

# finetuned model
vllm serve ../output/reranker/v2-20250925-195405/checkpoint-12000 \
   --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
   --port 8001 \
   --host 0.0.0.0 \
   --served_model_name Qwen/Qwen3-Reranker-0.6B-v2-12000 \
   --max-model-len 8192