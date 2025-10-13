# TCM Embedding Model

## 專案概述

本專案利用 Qwen3-Embedding 和 Qwen3-Reranker 模型，透過 TCM-SD 資料集進行微調，建立專門針對傳統中醫證型診斷的高品質嵌入和重排序模型。

## 目標功能

- 🏥 **中醫領域特化**：針對中醫術語、診斷、方劑等專業內容優化
- 🔍 **語義檢索**：提供精確的中醫文獻和知識檢索能力  
- 🚀 **高效訓練**：基於 ms-swift 框架的高效 fine-tuning 流程
- 😷 **病例對證型微調**：在 TCM-SD 資料集上成功完成 fine-tuning

## 環境需求

### 系統要求
- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) - 現代 Python 包管理工具
- CUDA >= 11.8 (推薦使用 GPU 訓練)
- 記憶體：建議 32GB+ RAM
- 硬碟空間：至少 100GB 可用空間

### 硬體建議
- **訓練環境**：NVIDIA GPU (V100/A6000/H100 或更高)
- **推理環境**：可在 CPU 上運行，GPU 可提升效能
> **備註**: 建議使用 Ampere 架構（A100/L40S/4090）以上的 GPU，因為支援 BF16 與 FlashAttention，可大幅降低記憶體占用並提升速度。

## 安裝指南

### 1. 安裝 uv
如果您還沒有安裝 uv，請先安裝：

```bash
# macOS 和 Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip 安裝
pip install uv
```

### 2. 克隆專案
```bash
git clone https://github.com/NightLightTw/TCMEmbeddingModel.git
cd TCMEmbeddingModel
```

### 3. 使用 uv 安裝環境和依賴
```bash
# uv 會自動創建虛擬環境並安裝依賴
uv sync

# 如果需要開發依賴
uv sync --extra dev
```

### 4. 激活虛擬環境
```bash
# 激活虛擬環境
source .venv/bin/activate  # Linux/Mac
```

### 5. (可選)加速工具
```bash
# Optional packages
uv pip install deepspeed # multi-GPU training
uv pip install liger-kernel # save GPU memory resources
uv pip install flash-attn --no-build-isolation
```

### Optional: Multi-Stage Sampler Extension

本專案有一個自訂分支（`feature/3.8.2-multi-stage-sampler`）提供增強版的多階段取樣機制。

```bash
uv pip install git+https://github.com/NightLightTw/ms-swift.git@feature/3.8.2-multi-stage-sampler
```

> ⚠️ 注意：此分支由 NightLightTw 維護，可能與上游版本不同。僅在需要多階段取樣功能時使用。

## 開發指南

### 使用 uv 進行開發

```bash
# 添加新的依賴
uv add package-name

# 添加開發依賴
uv add --dev package-name

# 移除依賴
uv remove package-name

# 更新依賴
uv sync --upgrade

# 運行腳本
uv run python your_script.py

```

### 資料準備

- 原始資料需自官方 TCM-SD 專案取得：[Borororo/ZY-BERT](https://github.com/Borororo/ZY-BERT)
> **注意**：官方釋出的 `train.json` 等檔案實際為 JSON Lines 格式，請改為 `.jsonl` 副檔名後再使用，避免解析失敗。

轉換工具支持多種數據增強策略，參考 `scripts/load_data.sh` 查看所有選項：

```bash
# 基礎格式（無硬負樣本）
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train.jsonl

# 使用 BM25 硬負樣本
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_with_hard_negatives.jsonl \
    --with-hard-negatives

# 使用自定義 Embedding 硬負樣本
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_with_hard_negatives_custom_embedding.jsonl \
    --with-hard-negatives-custom-embedding

# 混合硬負樣本（2隨機 + 3BM25 + 3Embedding）
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_hybrid.jsonl \
    --hybrid

# 字段組合增強（11種組合）
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_field_combinations.jsonl \
    --field-combinations \
    --with-hard-negatives

# 排列增強（zone-based，5倍數據）
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_permutation_argument_x5.jsonl \
    --permutation-argument \
    --multiplier 5 \
    --with-hard-negatives

# 限制樣本數量（測試用）
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_sample.jsonl \
    --max-samples 100
```

### 開始訓練

本專案使用 [ms-swift](https://github.com/modelscope/ms-swift) 框架對 [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) 模型進行 fine-tuning。

#### Embedding 模型訓練

參考 `scripts/train.sh`：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NVIDIA_TF32_OVERRIDE=1
export INFONCE_USE_BATCH=true
export INFONCE_MASK_FAKE_NEGATIVE=true

NPROC_PER_NODE=2 \
swift sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type full \
  --dataset data/train.jsonl \
  --val_dataset data/dev.jsonl \
  --output_dir output \
  --eval_strategy steps --eval_steps 100 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 6e-6 \
  --loss_type infonce \
  --dataloader_drop_last true \
  --dataloader_num_workers 16 \
  --dataloader_persistent_workers true \
  --attn_impl flash_attn \
  --bf16 true \
  --use_hf true \
  --seed 42
```

#### Reranker 模型訓練

```bash
# 使用相同的 train.jsonl 數據，但改用 reranker 任務類型
swift sft \
  --model Qwen/Qwen3-Reranker-4B \
  --task_type generative_reranker \
  --loss_type generative_reranker \
  --train_type full \
  --dataset data/train.jsonl \
  --val_dataset data/dev.jsonl \
  --output_dir output/reranker \
  --eval_strategy steps --eval_steps 100 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-6 \
  --dataloader_drop_last true \
  --bf16 true
```

### 訓練建議
- `per_device_train_batch_size` 建議在 GPU 記憶體允許下盡量拉高，提升訓練穩定度
- 訓練 InfoNCE 時，啟用 `INFONCE_USE_BATCH=true` 可將batch內的其他正樣本作爲負樣本，提升多樣性
- 配合 `INFONCE_MASK_FAKE_NEGATIVE=true` 可遮蔽假陰性樣本，提升效果
- 使用 `--attn_impl flash_attn` 和 `--bf16 true` 可大幅提升訓練速度（需 Ampere 架構以上 GPU）
- `--dataloader_persistent_workers true` 可減少 dataloader 重啟開銷

### 模型部署與推理

#### Embedding 模型部署

參考 `scripts/deploy.sh`：

```bash
# 使用部署腳本進行模型服務部署 (scripts/deploy.sh)
swift deploy \
  --ckpt_dir ../output/my-training/checkpoint-xxx \
  --served_model_name Qwen3-Embedding-0.6B-finetuned \
  --task_type embedding \
  --infer_backend vllm \
  --torch_dtype float16

# 使用vllm進行部署
vllm serve ../output/my-training/checkpoint-xxx \
  --served_model_name Qwen3-Embedding-0.6B-finetuned

# 部署原始版本模型
swift deploy \
  --model Qwen/Qwen3-Embedding-0.6B \
  --served_model_name Qwen3-Embedding-0.6B-base \
  --task_type embedding \
  --infer_backend vllm \
  --torch_dtype float16
```

#### Reranker 模型部署

參考 `scripts/deploy_rerank.sh`：

```bash
export CUDA_VISIBLE_DEVICES=1

# 使用 vllm 部署 Reranker（swift<3.9 有 bug，直接用 vllm）
vllm serve output/reranker/my-training/checkpoint-xxx \
   --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
   --port 8001 \
   --host 0.0.0.0 \
   --served_model_name Qwen/Qwen3-Reranker-0.6B-v2-12000 \
   --max-model-len 8192

# 部署基礎 Reranker 模型
vllm serve Qwen/Qwen3-Reranker-0.6B \
   --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
   --port 8001 \
   --host 0.0.0.0
```

### InfoNCE 資料格式規範

#### 格式
```json
# sample without rejected_response
{"query": "sentence1", "response": "sentence1-positive"}
# sample with multiple rejected_response
{"query": "sentence1", "response": "sentence1-positive", "rejected_response":  ["sentence1-negative1", "sentence1-negative2", ...]}
```

#### InfoNCE 環境變數設定

InfoNCE loss 支援以下環境變數：

- **INFONCE_TEMPERATURE**: 溫度參數。若未設定，預設值為 0.01
- **INFONCE_USE_BATCH**: 決定是否使用樣本內的 rejected_response（hard negative samples）或使用批次內的所有回應。預設為 True，表示使用批次內的回應
- **INFONCE_HARD_NEGATIVES**: hard negatives 的數量。若未設定，將使用 rejected_response 中的所有樣本。由於長度可能不一致，會使用 for 迴圈計算損失（較慢）。若設定為特定數值，且樣本不足時，會隨機取樣補足；若樣本過多，則會選取前 INFONCE_HARD_NEGATIVES 個
- **INFONCE_MASK_FAKE_NEGATIVE**: 遮蔽假陰性樣本。預設為 False。啟用時，會檢查樣本的相似度是否大於正樣本相似度加 0.1，若是則將該樣本的相似度設為 -inf，以防止正樣本洩漏

> **注意**: 也可以在資料集中設定相等的 hard negatives 數量，這樣即使未設定也不會使用 for 迴圈方法，從而加速計算。
> 
> rejected_response 也可以省略。在此情況下，INFONCE_USE_BATCH 保持為 True，會使用批次內的其他樣本作為 rejected responses。

#### InfoNCE 評估指標

InfoNCE loss 的評估包含以下指標：

- **mean_neg**: 所有 hard negatives 的平均值
- **mean_pos**: 所有 positives 的平均值  
- **margin**: (positive - max hard negative) 的平均值

參考資料：[ms-swift InfoNCE 格式文檔](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md#format-for-infonce)

## 資料準備和轉換

### 原始資料集
本專案使用[TCM-SD](https://github.com/Borororo/ZY-BERT)資料集，包含：
- **train.jsonl**: 43,180 筆訓練案例
- **test.jsonl**: 5,486 筆測試案例  
- **dev.jsonl**: 5,486 筆驗證資料
- **syndrome_knowledge.jsonl**: 1,027 筆症候知識
- **syndrome_vocab.txt**:148 筆症候詞

### 資料轉換工具

使用 `scripts/convert_to_infonce.py` 將原始病例資料轉換為適合 InfoNCE 訓練的格式：

```bash
# 轉換訓練資料
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train.jsonl

# 限制樣本數量（用於測試）
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_sample.jsonl \
    --max-samples 10
```

### 資料格式轉換說明

`convert_to_infonce.py` 將 TCM-SD 病例記錄轉換為 InfoNCE 訓練格式：

**基本轉換**：
- **query**: 組合「主訴」、「現病史」、「體格檢查」等臨床資訊
- **response**: 根據證型（syndrome）匹配知識庫，包含「名稱」、「定義」、「典型表現」、「常見疾病」等

**數據增強選項**：
- `--with-hard-negatives`: 使用 BM25 生成硬負樣本
- `--with-hard-negatives-custom-embedding`: 使用自定義 Embedding API 生成硬負樣本
- `--hybrid`: 混合負樣本（2 隨機 + 3 BM25 + 3 Embedding）
- `--split-negatives`: 將多個 rejected_response 拆分為獨立樣本
- `--field-combinations`: 生成 11 種字段組合（名稱、定義、典型表現等的不同組合）
- `--field-permutations`: 生成 24 種字段排列（所有可能的字段順序）
- `--permutation-argument --multiplier N`: Zone-based 排列增強（N 倍數據）

**輸出格式**：
```json
// 標準格式
{"query": "主訴：...現病史：...體格檢查：...", "response": "名稱：...定義：..."}

// 帶硬負樣本
{"query": "...", "response": "...", "rejected_response": ["...", "...", "..."]}
```
## 開發

### 本地開發 ms-swift
```bash
# 如需修改 ms-swift 源碼，可安裝本地版本
uv pip install -e ../ms-swift
```

## 專案結構

```
TCMEmbeddingModel/
├── README.md                                # 專案說明文件
├── pyproject.toml                           # 專案配置和依賴管理 (uv)
├── uv.lock                                  # uv 依賴鎖定文件
├── data/                                    # 資料目錄
│   ├── raw_data/TCM_SD/                    # 原始 TCM-SD 資料集
│   │   ├── train.jsonl                     # 43,180 筆訓練案例
│   │   ├── dev.jsonl                       # 5,486 筆驗證案例
│   │   ├── test.jsonl                      # 5,486 筆測試案例
│   │   ├── syndrome_knowledge.jsonl        # 1,027 筆證型知識
│   │   └── syndrome_vocab.txt              # 148 個證型詞彙
│   ├── train.jsonl                         # 轉換後的訓練資料
│   ├── dev.jsonl                           # 轉換後的驗證資料
│   ├── test.jsonl                          # 轉換後的測試資料
│   ├── train_with_hard_negatives.jsonl     # 帶 BM25 硬負樣本
│   ├── train_hybrid.jsonl                  # 混合硬負樣本
│   ├── train_field_combinations_*.jsonl    # 字段組合增強
│   └── train_permutation_argument_*.jsonl  # 排列增強
├── scripts/                                 # 工具腳本
│   ├── convert_to_infonce.py               # 資料轉換工具（1400+ 行）
│   ├── load_data.sh                        # 資料轉換範例腳本
│   ├── train.sh                            # Embedding 訓練腳本
│   ├── deploy.sh                           # Embedding 部署腳本
│   └── deploy_rerank.sh                    # Reranker 部署腳本
├── output/                                  # 訓練輸出
└── .venv/                                   # uv 虛擬環境 (不納入版控)
```

## uv 配置說明

### pyproject.toml 實際配置
```toml
[project]
name = "tcmembeddingmodel"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "jieba>=0.42.1",        # 中文分詞（用於 BM25）
    "ms-swift>=3.7.3",      # 訓練框架
    "rank-bm25>=0.2.2",     # BM25 硬負樣本生成
    "vllm>=0.10.1.1",       # 推理引擎
]
```

### 依賴說明
- **jieba**: 中文分詞工具，用於 BM25 硬負樣本生成
- **ms-swift**: ModelScope SWIFT 訓練框架（>=3.7.3）
- **rank-bm25**: BM25 檢索算法，用於生成硬負樣本
- **vllm**: 高效推理引擎，用於模型部署

## 技術參考文件
- [ms-swift Embedding 最佳實踐](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md)
- [Qwen3-Embedding SWIFT 訓練支援](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md#swift-training-support)
- [TCM-SD 資料集論文](https://github.com/Borororo/ZY-BERT)
