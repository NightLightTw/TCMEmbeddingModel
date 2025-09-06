# TCM Embedding Model

基於 Qwen3-Embedding 模型針對傳統中醫（Traditional Chinese Medicine, TCM）領域進行 Fine-tuning 的專案。

## 專案概述

本專案旨在利用 Qwen3-Embedding 模型的強大語義理解能力，透過傳統中醫相關文獻和資料進行微調，建立專門針對傳統中醫領域的高品質嵌入模型。

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
- 硬碟空間：至少 50GB 可用空間

### 硬體建議
- **訓練環境**：NVIDIA GPU (V100/A100/RTX 4090 或更高)
- **推理環境**：可在 CPU 上運行，GPU 可提升效能

## 安裝指南

### 1. 安裝 uv
如果您還沒有安裝 uv，請先安裝：

```bash
# macOS 和 Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip 安裝
pip install uv
```

### 2. 克隆專案
```bash
git clone <repository-url>
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
# 或 .venv\Scripts\activate  # Windows

# 或使用 uv run 直接運行命令（推薦）
uv run python main.py
```

### 5. (可選)加速工具
```bash
# Optional packages
uv pip install deepspeed # multi-GPU training
uv pip install liger-kernel # save GPU memory resources
uv pip install flash-attn --no-build-isolation
```

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
```bash
# 將原始資料轉換為 InfoNCE 格式
uv run python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train_full.jsonl
```

### 開始訓練
```bash
# 訓練腳本 (scripts/train.sh)
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
  --learning_rate 6e-6 \
  --loss_type infonce \
  --bf16 true
```

### 模型推理
```bash
# 使用部署腳本進行模型服務部署 (scripts/deploy.sh)
swift deploy \
  --ckpt_dir ../output/my-training/checkpoint-xxx \
  --served_model_name Qwen3-Embedding-0.6B-finetuned \
  --task_type embedding \
  --infer_backend vllm \
  --torch_dtype float16

# 部署原始版本模型
swift deploy \
  --model Qwen/Qwen3-Embedding-0.6B \
  --served_model_name Qwen3-Embedding-0.6B-base \
  --task_type embedding \
  --infer_backend vllm \
  --torch_dtype float16

# 或者使用推理模式
uv run swift infer \
    --ckpt_dir output/my-training/checkpoint-xxx \
    --infer_data_path data/infer_example.jsonl
```

### SWIFT Fine-tuning 指南

本專案使用 [ms-swift](https://github.com/modelscope/ms-swift) 框架對 [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) 模型進行 fine-tuning。

#### 安裝 SWIFT 框架
```bash
# 使用 uv 安裝 SWIFT 相關依賴
uv add ms-swift

# 驗證安裝
uv run swift --version
```

#### SWIFT 訓練範例
```bash
# 使用 SWIFT 命令行工具進行訓練
nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset sentence-transformers/stsb:positive \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
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

轉換腳本會將病例記錄：
- **query**: 組合「主訴」、「現病史」、「體格檢查」等臨床資訊
- **response**: 根據症候類型匹配對應的知識庫內容，包含「名稱」、「定義」、「典型表現」、「常見疾病」等

## 專案結構

```
TCMEmbeddingModel/
├── README.md                    # 專案說明文件
├── pyproject.toml               # 專案配置和依賴管理 (uv 配置)
├── uv.lock                     # uv 依賴鎖定文件
├── main.py                     # 主要執行入口
├── data/                       # 資料目錄
├── scripts/                    # 工具腳本
│   ├── convert_to_infonce.py  # 將案例資料轉換為 InfoNCE 格式
│   ├── train.sh               # 訓練腳本
│   └── deploy.sh              # 部署腳本
├── output/                     # 訓練輸出
│   └── vX-XXXXXXXX-XXXXXX/    # 訓練結果
│       ├── checkpoint-XXXX/   # 模型檢查點
│       ├── logging.jsonl      # 訓練日誌
│       └── runs/              # TensorBoard 日誌
└── .venv/                     # uv 建立的虛擬環境 (不納入版控)
```

## uv 配置說明

### pyproject.toml 重要配置
- **依賴管理**：使用 uv 進行快速依賴解析和安裝
- **Python 版本**：固定使用 Python 3.10
- **核心依賴**：ms-swift, torch, transformers
- **命令列工具**：預留了未來的 CLI 命令入口

### SWIFT 框架特色
- 🚀 **高效微調**：支援 LoRA、QLoRA、Adapter 等參數高效微調方法
- 🔄 **分散式訓練**：支援 DDP、模型並行、流水線並行
- 📊 **多種任務**：支援文本分類、序列標註、embedding 等任務
- 🛠️ **易於使用**：提供命令行工具和 Python API

### uv 優勢
- 🚀 **速度**：比 pip 快 10-100 倍的依賴安裝
- 🔒 **穩定**：uv.lock 確保依賴版本一致性
- 🛠️ **簡單**：統一的專案管理工具
- 🐍 **Python 管理**：自動管理 Python 版本

## 技術參考文件
- [ms-swift Embedding 最佳實踐](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md)
- [Qwen3-Embedding SWIFT 訓練支援](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md#swift-training-support)
- [TCM-SD 資料集論文](https://github.com/Borororo/ZY-BERT)
