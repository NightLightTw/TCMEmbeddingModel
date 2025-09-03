# TCM Embedding Model

基於 Qwen3-Embedding 模型針對傳統中醫（Traditional Chinese Medicine, TCM）領域進行 Fine-tuning 的專案。

## 專案概述

本專案旨在利用 Qwen3-Embedding 模型的強大語義理解能力，透過傳統中醫相關文獻和資料進行微調，建立專門針對傳統中醫領域的高品質嵌入模型。

## 目標功能

- 🏥 **中醫領域特化**：針對中醫術語、診斷、方劑等專業內容優化
- 🔍 **語義檢索**：提供精確的中醫文獻和知識檢索能力
- 🚀 **高效訓練**：基於 ms-swift 框架的高效 fine-tuning 流程

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


## 專案結構

```
TCMEmbeddingModel/
├── README.md              # 專案說明文件
├── pyproject.toml         # 專案配置和依賴管理 (uv 配置)
├── uv.lock               # uv 依賴鎖定文件
├── .python-version       # Python 版本指定
├── .gitignore           # Git 忽略文件配置
├── main.py              # 主要執行入口
└── .venv/               # uv 建立的虛擬環境 (不納入版控)
```

### 計劃中的結構
未來將逐步建立以下目錄結構：
- `src/tcmembeddingmodel/` - 主要模組
- `tests/` - 測試目錄  
- `configs/` - 配置文件
- `data/` - 資料目錄
- `scripts/` - 工具腳本
- `docs/` - 文檔目錄

## uv 配置說明

### pyproject.toml 重要配置
- **依賴管理**：使用 uv 進行快速依賴解析和安裝
- **Python 版本**：固定使用 Python 3.10
- **核心依賴**：ms-swift, torch, transformers
- **開發依賴**：包含代碼品質工具 (black, isort, pytest 等)
- **命令列工具**：預留了未來的 CLI 命令入口

### SWIFT 框架特色
- 🚀 **高效微調**：支援 LoRA、QLoRA、Adapter 等參數高效微調方法
- 🔄 **分散式訓練**：支援 DDP、模型並行、流水線並行
- 📊 **多種任務**：支援文本分類、序列標註、embedding 等任務
- 🛠️ **易於使用**：提供命令行工具和 Python API
- 📚 **豐富文檔**：詳細的 [最佳實踐指南](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md)

### uv 優勢
- 🚀 **速度**：比 pip 快 10-100 倍的依賴安裝
- 🔒 **穩定**：uv.lock 確保依賴版本一致性
- 🛠️ **簡單**：統一的專案管理工具
- 🐍 **Python 管理**：自動管理 Python 版本

## 技術參考文件
- [ms-swift Embedding 最佳實踐](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md)
- [Qwen3-Embedding SWIFT 訓練支援](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md#swift-training-support)
