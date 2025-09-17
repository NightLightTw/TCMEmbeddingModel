#!/usr/bin/env bash
set -euo pipefail

# Reranker API 部署腳本

# 配置參數
DEFAULT_MODEL_PATH="output/reranker/v0-20250911-010112/checkpoint-3375"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8001
DEFAULT_CUDA_DEVICE=1

# 解析命令行參數
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
HOST="${2:-$DEFAULT_HOST}"
PORT="${3:-$DEFAULT_PORT}"
CUDA_DEVICE="${4:-$DEFAULT_CUDA_DEVICE}"

echo "=================================================="
echo "           TCM Reranker API 部署腳本"
echo "=================================================="
echo "模型路徑: $MODEL_PATH"
echo "主機地址: $HOST"
echo "端口號: $PORT"
echo "CUDA 設備: $CUDA_DEVICE"
echo "=================================================="

# 檢查模型路徑是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 錯誤: 模型路徑 '$MODEL_PATH' 不存在"
    echo ""
    echo "可用的模型:"
    ls -la output/reranker/*/checkpoint-* 2>/dev/null | head -10 || echo "  (未找到模型)"
    echo ""
    echo "使用方法: $0 [MODEL_PATH] [HOST] [PORT] [CUDA_DEVICE]"
    echo "範例: $0 output/reranker/v0-20250911-010112/checkpoint-3375 0.0.0.0 8001 0"
    exit 1
fi

# 設置環境變量
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export NVIDIA_TF32_OVERRIDE=1

# 檢查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  警告: 端口 $PORT 已被占用"
    echo "正在嘗試停止現有服務..."
    
    # 嘗試優雅停止
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    if [ ! -z "$PID" ]; then
        kill -TERM $PID 2>/dev/null || true
        sleep 2
        
        # 如果還在運行，強制停止
        if kill -0 $PID 2>/dev/null; then
            kill -KILL $PID 2>/dev/null || true
            sleep 1
        fi
    fi
fi

echo ""
echo "🚀 正在啟動 Reranker API 服務..."
echo "📱 API 文檔將在以下地址可用:"
echo "   - Swagger UI: http://$HOST:$PORT/docs"
echo "   - ReDoc: http://$HOST:$PORT/redoc"
echo "   - 健康檢查: http://$HOST:$PORT/health"
echo ""
echo "💡 使用 Ctrl+C 停止服務"
echo ""

# 激活虛擬環境並啟動服務
source .venv/bin/activate

# 啟動 Reranker API 服務
python scripts/reranker_api.py \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT"