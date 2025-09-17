#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reranker API 服務
基於 FastAPI 提供 reranker 推理服務
"""

import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TCM Reranker API",
    description="Traditional Chinese Medicine Reranker Service",
    version="1.0.0"
)

# 全局變量存儲模型
reranker_tokenizer = None
reranker_model = None
model_name = None


class RerankRequest(BaseModel):
    """Rerank 請求格式 - 符合業界標準"""
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None  # 改為 top_n 符合標準
    return_documents: Optional[bool] = True  # 是否返回文檔內容


class RerankResponse(BaseModel):
    """Rerank 回應格式 - 符合 Jina AI 標準"""
    results: List[Dict[str, Any]]
    usage: Dict[str, int]


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    model: str
    device: str


def load_reranker_model(model_path: str):
    """載入 reranker 模型"""
    global reranker_tokenizer, reranker_model, model_name
    
    try:
        logger.info(f"正在載入 reranker 模型: {model_path}")
        
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).half()
        
        # 移動到 GPU
        if torch.cuda.is_available():
            reranker_model = reranker_model.cuda()
            device_info = f"GPU:{torch.cuda.get_device_name()}"
            logger.info(f"✓ Reranker 模型已載入到 GPU: {torch.cuda.get_device_name()}")
        else:
            device_info = "CPU"
            logger.info("✓ Reranker 模型已載入到 CPU")
        
        model_name = model_path
        logger.info("✓ Reranker 模型載入完成")
        
    except Exception as e:
        logger.error(f"✗ 載入 reranker 模型失敗: {str(e)}")
        raise RuntimeError(f"Failed to load reranker model: {str(e)}")


def classify_with_reranker(query: str, document: str) -> Dict[str, Any]:
    """使用 reranker 模型進行分類"""
    if reranker_tokenizer is None or reranker_model is None:
        raise HTTPException(status_code=500, detail="Reranker model not loaded")
    
    try:
        # 構建輸入
        instruction = (
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
        )
        prompt = (
            f"<Instruction>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        
        # 分詞
        inputs = reranker_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )
        
        # 移到正確的設備
        if torch.cuda.is_available() and reranker_model.device.type == 'cuda':
            inputs = inputs.to("cuda")
        
        # 推理
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
        
        # 解析結果
        label_idx = torch.argmax(probs).item()
        labels = ["no", "yes"]
        
        return {
            "label": labels[label_idx],
            "scores": {labels[i]: float(probs[i]) for i in range(len(labels))}
        }
        
    except Exception as e:
        logger.error(f"Reranker inference error: {str(e)}")
        return {"label": "no", "scores": {"no": 1.0, "yes": 0.0}, "error": str(e)}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    device = "GPU" if torch.cuda.is_available() and reranker_model.device.type == 'cuda' else "CPU"
    
    return HealthResponse(
        status="healthy",
        model=model_name or "unknown",
        device=device
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Rerank 端點 - 符合業界標準格式"""
    if reranker_tokenizer is None or reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded")
    
    # 參數驗證
    if not request.documents:
        raise HTTPException(status_code=400, detail="Documents list cannot be empty")
    
    if len(request.documents) > 1000:  # 設置合理的上限
        raise HTTPException(status_code=400, detail="Too many documents (max 1000)")
    
    try:
        results = []
        
        # 對每個文檔計算 relevance score
        for i, document in enumerate(request.documents):
            result = classify_with_reranker(request.query, document)
            
            relevance_score = result["scores"].get("yes", 0.0)
            
            # 構建符合標準的結果格式
            result_item = {
                "index": i,
                "relevance_score": relevance_score
            }
            
            # 根據 return_documents 參數決定是否包含文檔內容
            if request.return_documents:
                result_item["document"] = {
                    "text": document
                }
            
            results.append(result_item)
        
        # 按 relevance_score 排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 如果指定了 top_n，只返回前 n 個
        if request.top_n and request.top_n > 0:
            results = results[:request.top_n]
        
        return RerankResponse(
            results=results,
            usage={
                "total_documents": len(request.documents),
                "returned_documents": len(results)
            }
        )
        
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        logger.error(f"Rerank API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """根端點"""
    return {
        "message": "TCM Reranker API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "rerank": "/v1/rerank"
        }
    }


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="TCM Reranker API Server")
    parser.add_argument(
        "--model-path", 
        required=True,
        help="Path to the reranker model"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8001,
        help="Port to bind to (default: 8001)"
    )
    
    args = parser.parse_args()
    
    # 載入模型
    load_reranker_model(args.model_path)
    
    # 啟動服務
    logger.info(f"啟動 Reranker API 服務於 {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
