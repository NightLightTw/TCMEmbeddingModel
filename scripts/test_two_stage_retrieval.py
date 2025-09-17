#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兩階段檢索測試腳本：Embedding模型粗排(Top100) + Reranker精排
"""

import json
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import argparse


class TwoStageRetriever:
    def __init__(self, embedding_url="http://localhost:8000", reranker_model_path="Qwen/Qwen3-Reranker-0.6B", reranker_api_url=None):
        self.embedding_url = embedding_url
        self.embedding_model = "Qwen3-Embedding-0.6B-finetuned"
        self.reranker_model = reranker_model_path
        self.reranker_api_url = reranker_api_url
        
        # 決定使用 API 還是本地模型
        self.use_reranker_api = reranker_api_url is not None
        
        if self.use_reranker_api:
            print(f"✓ 使用 Reranker API: {reranker_api_url}")
            self.reranker_tokenizer = None
            self.reranker_model_instance = None
        else:
            print(f"✓ 使用本地 Reranker 模型: {reranker_model_path}")
            # 載入reranker模型
            self.reranker_tokenizer = None
            self.reranker_model_instance = None
            self._load_reranker_model()
        
        # 存儲文檔庫
        self.document_store = []
    
    def _load_reranker_model(self):
        """載入reranker模型"""
        try:
            print("正在載入reranker模型...")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model, 
                trust_remote_code=True
            )
            self.reranker_model_instance = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).half()
            
            # 如果有CUDA可用，移到GPU
            if torch.cuda.is_available():
                self.reranker_model_instance = self.reranker_model_instance.cuda()
                print("✓ Reranker模型已載入到GPU")
            else:
                print("✓ Reranker模型已載入到CPU")
                
        except Exception as e:
            print(f"✗ 載入reranker模型失敗: {str(e)}")
            self.reranker_tokenizer = None
            self.reranker_model_instance = None
    
    def load_document_store(self, jsonl_file: str, max_docs: int = 1000, deduplicate: bool = True) -> None:
        """從JSONL文件載入文檔庫"""
        print(f"正在載入文檔庫: {jsonl_file}")
        
        seen_responses = set()
        total_count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(self.document_store) >= max_docs:
                    break
                    
                total_count += 1
                data = json.loads(line.strip())
                response = data['response']
                
                # 如果啟用去重且回應已存在，跳過
                if deduplicate and response in seen_responses:
                    continue
                
                seen_responses.add(response)
                self.document_store.append({
                    'id': len(self.document_store),
                    'query': data['query'],
                    'response': response
                })
        
        if deduplicate:
            print(f"✓ 已載入 {len(self.document_store)} 個去重文檔 (原始總數: {total_count}, 去重率: {(total_count - len(self.document_store))/total_count*100:.1f}%)")
        else:
            print(f"✓ 已載入 {len(self.document_store)} 個文檔")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """獲取文本的嵌入向量"""
        try:
            response = requests.post(
                f"{self.embedding_url}/v1/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result["data"][0]["embedding"], dtype=np.float32)
                # L2歸一化
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                print(f"✗ 獲取嵌入向量失敗: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"✗ 獲取嵌入向量異常: {str(e)}")
            return None
    
    def embedding_retrieval(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """使用embedding模型進行粗排檢索"""
        print(f"階段1: 使用embedding模型檢索 Top-{top_k}")
        
        # 獲取查詢的嵌入向量
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # 計算所有文檔的相似度
        similarities = []
        for i, doc in enumerate(self.document_store):
            print(f"  處理文檔 {i+1}/{len(self.document_store)}", end='\r')
            
            # 使用response作為候選文檔進行匹配
            doc_embedding = self.get_embedding(doc['response'])
            if doc_embedding is not None:
                # 計算余弦相似度
                similarity = float(query_embedding @ doc_embedding)
                similarities.append({
                    'doc_id': doc['id'],
                    'similarity': similarity,
                    'query': doc['query'],
                    'response': doc['response']
                })
        
        print()  # 換行
        
        # 排序並返回top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _classify_with_reranker(self, query: str, document: str) -> Dict[str, Any]:
        """使用reranker模型進行分類"""
        if self.reranker_tokenizer is None or self.reranker_model_instance is None:
            return {"label": "no", "scores": {"no": 1.0, "yes": 0.0}, "error": "模型未載入"}
        
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
            inputs = self.reranker_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            )
            
            # 移到正確的設備
            if torch.cuda.is_available() and self.reranker_model_instance.device.type == 'cuda':
                inputs = inputs.to("cuda")
            
            # 推理
            with torch.no_grad():
                outputs = self.reranker_model_instance(**inputs)
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
            return {"label": "no", "scores": {"no": 1.0, "yes": 0.0}, "error": str(e)}
    
    def _rerank_with_api(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """使用 Reranker API 進行批量重排 - 更新為標準格式"""
        try:
            response = requests.post(
                f"{self.reranker_api_url}/v1/rerank",
                json={
                    "model": self.reranker_model,
                    "query": query,
                    "documents": documents,
                    "return_documents": True  # 確保返回文檔內容
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["results"]
            else:
                print(f"✗ Reranker API 調用失敗: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"✗ Reranker API 調用異常: {str(e)}")
            return []

    def rerank_retrieval(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用reranker模型進行精排"""
        print(f"階段2: 使用reranker對 {len(candidates)} 個候選文檔進行重排")
        
        if self.use_reranker_api:
            # 使用 API 進行重排
            documents = [candidate['response'] for candidate in candidates]
            api_results = self._rerank_with_api(query, documents)
            
            if not api_results:
                print("✗ Reranker API 調用失敗，跳過重排序")
                return candidates
            
            # 將 API 結果映射回原始候選者
            reranked_results = []
            for api_result in api_results:
                original_idx = api_result["index"]
                if original_idx < len(candidates):
                    candidate = candidates[original_idx]
                    relevance_score = api_result["relevance_score"]
                    # 根據相關性分數判斷標籤 (>0.5 為 yes)
                    rerank_label = "yes" if relevance_score > 0.5 else "no"
                    
                    reranked_results.append({
                        **candidate,
                        'rerank_score': relevance_score,
                        'rerank_label': rerank_label,
                        'rerank_yes_prob': relevance_score,
                        'rerank_no_prob': 1.0 - relevance_score
                    })
            
            return reranked_results
        
        else:
            # 使用本地模型進行重排
            if self.reranker_tokenizer is None or self.reranker_model_instance is None:
                print("✗ Reranker模型未載入，跳過重排序")
                return candidates
            
            reranked_results = []
            
            for i, candidate in enumerate(candidates):
                print(f"  重排序文檔 {i+1}/{len(candidates)}", end='\r')
                
                # 使用reranker評估相關性
                result = self._classify_with_reranker(query, candidate['response'])
                
                yes_prob = result["scores"].get("yes", 0.0)
                reranked_results.append({
                    **candidate,
                    'rerank_score': yes_prob,
                    'rerank_label': result["label"],
                    'rerank_yes_prob': yes_prob,
                    'rerank_no_prob': result["scores"].get("no", 1.0)
                })
            
            print()  # 換行
            
            # 按rerank分數排序
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked_results
    
    def two_stage_retrieval(self, query: str, embedding_top_k: int = 100, final_top_k: int = 10) -> Dict[str, Any]:
        """執行兩階段檢索"""
        print("=" * 60)
        print("開始兩階段檢索")
        print("=" * 60)
        print(f"查詢: {query[:100]}...")
        print(f"文檔庫大小: {len(self.document_store)}")
        print()
        
        start_time = time.time()
        
        # 階段1: Embedding粗排
        embedding_start = time.time()
        embedding_results = self.embedding_retrieval(query, embedding_top_k)
        embedding_time = time.time() - embedding_start
        print(f"✓ 階段1完成，耗時: {embedding_time:.2f}秒")
        
        if not embedding_results:
            return {
                "success": False,
                "error": "Embedding檢索失敗",
                "query": query
            }
        
        # 階段2: Reranker精排
        rerank_start = time.time()
        reranked_results = self.rerank_retrieval(query, embedding_results)
        rerank_time = time.time() - rerank_start
        print(f"✓ 階段2完成，耗時: {rerank_time:.2f}秒")
        
        total_time = time.time() - start_time
        
        # 返回最終結果
        final_results = reranked_results[:final_top_k]
        
        return {
            "success": True,
            "query": query,
            "embedding_results": embedding_results,
            "reranked_results": reranked_results,
            "final_results": final_results,
            "timing": {
                "embedding_time": embedding_time,
                "rerank_time": rerank_time,
                "total_time": total_time
            },
            "stats": {
                "total_docs": len(self.document_store),
                "embedding_top_k": len(embedding_results),
                "final_top_k": len(final_results)
            }
        }


def analyze_results(result: Dict[str, Any]) -> None:
    """分析檢索結果"""
    print("\n" + "=" * 60)
    print("結果分析")
    print("=" * 60)
    
    if not result["success"]:
        print(f"✗ 檢索失敗: {result.get('error')}")
        return
    
    # 時間統計
    timing = result["timing"]
    print(f"時間統計:")
    print(f"  Embedding檢索: {timing['embedding_time']:.2f}秒")
    print(f"  Reranker重排: {timing['rerank_time']:.2f}秒")
    print(f"  總耗時: {timing['total_time']:.2f}秒")
    
    # 統計信息
    stats = result["stats"]
    print(f"\n檢索統計:")
    print(f"  文檔庫大小: {stats['total_docs']}")
    print(f"  Embedding Top-K: {stats['embedding_top_k']}")
    print(f"  最終結果數: {stats['final_top_k']}")
    
    # Reranker分析
    reranked_results = result["reranked_results"]
    yes_count = sum(1 for r in reranked_results if r['rerank_label'] == 'yes')
    avg_yes_prob = np.mean([r['rerank_yes_prob'] for r in reranked_results])
    
    print(f"\nReranker分析:")
    print(f"  相關文檔數: {yes_count}/{len(reranked_results)}")
    print(f"  平均相關性概率: {avg_yes_prob:.3f}")
    
    # 顯示前5個結果
    print(f"\n前5個最終結果:")
    for i, item in enumerate(result["final_results"][:5]):
        embedding_sim = item['similarity']
        rerank_score = item['rerank_score']
        rerank_label = item['rerank_label']
        response = item['response']
        
        # 提取證型名稱（第一行）
        syndrome_name = response.split('\n')[0] if '\n' in response else response[:50]
        
        print(f"  {i+1}. 嵌入相似度: {embedding_sim:.3f} | "
              f"重排分數: {rerank_score:.3f} ({rerank_label}) | "
              f"證型: {syndrome_name}")
        
        # 如果想看完整內容，可以顯示更多信息
        if i < 3:  # 只顯示前3個的詳細信息
            definition_line = response.split('\n')[1] if '\n' in response else ""
            if definition_line.startswith("定义："):
                print(f"    {definition_line}")
        print()


def get_random_test_queries_from_store(document_store: List[Dict[str, Any]], n_queries: int = 3) -> List[str]:
    """從已載入的文檔庫中隨機選擇n個查詢（確保多樣性）"""
    print(f"正在從文檔庫中隨機選擇 {n_queries} 個測試查詢...")
    
    # 從去重後的文檔庫中選擇查詢
    all_queries = [doc['query'] for doc in document_store]
    
    # 隨機選擇n個查詢
    selected_queries = random.sample(all_queries, min(n_queries, len(all_queries)))
    
    print(f"✓ 已選擇 {len(selected_queries)} 個隨機測試查詢")
    return selected_queries


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='TCM 兩階段檢索測試')
    parser.add_argument('--reranker', '--reranker-model', 
                       default='Qwen/Qwen3-Reranker-0.6B',
                       help='Reranker模型路徑 (例如: output/reranker/v0-20250911-010112/checkpoint-3375)')
    parser.add_argument('--reranker-api-url',
                       help='Reranker API URL (例如: http://localhost:8001)，如果提供則使用API而非本地模型')
    parser.add_argument('--embedding-url', 
                       default='http://localhost:8000',
                       help='Embedding API 端點URL')
    parser.add_argument('--max-docs', type=int, default=500,
                       help='從文檔庫載入的最大文檔數')
    parser.add_argument('--test-queries', type=int, default=3,
                       help='測試查詢數量')
    
    args = parser.parse_args()
    
    print("TCM 兩階段檢索測試")
    print("=" * 60)
    if args.reranker_api_url:
        print(f"Reranker API: {args.reranker_api_url}")
        print(f"Reranker模型名: {args.reranker}")
    else:
        print(f"Reranker模型: {args.reranker} (本地載入)")
    print(f"Embedding API: {args.embedding_url}")
    print(f"最大文檔數: {args.max_docs}")
    print(f"測試查詢數: {args.test_queries}")
    print("=" * 60)
    
    # 初始化檢索器
    retriever = TwoStageRetriever(
        embedding_url=args.embedding_url,
        reranker_model_path=args.reranker,
        reranker_api_url=args.reranker_api_url
    )
    
    # 載入文檔庫（啟用去重）
    retriever.load_document_store("/mnt/nfs/work/david97099/Github/TCMEmbeddingModel/data/test.jsonl", max_docs=args.max_docs, deduplicate=True)
    
    if not retriever.document_store:
        print("✗ 文檔庫為空，退出")
        return
    
    # 從去重後的文檔庫中隨機選擇測試查詢（確保多樣性）
    test_queries = get_random_test_queries_from_store(retriever.document_store, n_queries=args.test_queries)
    
    # 對每個查詢進行兩階段檢索
    for i, query in enumerate(test_queries):
        print(f"\n{'='*20} 測試案例 {i+1} {'='*20}")
        print(f"隨機選擇的查詢: {query[:100]}...")
        
        # 執行兩階段檢索
        result = retriever.two_stage_retrieval(
            query=query,
            embedding_top_k=100,
            final_top_k=10
        )
        
        # 分析結果
        analyze_results(result)


if __name__ == "__main__":
    main()
