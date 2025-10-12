#!/usr/bin/env python3
"""
Convert case records into InfoNCE JSONL for embedding fine-tuning.

Input cases can be JSON (array or object with a list field) or JSONL.
For each record, the script builds:

  - query: "主诉：{chief_complaint}现病史：{description}体格检查：{detection}"
  - response: syndrome knowledge matched by {norm_syndrome|syndrome} from knowledge JSONL

Usage examples:
  # Standard format
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl

  # With hard negatives using BM25
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --with-hard-negatives

  # With hard negatives using custom embedding
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --with-hard-negatives-custom-embedding

  # With hybrid hard negatives (2 random + top-3 BM25 + top-3 custom embedding)
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --hybrid

  # With hybrid hard negatives, split into individual samples
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --hybrid \
    --split-negatives

  # With field combinations (generate multiple response formats)
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --field-combinations

  # With field permutations (generate all 24 field orderings)
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --field-permutations

  # With permutation argument (zone-based data augmentation with multiplier)
  python scripts/convert_to_infonce.py \
    --input ./data/raw_data/TCM_SD/train.jsonl \
    --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output ./data/train.jsonl \
    --permutation-argument \
    --multiplier 5

Options:
  --max-samples N                         Limit number of processed samples
  --with-hard-negatives                   Generate hard negatives using BM25; outputs format with rejected_response
  --with-hard-negatives-custom-embedding  Generate hard negatives using custom embedding API; outputs format with rejected_response
  --hybrid                                Generate hybrid hard negatives (2 random + top-3 BM25 + top-3 custom embedding); outputs format with rejected_response
  --split-negatives                       Split each sample with multiple rejected_response into separate samples, each with single rejected_response
  --field-combinations                    Generate multiple response formats with different field combinations
  --field-permutations                    Generate multiple response formats with different field permutations (all 24 orderings)
  --permutation-argument                  Generate augmented data using zone-based field permutations with multiplier
  --multiplier N                          Multiplier for data augmentation (default: 5). Only used with --permutation-argument

Output formats:
  Standard: {"query": "...", "response": "..."}
  Hard negatives: {"query": "...", "response": "...", "rejected_response": ["...", "..."]}
  Split negatives: {"query": "...", "response": "...", "rejected_response": ["..."]}
                   {"query": "...", "response": "...", "rejected_response": ["..."]}
  Field combinations: Multiple samples per case with different response field combinations:
                     {"query": "...", "response": "名称：..."}
                     {"query": "...", "response": "定义：..."}
                     {"query": "...", "response": "名称：... 定义：..."}
                     ... (11 combinations total)
  Field permutations: Multiple samples per case with different response field orderings:
                     {"query": "...", "response": "名称：... 定义：... 典型表现：... 常见疾病：..."}
                     {"query": "...", "response": "名称：... 定义：... 常见疾病：... 典型表现：..."}
                     {"query": "...", "response": "名称：... 典型表现：... 定义：... 常见疾病：..."}
                     ... (24 permutations total)
  Permutation argument: Zone-based data augmentation with multiplier:
                     Zone 1 (samples 1-1000): {"query": "...", "response": "名称：... 定义：... 典型表现：... 常见疾病：..."}
                     Zone 2 (samples 1001-2000): {"query": "...", "response": "定义：... 名称：... 常见疾病：... 典型表现：..."}
                     Zone 3 (samples 2001-3000): {"query": "...", "response": "典型表现：... 常见疾病：... 名称：... 定义：..."}
                     ... (multiplier zones total, each zone uses one fixed permutation)

Dependencies:
  - tqdm: For progress bars during processing
  - jieba: For Chinese text tokenization (with --with-hard-negatives)
  - rank-bm25: For BM25 similarity search (with --with-hard-negatives)
  - requests: For HTTP API calls (with --with-hard-negatives-custom-embedding)
  - numpy: For embedding similarity calculations (with --with-hard-negatives-custom-embedding)
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Tuple

import jieba
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _norm_key(s: str) -> str:
    if s is None:
        return ""
    # Normalize by stripping spaces and lowercasing; suitable for simple exact matches
    return "".join(str(s).split()).lower()


def load_cases(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")
    data: List[dict] = []
    if path.suffix.lower() == ".jsonl":
        # First pass: count total lines for progress bar
        total_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
        
        with path.open("r", encoding="utf-8") as f:
            for ln, line in tqdm(enumerate(f, 1), total=total_lines, desc="Loading cases"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {path}:{ln}: {e}") from e
    elif path.suffix.lower() == ".json":
        # First try as a proper JSON document
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError:
            # Fallback: treat as JSONL (many JSON objects) despite .json extension
            # First pass: count total lines for progress bar
            total_lines = 0
            with path.open("r", encoding="utf-8") as f:
                for _ in f:
                    total_lines += 1
            
            with path.open("r", encoding="utf-8") as f:
                for ln, line in tqdm(enumerate(f, 1), total=total_lines, desc="Loading cases"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON/JSONL at {path}:{ln}: {e}") from e
        else:
            if isinstance(obj, list):
                data = obj
            elif isinstance(obj, dict):
                # Try common containers
                for key in ("data", "records", "items", "list", "samples"):
                    if key in obj and isinstance(obj[key], list):
                        data = obj[key]
                        break
                else:
                    # If the JSON is a dict representing a single case
                    data = [obj]
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")
    
    return data


def load_knowledge(jsonl_path: Path) -> Dict[str, dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {jsonl_path}")
    
    # First pass: count total lines for progress bar
    total_lines = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    
    index: Dict[str, dict] = {}
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln, line in tqdm(enumerate(f, 1), total=total_lines, desc="Loading knowledge"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {jsonl_path}:{ln}: {e}") from e
            name = obj.get("Name")
            if not name:
                continue
            index[_norm_key(name)] = obj

        return index


def build_query(rec: dict) -> str:
    parts: List[str] = []
    chief = (rec.get("chief_complaint") or rec.get("主诉") or "").strip()
    if chief:
        parts.append(f"主诉：{chief}\n")
    desc = (rec.get("description") or rec.get("现病史") or "").strip()
    if desc:
        parts.append(f"现病史：{desc}\n")
    det = (rec.get("detection") or rec.get("体格检查") or "").strip()
    if det:
        parts.append(f"体格检查：{det}\n")
    return "".join(parts)


def build_response_from_knowledge(k: dict) -> str:
    name = k.get("Name")
    defin = k.get("Definition")
    perf = k.get("Typical_performance")
    common = k.get("Common_isease")  # note: source key uses single 'd'

    segments: List[str] = []
    if name:
        segments.append(f"名称：{name}。\n")
    if defin:
        segments.append(f"定义：{defin}\n")
    if perf:
        segments.append(f"典型表现：{perf}\n")
    if common:
        segments.append(f"常见疾病：{common}\n")
    return "".join(segments).strip()


def get_field_combinations() -> List[List[str]]:
    """
    Get all possible field combinations for knowledge response generation.
    
    Returns a list of field combinations, where each combination is a list of field names.
    Fields: name, definition, typical_performance, common_disease
    """
    return [
        ["name"],                                           # 名称
        ["definition"],                                     # 定义
        ["typical_performance"],                            # 典型表现
        ["common_disease"],                                # 常见疾病
        ["name", "definition"],                            # 名称+定义
        ["name", "typical_performance"],                   # 名称+典型表现
        ["name", "common_disease"],                        # 名称+常见疾病
        ["name", "definition", "typical_performance"],     # 名称+定义+典型表现
        ["name", "definition", "common_disease"],          # 名称+定义+常见疾病
        ["name", "typical_performance", "common_disease"], # 名称+典型表现+常见疾病
        ["name", "definition", "typical_performance", "common_disease"]  # 名称+定义+典型表现+常见疾病
    ]


def get_field_permutations() -> List[List[str]]:
    """
    Get all possible field permutations for knowledge response generation.
    
    Returns a list of field permutations, where each permutation is a list of field names
    in different orders. All 4! = 24 permutations of the four fields are included.
    Fields: name, definition, typical_performance, common_disease
    """
    base_fields = ["name", "definition", "typical_performance", "common_disease"]
    return [list(perm) for perm in permutations(base_fields)]


def build_response_with_field_combination(k: dict, fields: List[str], strict_mode: bool = False) -> str:
    """
    Build response from knowledge using specified field combination.
    
    Args:
        k: Knowledge dictionary
        fields: List of fields to include in response
        strict_mode: If True, return empty string if any required field is missing
    
    Returns:
        Formatted response string
    """
    name = k.get("Name")
    defin = k.get("Definition")
    perf = k.get("Typical_performance")
    common = k.get("Common_isease")  # note: source key uses single 'd'

    # Check if all required fields are available in strict mode
    if strict_mode:
        field_values = {
            "name": name,
            "definition": defin,
            "typical_performance": perf,
            "common_disease": common
        }
        for field in fields:
            if not field_values.get(field):
                return ""  # Return empty if any required field is missing

    segments: List[str] = []
    
    for field in fields:
        if field == "name" and name:
            segments.append(f"名称：{name}。\n")
        elif field == "definition" and defin:
            segments.append(f"定义：{defin}\n")
        elif field == "typical_performance" and perf:
            segments.append(f"典型表现：{perf}\n")
        elif field == "common_disease" and common:
            segments.append(f"常见疾病：{common}\n")
    
    return "".join(segments).strip()


def pick_syndrome_key(rec: dict) -> Tuple[str, str]:
    # Return (original_value, normalized_key)
    for key in ("norm_syndrome", "syndrome", "证候", "证型"):
        val = rec.get(key)
        if val:
            return str(val), _norm_key(val)
    return "", ""


def tokenize_chinese(text: str) -> List[str]:
    """Tokenize Chinese text using jieba."""
    if not text:
        return []
    # Use jieba for Chinese word segmentation
    return list(jieba.cut(text.strip()))


def build_bm25_index(knowledge_index: Dict[str, dict]) -> Tuple[BM25Okapi, List[str], List[dict]]:
    """Build BM25 index for syndrome knowledge."""
    documents = []
    norm_keys = []
    knowledge_objects = []
    
    print("Processing syndrome knowledge for BM25 indexing...")
    for norm_key, knowledge_obj in tqdm(knowledge_index.items(), desc="Tokenizing documents"):
        # Build searchable text from knowledge
        searchable_parts = []
        name = knowledge_obj.get("Name", "")
        defin = knowledge_obj.get("Definition", "")
        perf = knowledge_obj.get("Typical_performance", "")
        common = knowledge_obj.get("Common_isease", "")
        
        if name:
            searchable_parts.append(name)
        if defin:
            searchable_parts.append(defin)
        if perf:
            searchable_parts.append(perf)
        if common:
            searchable_parts.append(common)
            
        searchable_text = " ".join(searchable_parts)
        tokenized_doc = tokenize_chinese(searchable_text)
        
        documents.append(tokenized_doc)
        norm_keys.append(norm_key)
        knowledge_objects.append(knowledge_obj)
    
    bm25_index = BM25Okapi(documents)
    return bm25_index, norm_keys, knowledge_objects


def get_hard_negatives(
    query: str, 
    correct_norm_key: str,
    bm25_index: BM25Okapi, 
    norm_keys: List[str], 
    knowledge_objects: List[dict],
    num_negatives: int = 5,
    field_combination: List[str] | None = None
) -> List[str]:
    """Get hard negative responses using BM25 similarity."""
    # Tokenize the query
    query_tokens = tokenize_chinese(query)
    
    # Get BM25 scores for all documents
    scores = bm25_index.get_scores(query_tokens)
    
    # Create list of (score, index) pairs, excluding the correct syndrome
    candidates = []
    for i, (score, norm_key) in enumerate(zip(scores, norm_keys)):
        if norm_key != correct_norm_key:  # Exclude correct answer
            candidates.append((score, i))
    
    # Sort by score (descending) and take top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Build responses for top candidates
    hard_negatives = []
    for score, idx in candidates[:num_negatives]:
        knowledge_obj = knowledge_objects[idx]
        if field_combination is not None:
            # Use specified field combination format with strict mode for consistency
            negative_response = build_response_with_field_combination(knowledge_obj, field_combination, strict_mode=True)
        else:
            # Use standard format
            negative_response = build_response_from_knowledge(knowledge_obj)
        if negative_response:  # Only add non-empty responses
            hard_negatives.append(negative_response)
    
    return hard_negatives


def call_custom_embedding_api(
    text: str, 
    api_url: str = "http://0.0.0.0:8000/v1/embeddings",
    model: str = "Qwen3-Embedding-0.6B-finetuned"
) -> np.ndarray:
    """Call custom embedding API to get text embeddings."""
    try:
        payload = {
            "model": model,
            "input": text
        }
        
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        # Extract embedding from API response
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            return np.array(embedding)
        else:
            raise ValueError(f"Invalid API response format: {result}")
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to call embedding API: {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Failed to parse embedding API response: {e}")


def build_custom_embedding_index(
    knowledge_index: Dict[str, dict],
    api_url: str = "http://0.0.0.0:8000/v1/embeddings",
    model: str = "Qwen3-Embedding-0.6B-finetuned"
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Build custom embedding index for syndrome knowledge."""
    embeddings = []
    norm_keys = []
    knowledge_objects = []
    
    print("Processing syndrome knowledge for custom embedding indexing...")
    for norm_key, knowledge_obj in tqdm(knowledge_index.items(), desc="Getting embeddings"):
        # Build searchable text from knowledge  
        searchable_parts = []
        name = knowledge_obj.get("Name", "")
        defin = knowledge_obj.get("Definition", "")
        perf = knowledge_obj.get("Typical_performance", "")
        common = knowledge_obj.get("Common_isease", "")
        
        if name:
            searchable_parts.append(name)
        if defin:
            searchable_parts.append(defin)
        if perf:
            searchable_parts.append(perf)
        if common:
            searchable_parts.append(common)
            
        searchable_text = " ".join(searchable_parts)
        
        # Get embedding from API
        try:
            embedding = call_custom_embedding_api(searchable_text, api_url, model)
            embeddings.append(embedding)
            norm_keys.append(norm_key)
            knowledge_objects.append(knowledge_obj)
        except Exception as e:
            print(f"Warning: Failed to get embedding for '{name}': {e}")
            continue
    
    # Stack all embeddings into a matrix
    if embeddings:
        embedding_matrix = np.vstack(embeddings)
        return embedding_matrix, norm_keys, knowledge_objects
    else:
        raise RuntimeError("No embeddings were successfully generated")


def get_hard_negatives_with_custom_embedding(
    query: str,
    correct_norm_key: str,
    embedding_matrix: np.ndarray,
    norm_keys: List[str],
    knowledge_objects: List[dict],
    api_url: str = "http://0.0.0.0:8000/v1/embeddings",
    model: str = "Qwen3-Embedding-0.6B-finetuned",
    num_negatives: int = 5,
    field_combination: List[str] | None = None
) -> List[str]:
    """Get hard negative responses using custom embedding similarity."""
    # Get query embedding
    query_embedding = call_custom_embedding_api(query, api_url, model)
    
    # Calculate cosine similarities with all knowledge embeddings
    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    knowledge_norms = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    
    # Compute cosine similarities
    similarities = np.dot(knowledge_norms, query_norm)
    
    # Create list of (similarity, index) pairs, excluding the correct syndrome
    candidates = []
    for i, (similarity, norm_key) in enumerate(zip(similarities, norm_keys)):
        if norm_key != correct_norm_key:  # Exclude correct answer
            candidates.append((similarity, i))
    
    # Sort by similarity (descending) and take top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Build responses for top candidates
    hard_negatives = []
    for similarity, idx in candidates[:num_negatives]:
        knowledge_obj = knowledge_objects[idx]
        if field_combination is not None:
            # Use specified field combination format with strict mode for consistency
            negative_response = build_response_with_field_combination(knowledge_obj, field_combination, strict_mode=True)
        else:
            # Use standard format
            negative_response = build_response_from_knowledge(knowledge_obj)
        if negative_response:  # Only add non-empty responses
            hard_negatives.append(negative_response)
    
    return hard_negatives


def get_random_negatives(
    correct_norm_key: str,
    knowledge_index: Dict[str, dict],
    num_negatives: int = 2,
    field_combination: List[str] | None = None
) -> List[str]:
    """Get random negative responses, excluding the correct syndrome."""
    # Get all available knowledge objects excluding the correct one
    available_objects = []
    for norm_key, knowledge_obj in knowledge_index.items():
        if norm_key != correct_norm_key:
            available_objects.append(knowledge_obj)
    
    # Randomly sample without replacement
    if len(available_objects) < num_negatives:
        sampled_objects = available_objects
    else:
        sampled_objects = random.sample(available_objects, num_negatives)
    
    # Build responses for sampled objects
    random_negatives = []
    for knowledge_obj in sampled_objects:
        if field_combination is not None:
            # Use specified field combination format with strict mode for consistency
            negative_response = build_response_with_field_combination(knowledge_obj, field_combination, strict_mode=True)
        else:
            # Use standard format
            negative_response = build_response_from_knowledge(knowledge_obj)
        if negative_response:  # Only add non-empty responses
            random_negatives.append(negative_response)
    
    return random_negatives


def get_hybrid_hard_negatives(
    query: str,
    correct_norm_key: str,
    knowledge_index: Dict[str, dict],
    bm25_index: BM25Okapi,
    bm25_norm_keys: List[str],
    bm25_knowledge_objects: List[dict],
    embedding_matrix: np.ndarray,
    embedding_norm_keys: List[str],
    embedding_knowledge_objects: List[dict],
    api_url: str = "http://0.0.0.0:8000/v1/embeddings",
    model: str = "Qwen3-Embedding-0.6B-finetuned",
    field_combination: List[str] | None = None
) -> List[str]:
    """Get hybrid hard negatives: 2 random + top-3 BM25 + top-3 custom embedding, ensuring uniqueness."""
    all_negatives = []
    used_responses = set()
    
    # 1. Get 2 random negatives
    random_negatives = get_random_negatives(correct_norm_key, knowledge_index, num_negatives=2, field_combination=field_combination)
    for neg in random_negatives:
        if neg not in used_responses:
            all_negatives.append(neg)
            used_responses.add(neg)
    
    # 2. Get top-3 BM25 negatives
    bm25_negatives = get_hard_negatives(
        query=query,
        correct_norm_key=correct_norm_key,
        bm25_index=bm25_index,
        norm_keys=bm25_norm_keys,
        knowledge_objects=bm25_knowledge_objects,
        num_negatives=3,
        field_combination=field_combination
    )
    for neg in bm25_negatives:
        if neg not in used_responses:
            all_negatives.append(neg)
            used_responses.add(neg)
    
    # 3. Get top-3 custom embedding negatives
    embedding_negatives = get_hard_negatives_with_custom_embedding(
        query=query,
        correct_norm_key=correct_norm_key,
        embedding_matrix=embedding_matrix,
        norm_keys=embedding_norm_keys,
        knowledge_objects=embedding_knowledge_objects,
        api_url=api_url,
        model=model,
        num_negatives=3,
        field_combination=field_combination
    )
    for neg in embedding_negatives:
        if neg not in used_responses:
            all_negatives.append(neg)
            used_responses.add(neg)
    
    # If we don't have enough unique negatives, try to get more from available sources
    target_count = 8
    if len(all_negatives) < target_count:
        # Try to get more from BM25 with higher num_negatives
        additional_bm25 = get_hard_negatives(
            query=query,
            correct_norm_key=correct_norm_key,
            bm25_index=bm25_index,
            norm_keys=bm25_norm_keys,
            knowledge_objects=bm25_knowledge_objects,
            num_negatives=10,  # Get more candidates
            field_combination=field_combination
        )
        for neg in additional_bm25:
            if neg not in used_responses and len(all_negatives) < target_count:
                all_negatives.append(neg)
                used_responses.add(neg)
        
        # If still not enough, try embedding
        if len(all_negatives) < target_count:
            additional_embedding = get_hard_negatives_with_custom_embedding(
                query=query,
                correct_norm_key=correct_norm_key,
                embedding_matrix=embedding_matrix,
                norm_keys=embedding_norm_keys,
                knowledge_objects=embedding_knowledge_objects,
                api_url=api_url,
                model=model,
                num_negatives=10,  # Get more candidates
                field_combination=field_combination
            )
            for neg in additional_embedding:
                if neg not in used_responses and len(all_negatives) < target_count:
                    all_negatives.append(neg)
                    used_responses.add(neg)
        
        # If still not enough, use more random
        if len(all_negatives) < target_count:
            additional_random = get_random_negatives(
                correct_norm_key, 
                knowledge_index, 
                num_negatives=target_count - len(all_negatives),
                field_combination=field_combination
            )
            for neg in additional_random:
                if neg not in used_responses and len(all_negatives) < target_count:
                    all_negatives.append(neg)
                    used_responses.add(neg)
    
    return all_negatives[:target_count]  # Ensure we return exactly 8 negatives


def convert(
    cases: List[dict],
    knowledge_index: Dict[str, dict],
    max_samples: int | None = None,
    with_hard_negatives: bool = False,
    with_hard_negatives_custom_embedding: bool = False,
    hybrid: bool = False,
    split_negatives: bool = False,
    field_combinations: bool = False,
    field_permutations: bool = False,
    permutation_argument: bool = False,
    multiplier: int = 5,
    bm25_index: BM25Okapi | None = None,
    norm_keys: List[str] | None = None,
    knowledge_objects: List[dict] | None = None,
    embedding_matrix: np.ndarray | None = None,
    embedding_norm_keys: List[str] | None = None,
    embedding_knowledge_objects: List[dict] | None = None,
    api_url: str = "http://0.0.0.0:8000/v1/embeddings",
    model: str = "Qwen3-Embedding-0.6B-finetuned",
) -> Iterable[dict]:
    # Handle permutation argument mode: zone-based data augmentation
    if permutation_argument:
        # Get all available permutations
        all_permutations = get_field_permutations()
        
        # Generate zone permutations (with cycling if multiplier > 24)
        zone_permutations = []
        for zone_idx in range(multiplier):
            # Use modulo to cycle through permutations if multiplier > 24
            perm_idx = zone_idx % len(all_permutations)
            zone_permutations.append(all_permutations[perm_idx])
        
        # Shuffle the zone permutations to randomize them
        zone_permutations_shuffled = zone_permutations.copy()
        random.shuffle(zone_permutations_shuffled)
        
        total_cases = len(cases)
        if max_samples is not None:
            total_cases = min(total_cases, max_samples)
        
        # Generate samples for each zone
        for zone_idx in range(multiplier):
            current_permutation = zone_permutations_shuffled[zone_idx]
            
            # Process each case in this zone
            for case_idx, rec in enumerate(cases):
                if max_samples is not None and case_idx >= max_samples:
                    break
                    
                query = build_query(rec)
                if not query:
                    ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{case_idx+1}"
                    raise ValueError(f"Empty query after concatenation for {ident}")

                syn_raw, syn_key = pick_syndrome_key(rec)
                if not syn_key:
                    ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{case_idx+1}"
                    raise ValueError(f"Missing syndrome/norm_syndrome for {ident}")

                k = knowledge_index.get(syn_key)
                if not k:
                    ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{case_idx+1}"
                    raise ValueError(f"Syndrome knowledge not found for '{syn_raw}' in {ident}")

                # Use current zone's permutation to build response
                response = build_response_with_field_combination(k, current_permutation, strict_mode=True)
                if not response:
                    # Skip samples with missing fields for this permutation
                    continue
                
                # Check if we need to generate hard negatives with the same field permutation
                if with_hard_negatives:
                    # Generate hard negatives using BM25 with the same field permutation format
                    if bm25_index is None or norm_keys is None or knowledge_objects is None:
                        raise ValueError("BM25 index components required for hard negatives generation")
                    
                    hard_negatives = get_hard_negatives(
                        query=query,
                        correct_norm_key=syn_key,
                        bm25_index=bm25_index,
                        norm_keys=norm_keys,
                        knowledge_objects=knowledge_objects,
                        num_negatives=5,
                        field_combination=current_permutation
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                elif with_hard_negatives_custom_embedding:
                    # Generate hard negatives using custom embedding with the same field permutation format
                    if (embedding_matrix is None or embedding_norm_keys is None or 
                        embedding_knowledge_objects is None):
                        raise ValueError("Custom embedding index components required for hard negatives generation")
                    
                    hard_negatives = get_hard_negatives_with_custom_embedding(
                        query=query,
                        correct_norm_key=syn_key,
                        embedding_matrix=embedding_matrix,
                        norm_keys=embedding_norm_keys,
                        knowledge_objects=embedding_knowledge_objects,
                        api_url=api_url,
                        model=model,
                        num_negatives=5,
                        field_combination=current_permutation
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                elif hybrid:
                    # Generate hybrid hard negatives with the same field permutation format
                    if (bm25_index is None or norm_keys is None or knowledge_objects is None or
                        embedding_matrix is None or embedding_norm_keys is None or 
                        embedding_knowledge_objects is None):
                        raise ValueError("Both BM25 and custom embedding index components required for hybrid hard negatives generation")
                    
                    hard_negatives = get_hybrid_hard_negatives(
                        query=query,
                        correct_norm_key=syn_key,
                        knowledge_index=knowledge_index,
                        bm25_index=bm25_index,
                        bm25_norm_keys=norm_keys,
                        bm25_knowledge_objects=knowledge_objects,
                        embedding_matrix=embedding_matrix,
                        embedding_norm_keys=embedding_norm_keys,
                        embedding_knowledge_objects=embedding_knowledge_objects,
                        api_url=api_url,
                        model=model,
                        field_combination=current_permutation
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                else:
                    # Generate standard format sample for this field permutation (no hard negatives)
                    yield {"query": query, "response": response}
        return
    
    # Original logic for other modes (non-permutation-argument)
    for idx, rec in enumerate(cases, 1):
        if max_samples is not None and idx > max_samples:
            break
        query = build_query(rec)
        if not query:
            ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{idx}"
            raise ValueError(f"Empty query after concatenation for {ident}")

        syn_raw, syn_key = pick_syndrome_key(rec)
        if not syn_key:
            ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{idx}"
            raise ValueError(f"Missing syndrome/norm_syndrome for {ident}")

        k = knowledge_index.get(syn_key)
        if not k:
            ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{idx}"
            raise ValueError(f"Syndrome knowledge not found for '{syn_raw}' in {ident}")

        # Generate responses based on field combinations, field permutations, or standard format
        if field_combinations:
            # Generate multiple training samples with different field combinations
            combinations = get_field_combinations()
            for field_combo in combinations:
                # Use strict mode to ensure all required fields are present
                response = build_response_with_field_combination(k, field_combo, strict_mode=True)
                if not response:
                    # Skip samples with missing fields for this combination
                    continue
                
                # Check if we need to generate hard negatives with the same field combination
                if with_hard_negatives:
                    # Generate hard negatives using BM25 with the same field combination format
                    if bm25_index is None or norm_keys is None or knowledge_objects is None:
                        raise ValueError("BM25 index components required for hard negatives generation")
                    
                    hard_negatives = get_hard_negatives(
                        query=query,
                        correct_norm_key=syn_key,
                        bm25_index=bm25_index,
                        norm_keys=norm_keys,
                        knowledge_objects=knowledge_objects,
                        num_negatives=5,
                        field_combination=field_combo
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                elif with_hard_negatives_custom_embedding:
                    # Generate hard negatives using custom embedding with the same field combination format
                    if (embedding_matrix is None or embedding_norm_keys is None or 
                        embedding_knowledge_objects is None):
                        raise ValueError("Custom embedding index components required for hard negatives generation")
                    
                    hard_negatives = get_hard_negatives_with_custom_embedding(
                        query=query,
                        correct_norm_key=syn_key,
                        embedding_matrix=embedding_matrix,
                        norm_keys=embedding_norm_keys,
                        knowledge_objects=embedding_knowledge_objects,
                        api_url=api_url,
                        model=model,
                        num_negatives=5,
                        field_combination=field_combo
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                elif hybrid:
                    # Generate hybrid hard negatives with the same field combination format
                    if (bm25_index is None or norm_keys is None or knowledge_objects is None or
                        embedding_matrix is None or embedding_norm_keys is None or 
                        embedding_knowledge_objects is None):
                        raise ValueError("Both BM25 and custom embedding index components required for hybrid hard negatives generation")
                    
                    hard_negatives = get_hybrid_hard_negatives(
                        query=query,
                        correct_norm_key=syn_key,
                        knowledge_index=knowledge_index,
                        bm25_index=bm25_index,
                        bm25_norm_keys=norm_keys,
                        bm25_knowledge_objects=knowledge_objects,
                        embedding_matrix=embedding_matrix,
                        embedding_norm_keys=embedding_norm_keys,
                        embedding_knowledge_objects=embedding_knowledge_objects,
                        api_url=api_url,
                        model=model,
                        field_combination=field_combo
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                else:
                    # Generate standard format sample for this field combination (no hard negatives)
                    yield {"query": query, "response": response}
        elif field_permutations:
            # Generate multiple training samples with different field permutations
            permutations_list = get_field_permutations()
            for field_permutation in permutations_list:
                # Use strict mode to ensure all required fields are present
                response = build_response_with_field_combination(k, field_permutation, strict_mode=True)
                if not response:
                    # Skip samples with missing fields for this permutation
                    continue
                
                # Check if we need to generate hard negatives with the same field permutation
                if with_hard_negatives:
                    # Generate hard negatives using BM25 with the same field permutation format
                    if bm25_index is None or norm_keys is None or knowledge_objects is None:
                        raise ValueError("BM25 index components required for hard negatives generation")
                    
                    hard_negatives = get_hard_negatives(
                        query=query,
                        correct_norm_key=syn_key,
                        bm25_index=bm25_index,
                        norm_keys=norm_keys,
                        knowledge_objects=knowledge_objects,
                        num_negatives=5,
                        field_combination=field_permutation
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                elif with_hard_negatives_custom_embedding:
                    # Generate hard negatives using custom embedding with the same field permutation format
                    if (embedding_matrix is None or embedding_norm_keys is None or 
                        embedding_knowledge_objects is None):
                        raise ValueError("Custom embedding index components required for hard negatives generation")
                    
                    hard_negatives = get_hard_negatives_with_custom_embedding(
                        query=query,
                        correct_norm_key=syn_key,
                        embedding_matrix=embedding_matrix,
                        norm_keys=embedding_norm_keys,
                        knowledge_objects=embedding_knowledge_objects,
                        api_url=api_url,
                        model=model,
                        num_negatives=5,
                        field_combination=field_permutation
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                elif hybrid:
                    # Generate hybrid hard negatives with the same field permutation format
                    if (bm25_index is None or norm_keys is None or knowledge_objects is None or
                        embedding_matrix is None or embedding_norm_keys is None or 
                        embedding_knowledge_objects is None):
                        raise ValueError("Both BM25 and custom embedding index components required for hybrid hard negatives generation")
                    
                    hard_negatives = get_hybrid_hard_negatives(
                        query=query,
                        correct_norm_key=syn_key,
                        knowledge_index=knowledge_index,
                        bm25_index=bm25_index,
                        bm25_norm_keys=norm_keys,
                        bm25_knowledge_objects=knowledge_objects,
                        embedding_matrix=embedding_matrix,
                        embedding_norm_keys=embedding_norm_keys,
                        embedding_knowledge_objects=embedding_knowledge_objects,
                        api_url=api_url,
                        model=model,
                        field_combination=field_permutation
                    )
                    
                    if split_negatives:
                        # Split into separate samples, each with single rejected_response
                        for negative in hard_negatives:
                            yield {
                                "query": query,
                                "response": response,
                                "rejected_response": [negative]
                            }
                    else:
                        # Original format with all negatives in one sample
                        yield {
                            "query": query,
                            "response": response,
                            "rejected_response": hard_negatives
                        }
                else:
                    # Generate standard format sample for this field permutation (no hard negatives)
                    yield {"query": query, "response": response}
        elif with_hard_negatives:
            # Generate hard negatives using BM25
            response = build_response_from_knowledge(k)
            if not response:
                ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{idx}"
                raise ValueError(f"Built empty response from knowledge for {ident}")
                
            if bm25_index is None or norm_keys is None or knowledge_objects is None:
                raise ValueError("BM25 index components required for hard negatives generation")
            
            hard_negatives = get_hard_negatives(
                query=query,
                correct_norm_key=syn_key,
                bm25_index=bm25_index,
                norm_keys=norm_keys,
                knowledge_objects=knowledge_objects,
                num_negatives=5
            )
            
            if split_negatives:
                # Split into separate samples, each with single rejected_response
                for negative in hard_negatives:
                    yield {
                        "query": query,
                        "response": response,
                        "rejected_response": [negative]
                    }
            else:
                # Original format with all negatives in one sample
                yield {
                    "query": query,
                    "response": response,
                    "rejected_response": hard_negatives
                }
        elif with_hard_negatives_custom_embedding:
            # Generate hard negatives using custom embedding
            if (embedding_matrix is None or embedding_norm_keys is None or 
                embedding_knowledge_objects is None):
                raise ValueError("Custom embedding index components required for hard negatives generation")
            
            hard_negatives = get_hard_negatives_with_custom_embedding(
                query=query,
                correct_norm_key=syn_key,
                embedding_matrix=embedding_matrix,
                norm_keys=embedding_norm_keys,
                knowledge_objects=embedding_knowledge_objects,
                api_url=api_url,
                model=model,
                num_negatives=5
            )
            
            if split_negatives:
                # Split into separate samples, each with single rejected_response
                for negative in hard_negatives:
                    yield {
                        "query": query,
                        "response": response,
                        "rejected_response": [negative]
                    }
            else:
                # Original format with all negatives in one sample
                yield {
                    "query": query,
                    "response": response,
                    "rejected_response": hard_negatives
                }
        elif hybrid:
            # Generate hybrid hard negatives (2 random + top-3 BM25 + top-3 custom embedding)
            if (bm25_index is None or norm_keys is None or knowledge_objects is None or
                embedding_matrix is None or embedding_norm_keys is None or 
                embedding_knowledge_objects is None):
                raise ValueError("Both BM25 and custom embedding index components required for hybrid hard negatives generation")
            
            hard_negatives = get_hybrid_hard_negatives(
                query=query,
                correct_norm_key=syn_key,
                knowledge_index=knowledge_index,
                bm25_index=bm25_index,
                bm25_norm_keys=norm_keys,
                bm25_knowledge_objects=knowledge_objects,
                embedding_matrix=embedding_matrix,
                embedding_norm_keys=embedding_norm_keys,
                embedding_knowledge_objects=embedding_knowledge_objects,
                api_url=api_url,
                model=model
            )
            
            if split_negatives:
                # Split into separate samples, each with single rejected_response
                for negative in hard_negatives:
                    yield {
                        "query": query,
                        "response": response,
                        "rejected_response": [negative]
                    }
            else:
                # Original format with all negatives in one sample
                yield {
                    "query": query,
                    "response": response,
                    "rejected_response": hard_negatives
                }
        else:
            # Standard format (no field combinations, no hard negatives)
            response = build_response_from_knowledge(k)
            if not response:
                ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{idx}"
                raise ValueError(f"Built empty response from knowledge for {ident}")
            yield {"query": query, "response": response}


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Convert cases to InfoNCE JSONL")
    p.add_argument("--input", required=True, type=Path, help="Path to cases JSON/JSONL (e.g., train.json)")
    p.add_argument(
        "--knowledge",
        type=Path,
        default=Path("../data/raw_data/TCM_SD/syndrome_knowledge.jsonl"),
        help="Path to syndrome knowledge JSONL",
    )
    p.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    p.add_argument("--with-hard-negatives", action="store_true", 
                   help="Generate hard negatives using BM25, outputs format with rejected_response")
    p.add_argument("--with-hard-negatives-custom-embedding", action="store_true",
                   help="Generate hard negatives using custom embedding API, outputs format with rejected_response")
    p.add_argument("--hybrid", action="store_true",
                   help="Generate hybrid hard negatives (2 random + top-3 BM25 + top-3 custom embedding), outputs format with rejected_response")
    p.add_argument("--split-negatives", action="store_true",
                   help="Split each sample with multiple rejected_response into separate samples, each with single rejected_response")
    p.add_argument("--field-combinations", action="store_true",
                   help="Generate multiple response formats with different field combinations")
    p.add_argument("--field-permutations", action="store_true",
                   help="Generate multiple response formats with different field permutations (all 24 orderings)")
    p.add_argument("--permutation-argument", action="store_true",
                   help="Generate augmented data using zone-based field permutations with multiplier")
    p.add_argument("--multiplier", type=int, default=5,
                   help="Multiplier for data augmentation (default: 5). Only used with --permutation-argument")
    args = p.parse_args(argv)

    cases = load_cases(args.input)
    knowledge = load_knowledge(args.knowledge)
    
    # Check for conflicting hard negative options (excluding field-combinations and field-permutations)
    hard_negative_options = [args.with_hard_negatives, args.with_hard_negatives_custom_embedding, args.hybrid]
    hard_negative_count = sum(hard_negative_options)
    if hard_negative_count > 1:
        raise ValueError("Cannot use multiple hard negative generation options simultaneously (--with-hard-negatives, --with-hard-negatives-custom-embedding, --hybrid)")
    
    # Check for conflicting field options
    if args.field_combinations and args.field_permutations:
        raise ValueError("Cannot use both --field-combinations and --field-permutations simultaneously")
    
    if args.permutation_argument and (args.field_combinations or args.field_permutations):
        raise ValueError("Cannot use --permutation-argument with --field-combinations or --field-permutations")
    
    # Check if split-negatives is used without hard negatives
    if args.split_negatives and not any(hard_negative_options) and not args.field_combinations and not args.field_permutations and not args.permutation_argument:
        raise ValueError("--split-negatives can only be used with hard negative generation options (--with-hard-negatives, --with-hard-negatives-custom-embedding, --hybrid) or --field-combinations or --field-permutations or --permutation-argument")
    
    # Adjust output filename based on options
    output_path = args.output
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    suffixes = []
    
    # Add field combinations, permutations, or permutation-argument suffix
    if args.field_combinations:
        suffixes.append("field_combinations")
    elif args.field_permutations:
        suffixes.append("field_permutations")
    elif args.permutation_argument:
        suffixes.append(f"permutation_argument_x{args.multiplier}")
    
    # Add hard negatives suffix
    if args.with_hard_negatives:
        suffixes.append("with_hard_negatives")
    elif args.with_hard_negatives_custom_embedding:
        suffixes.append("with_hard_negatives_custom_embedding")
    elif args.hybrid:
        suffixes.append("hybrid")
    
    # Add split suffix if applicable
    if args.split_negatives and any([args.with_hard_negatives, args.with_hard_negatives_custom_embedding, args.hybrid]):
        suffixes.append("split")
    
    # Build final filename
    if suffixes:
        suffix = "_" + "_".join(suffixes)
        output_path = output_path.parent / f"{output_stem}{suffix}{output_suffix}"
    
    # Build BM25 index if needed
    bm25_index, norm_keys, knowledge_objects = None, None, None
    if (args.with_hard_negatives or args.hybrid) or ((args.field_combinations or args.field_permutations or args.permutation_argument) and (args.with_hard_negatives or args.hybrid)):
        print("Building BM25 index for hard negatives generation...")
        bm25_index, norm_keys, knowledge_objects = build_bm25_index(knowledge)
        print(f"Built BM25 index with {len(norm_keys)} syndrome entries")
    
    # Build custom embedding index if needed
    embedding_matrix, embedding_norm_keys, embedding_knowledge_objects = None, None, None
    if (args.with_hard_negatives_custom_embedding or args.hybrid) or ((args.field_combinations or args.field_permutations or args.permutation_argument) and (args.with_hard_negatives_custom_embedding or args.hybrid)):
        print("Building custom embedding index for hard negatives generation...")
        embedding_matrix, embedding_norm_keys, embedding_knowledge_objects = build_custom_embedding_index(knowledge)
        print(f"Built custom embedding index with {len(embedding_norm_keys)} syndrome entries")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    
    # Determine total samples for progress bar
    total_samples = len(cases)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)
    
    # Adjust expected output count for field combinations, permutations, or permutation-argument
    if args.field_combinations:
        # Each case can generate up to 11 different response formats
        estimated_total = total_samples * 11
        print(f"Converting {total_samples} cases with field combinations (estimated {estimated_total} output samples)...")
        # Use None for progress bar total since the exact count depends on available fields
        progress_total = None
    elif args.field_permutations:
        # Each case can generate up to 24 different response formats (all permutations)
        estimated_total = total_samples * 24
        print(f"Converting {total_samples} cases with field permutations (estimated {estimated_total} output samples)...")
        # Use None for progress bar total since the exact count depends on available fields
        progress_total = None
    elif args.permutation_argument:
        # Each case will be replicated multiplier times, each zone using a different permutation
        estimated_total = total_samples * args.multiplier
        print(f"Converting {total_samples} cases with permutation argument x{args.multiplier} (estimated {estimated_total} output samples)...")
        progress_total = estimated_total
    else:
        print(f"Converting {total_samples} cases...")
        progress_total = total_samples
    
    with output_path.open("w", encoding="utf-8") as out:
        converted_items = convert(
            cases, 
            knowledge, 
            max_samples=args.max_samples,
            with_hard_negatives=args.with_hard_negatives,
            with_hard_negatives_custom_embedding=args.with_hard_negatives_custom_embedding,
            hybrid=args.hybrid,
            split_negatives=args.split_negatives,
            field_combinations=args.field_combinations,
            field_permutations=args.field_permutations,
            permutation_argument=args.permutation_argument,
            multiplier=args.multiplier,
            bm25_index=bm25_index,
            norm_keys=norm_keys,
            knowledge_objects=knowledge_objects,
            embedding_matrix=embedding_matrix,
            embedding_norm_keys=embedding_norm_keys,
            embedding_knowledge_objects=embedding_knowledge_objects
        )
        
        for item in tqdm(converted_items, total=progress_total, desc="Converting cases"):
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    
    # Build format description
    format_parts = []
    
    if args.field_combinations:
        format_parts.append("field combinations")
    elif args.field_permutations:
        format_parts.append("field permutations")
    elif args.permutation_argument:
        format_parts.append(f"permutation argument x{args.multiplier}")
    
    if args.with_hard_negatives:
        format_parts.append("hard negatives (BM25)")
    elif args.with_hard_negatives_custom_embedding:
        format_parts.append("hard negatives (custom embedding)")
    elif args.hybrid:
        format_parts.append("hybrid hard negatives")
    
    if args.split_negatives and any([args.with_hard_negatives, args.with_hard_negatives_custom_embedding, args.hybrid]):
        format_parts.append("split into individual samples")
    
    if format_parts:
        format_desc = "with " + " + ".join(format_parts)
    else:
        format_desc = "standard"
    
    print(f"Wrote {written} samples ({format_desc}) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
