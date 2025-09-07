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

Options:
  --max-samples N          Limit number of processed samples
  --with-hard-negatives   Generate hard negatives using BM25; outputs format with rejected_response

Output formats:
  Standard: {"query": "...", "response": "..."}
  Hard negatives: {"query": "...", "response": "...", "rejected_response": ["...", "..."]}

Dependencies:
  - tqdm: For progress bars during processing
  - jieba: For Chinese text tokenization (with --with-hard-negatives)
  - rank-bm25: For BM25 similarity search (with --with-hard-negatives)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import jieba
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
    num_negatives: int = 5
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
        negative_response = build_response_from_knowledge(knowledge_obj)
        if negative_response:  # Only add non-empty responses
            hard_negatives.append(negative_response)
    
    return hard_negatives


def convert(
    cases: List[dict],
    knowledge_index: Dict[str, dict],
    max_samples: int | None = None,
    with_hard_negatives: bool = False,
    bm25_index: BM25Okapi | None = None,
    norm_keys: List[str] | None = None,
    knowledge_objects: List[dict] | None = None,
) -> Iterable[dict]:
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

        response = build_response_from_knowledge(k)
        if not response:
            ident = rec.get("lcd_id") or rec.get("user_id") or rec.get("id") or f"record#{idx}"
            raise ValueError(f"Built empty response from knowledge for {ident}")

        if with_hard_negatives:
            # Generate hard negatives using BM25
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
            
            yield {
                "query": query,
                "response": response,
                "rejected_response": hard_negatives
            }
        else:
            # Original format
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
    args = p.parse_args(argv)

    cases = load_cases(args.input)
    knowledge = load_knowledge(args.knowledge)
    
    # Adjust output filename if using hard negatives
    output_path = args.output
    if args.with_hard_negatives:
        output_stem = output_path.stem
        output_suffix = output_path.suffix
        output_path = output_path.parent / f"{output_stem}_with_hard_negatives{output_suffix}"
    
    # Build BM25 index if needed
    bm25_index, norm_keys, knowledge_objects = None, None, None
    if args.with_hard_negatives:
        print("Building BM25 index for hard negatives generation...")
        bm25_index, norm_keys, knowledge_objects = build_bm25_index(knowledge)
        print(f"Built BM25 index with {len(norm_keys)} syndrome entries")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    
    # Determine total samples for progress bar
    total_samples = len(cases)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)
    
    print(f"Converting {total_samples} cases...")
    with output_path.open("w", encoding="utf-8") as out:
        converted_items = convert(
            cases, 
            knowledge, 
            max_samples=args.max_samples,
            with_hard_negatives=args.with_hard_negatives,
            bm25_index=bm25_index,
            norm_keys=norm_keys,
            knowledge_objects=knowledge_objects
        )
        
        for item in tqdm(converted_items, total=total_samples, desc="Converting cases"):
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    
    format_desc = "with hard negatives" if args.with_hard_negatives else "standard"
    print(f"Wrote {written} samples ({format_desc}) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
