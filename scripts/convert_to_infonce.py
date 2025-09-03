#!/usr/bin/env python3
"""
Convert case records into InfoNCE JSONL for embedding fine-tuning.

Input cases can be JSON (array or object with a list field) or JSONL.
For each record, the script builds:

  - query: "主诉：{chief_complaint}现病史：{description}体格检查：{detection}"
  - response: syndrome knowledge matched by {norm_syndrome|syndrome} from knowledge JSONL

Usage examples:
  python scripts/convert_to_infonce.py \
    --input data/raw_data/TCM_SD/train.jsonl \
    --knowledge data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
    --output data/train.jsonl

Options:
  --max-samples N      Limit number of processed samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple


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
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
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
            with path.open("r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
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
    index: Dict[str, dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
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


def convert(
    cases: List[dict],
    knowledge_index: Dict[str, dict],
    max_samples: int | None = None,
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

        yield {"query": query, "response": response}


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Convert cases to InfoNCE JSONL")
    p.add_argument("--input", required=True, type=Path, help="Path to cases JSON/JSONL (e.g., train.json)")
    p.add_argument(
        "--knowledge",
        type=Path,
        default=Path("data/raw_data/TCM_SD/syndrome_knowledge.jsonl"),
        help="Path to syndrome knowledge JSONL",
    )
    p.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = p.parse_args(argv)

    cases = load_cases(args.input)
    knowledge = load_knowledge(args.knowledge)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in convert(cases, knowledge, max_samples=args.max_samples):
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
