
# With standard format
python convert_to_infonce.py \
  --input ../data/raw_data/TCM_SD/train.jsonl \
  --knowledge ../data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
  --output ../data/train.jsonl

# With BM25 hard negatives
python convert_to_infonce.py \
  --input ../data/raw_data/TCM_SD/train.jsonl \
  --knowledge ../data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
  --output ../data/train.jsonl \
  --with-hard-negatives

# With custom embedding hard negatives
python convert_to_infonce.py \
  --input ../data/raw_data/TCM_SD/train.jsonl \
  --knowledge ../data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
  --output ../data/train.jsonl \
  --with-hard-negatives-custom-embedding

# With field combinations and hard negatives
python scripts/convert_to_infonce.py \
  --input ./data/raw_data/TCM_SD/train.jsonl \
  --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
  --output ./data/train.jsonl \
  --field-combinations \
  --with-hard-negatives

# With permutation argument and hard negatives
python scripts/convert_to_infonce.py \
  --input ./data/raw_data/TCM_SD/train.jsonl \
  --knowledge ./data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
  --output ./data/train.jsonl \
  --with-hard-negatives \
  --permutation-argument \
  --multiplier 5
