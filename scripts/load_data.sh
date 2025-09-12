
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

sleep 10

python convert_to_infonce.py \
  --input ../data/raw_data/TCM_SD/dev.jsonl \
  --knowledge ../data/raw_data/TCM_SD/syndrome_knowledge.jsonl \
  --output ../data/dev.jsonl \
  --with-hard-negatives-custom-embedding