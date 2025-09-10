import requests, numpy as np

HOST = "http://localhost:8000/v1/embeddings"
MODEL = "Qwen3-Embedding-0.6B-finetuned"
# MODEL = "Qwen3-Embedding-0.6B-base"

texts = ["濕熱下注證", "肛內腫物外脫伴肛痛"]
embs = []
for t in texts:
    r = requests.post(HOST, json={"model": MODEL, "input": t})
    r.raise_for_status()
    embs.append(r.json()["data"][0]["embedding"])

embs = np.array(embs, dtype=np.float32)

# L2-normalize
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

# cosine similarity
cos = float(embs[0] @ embs[1])

# L2 distance
l2 = float(np.linalg.norm(embs[0] - embs[1]))

print(f"cosine similarity = {cos:.4f}")
print(f"L2 distance       = {l2:.4f}")