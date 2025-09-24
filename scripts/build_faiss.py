# scripts/build_faiss.py
import json, os
from pathlib import Path
import numpy as np
import faiss, typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

app = typer.Typer()

@app.command()
def main(input: str = "data/passages.jsonl",
         index_dir: str = "indexes/faiss_bge_small",
         model_name: str = "BAAI/bge-small-en-v1.5",
         batch_size: int = 128):
    os.makedirs(index_dir, exist_ok=True)
    texts, metas = [], []
    with open(input, "r") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            texts.append(j.get("chunk_text") or "")
            metas.append({
                "id": j.get("id"), "title": j.get("title"),
                "url": j.get("url"), "chunk_text": j.get("chunk_text")
            })

    model = SentenceTransformer(model_name)
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        E = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
        embs.append(E.astype("float32"))
    X = np.vstack(embs) if embs else np.zeros((0, 384), dtype="float32")
    d = X.shape[1] if X.size else 384

    # Simple & robust: flat index with inner product (cosine because normalized)
    index = faiss.IndexFlatIP(d)
    if X.size:
        index.add(X)

    faiss.write_index(index, os.path.join(index_dir, "index.flatip"))
    with open(os.path.join(index_dir, "meta.jsonl"), "w") as w:
        for m in metas:
            w.write(json.dumps(m) + "\n")
    with open(os.path.join(index_dir, "model.txt"), "w") as w:
        w.write(model_name + "\n")
    print(f"FAISS index built: {index.ntotal} vectors, dim={d}")

if __name__ == "__main__":
    app()
