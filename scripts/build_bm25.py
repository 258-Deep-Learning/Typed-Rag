# scripts/build_bm25.py
import json, re, joblib, os
from pathlib import Path
import typer

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def tok(s: str): return TOKEN_RE.findall(s.lower())

app = typer.Typer()

@app.command()
def main(input: str = "data/passages.jsonl",
         out: str = "indexes/bm25_rank.pkl",
         meta_out: str = "indexes/meta.jsonl"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ids, titles, urls, texts, tokens = [], [], [], [], []
    with open(input, "r") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            ids.append(j.get("id") or str(len(ids)))
            titles.append(j.get("title") or "")
            urls.append(j.get("url") or "")
            t = j.get("chunk_text") or ""
            texts.append(t)
            tokens.append(tok(t))

    # persist tokens + metadata (we'll build BM25 at load time for portability)
    joblib.dump(
        {"ids": ids, "titles": titles, "urls": urls, "texts": texts, "tokens": tokens},
        out
    )

    # also write meta.jsonl for other indexes (same order == same idx mapping)
    with open(meta_out, "w") as w:
        for i in range(len(ids)):
            w.write(json.dumps({
                "id": ids[i], "title": titles[i], "url": urls[i], "chunk_text": texts[i]
            }) + "\n")
    print(f"Saved: {out} and {meta_out}  (docs={len(ids)})")

if __name__ == "__main__":
    app()
