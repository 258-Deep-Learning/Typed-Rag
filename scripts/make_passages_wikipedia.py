# scripts/make_passages_wikipedia.py
import json, typer, re
from datasets import load_dataset
from tqdm import tqdm

app = typer.Typer()
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def chunk_by_tokens(text: str, chunk_tokens: int, stride_tokens: int):
    toks = text.split()
    n = len(toks)
    i = 0
    while i < n:
        j = min(n, i + chunk_tokens)
        chunk = " ".join(toks[i:j]).strip()
        if chunk:
            yield chunk
        if j == n:
            break
        step = max(1, chunk_tokens - stride_tokens)
        i += step

@app.command()
def main(
    snapshot: str = "20220301",    # HF snapshot id (e.g., 20220301, 20231101)
    lang: str = "en",              # "en" or "simple" for Simple English
    out: str = "data/passages.jsonl",
    max_pages: int = 5000,         # cap for quick runs; set 0 to process all
    chunk_tokens: int = 200,
    stride_tokens: int = 60,
    streaming: bool = True,
):
    name = f"{snapshot}.{lang}"
    ds = load_dataset("wikimedia/wikipedia", f"{snapshot}.{lang}", split="train", streaming=streaming)

    count = 0
    wrote = 0
    with open(out, "w") as w:
        for ex in tqdm(ds, desc=f"Reading wikipedia:{name}"):
            title = ex.get("title") or ""
            text = ex.get("text") or ""
            page_id = str(ex.get("id", count))
            url = f"https://{lang}.wikipedia.org/wiki/" + title.replace(" ", "_")

            idx = 0
            for chunk in chunk_by_tokens(text, chunk_tokens, stride_tokens):
                _id = f"{page_id}_{idx}"
                w.write(json.dumps({
                    "id": _id, "title": title, "url": url, "chunk_text": chunk
                }) + "\n")
                idx += 1
                wrote += 1

            count += 1
            if max_pages and count >= max_pages:
                break

    print(f"Processed pages: {count}, wrote chunks: {wrote}, to {out}")

if __name__ == "__main__":
    app()
