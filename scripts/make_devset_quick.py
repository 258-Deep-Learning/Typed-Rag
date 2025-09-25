# scripts/make_devset_quick.py
import json, random, re
from pathlib import Path
import typer

app = typer.Typer(help="Create a quick 100-question dev set from your Wikipedia corpus.")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

OCCUPATION_HINTS = {
    "actor","actress","singer","rapper","politician","president","prime minister",
    "footballer","cricketer","basketball","player","scientist","physicist","chemist",
    "mathematician","economist","author","writer","poet","artist","engineer","developer"
}

def first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = SENT_SPLIT.split(text)
    return parts[0].strip()

def make_question(title: str, first_sent: str) -> str:
    ls = first_sent.lower()

    # geography-ish
    if any(x in ls for x in [" is a city", " is the capital", " city in ", " town in ", " located in "]):
        return f"Where is {title}?"

    # person-ish
    if (" was born" in ls) or (" is a " in ls) or (" is an " in ls):
        if any(h in ls for h in OCCUPATION_HINTS) or "born" in ls:
            return f"Who is {title}?"

    # lists
    if title.startswith("List of"):
        return f"What is the {title.lower()}?"

    # default aboutness
    if " is a " in ls or " is an " in ls:
        return f"What is {title}?"
    return f"What can you tell me about {title}?"

@app.command()
def main(
    meta_path: str = "indexes/meta.jsonl",
    out_path: str = "data/dev100.jsonl",
    n: int = 100,
    seed: int = 42,
):
    meta_file = Path(meta_path)
    assert meta_file.exists(), f"Not found: {meta_file}"

    # Collapse chunks -> one entry per unique title (take the first seen chunk for a title)
    by_title = {}
    with meta_file.open() as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            t = j.get("title") or ""
            if not t: continue
            if t not in by_title:
                by_title[t] = {
                    "title": t,
                    "url": j.get("url") or "",
                    "first_sent": first_sentence(j.get("chunk_text") or "")
                }

    titles = list(by_title.keys())
    if not titles:
        raise RuntimeError("No titles found in meta.jsonl")

    random.seed(seed)
    sample = random.sample(titles, min(n, len(titles)))

    records = []
    for i, t in enumerate(sample, start=1):
        info = by_title[t]
        q = make_question(t, info["first_sent"])
        records.append({
            "question_id": f"dev{i:03d}",
            "question_text": q,
            "source_title": t,
            "source_url": info["url"],
        })

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # quick preview to stdout
    print(f"Wrote {len(records)} questions to {out_file}")
    for r in records[:5]:
        print(" -", r["question_text"])

if __name__ == "__main__":
    app()
