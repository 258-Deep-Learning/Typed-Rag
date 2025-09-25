import json, time, typer, os, math
from typing import Optional
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

app = typer.Typer(help="LLM-only baseline: no retrieval, just answer the question.")
SYS = "You are a concise, factual assistant. If the answer isn't clear, say you don't know."

def call_llm(client, model: str, question: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYS},{"role":"user","content":question}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

@app.command()
def main(
    input_path: str = "data/dev100.jsonl",
    out_path: str = "runs/llm_only.jsonl",
    model: str = "gpt-4o-mini",
    max_items: int = 0,   # 0 = all
):
    # Get the project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(input_path):
        input_path = str(project_root / input_path)
    if not os.path.isabs(out_path):
        out_path = str(project_root / out_path)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    client = OpenAI()

    rows = [json.loads(l) for l in open(input_path)]
    if max_items:
        rows = rows[:max_items]

    # Resume support - check what's already done
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try: 
                done.add(json.loads(l)["question_id"])
            except: 
                pass
    rows = [r for r in rows if r.get("question_id") not in done]

    mode = "a" if os.path.exists(out_path) else "w"
    total = len(rows)
    with open(out_path, mode) as w:
        try:
            for r in tqdm(rows, total=total, desc="LLM-only"):
                qid = r.get("question_id")
                q = r.get("question_text","").strip()
                t0 = time.time()
                try:
                    ans = call_llm(client, model, q)
                    dt = (time.time()-t0)*1000
                    w.write(json.dumps({
                        "question_id": qid,
                        "question_text": q,
                        "model": model,
                        "prompt": q,
                        "answer": ans,
                        "latency_ms": round(dt,2)
                    }, ensure_ascii=False) + "\n")
                    w.flush()
                except Exception as e:
                    w.write(json.dumps({
                        "question_id": qid, "question_text": q, "error": str(e)
                    }, ensure_ascii=False) + "\n")
                    w.flush()
        except KeyboardInterrupt:
            print("\nInterrupted — partial results saved to", out_path)
    
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    app()
