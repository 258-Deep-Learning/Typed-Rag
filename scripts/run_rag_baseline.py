import json, time, typer, os, sys
from typing import Optional, Literal
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import retrieval module
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.hybrid import HybridRetriever

app = typer.Typer(help="RAG baseline: retrieve passages and answer with LLM.")

def call_llm_with_context(client, model: str, question: str, passages: list, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call LLM with retrieved context passages."""
    
    # Format context from passages
    context_parts = []
    for i, passage in enumerate(passages, 1):
        title = passage.get('title', 'Unknown')
        text = passage.get('text', '')
        url = passage.get('url', '')
        context_parts.append(f"[{i}] {title}\n{text}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the following question using the provided context. Be specific and cite relevant information from the sources.

Context:
{context}

Question: {question}

Answer:"""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be factual and specific."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

@app.command()
def main(
    input_path: str = "data/dev100.jsonl",
    out_path: str = "runs/rag.jsonl",
    indexes_dir: str = "indexes",
    mode: Literal["bm25", "faiss", "hybrid"] = "hybrid",
    k: int = 5,
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
    if not os.path.isabs(indexes_dir):
        indexes_dir = str(project_root / indexes_dir)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Initialize retriever and LLM client
    retriever = HybridRetriever(
        bm25_path=f"{indexes_dir}/bm25_rank.pkl",
        faiss_dir=f"{indexes_dir}/faiss_bge_small",
        meta_path=f"{indexes_dir}/meta.jsonl"
    )
    client = OpenAI()
    
    # Check retriever health
    health = retriever.health_check()
    print("Retrieval system health:", health)
    if not all(health.values()):
        print("⚠️  Warning: Some retrieval components are not loaded!")

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

    mode_file = "a" if os.path.exists(out_path) else "w"
    total = len(rows)
    
    with open(out_path, mode_file) as w:
        try:
            for r in tqdm(rows, total=total, desc=f"RAG-{mode}"):
                qid = r.get("question_id")
                q = r.get("question_text","").strip()
                
                # Retrieval step
                retrieval_start = time.time()
                try:
                    passages = retriever.retrieve(query=q, k=k, mode=mode)
                    retrieval_latency = (time.time() - retrieval_start) * 1000
                except Exception as e:
                    print(f"Retrieval error for {qid}: {e}")
                    passages = []
                    retrieval_latency = 0
                
                # LLM step
                llm_start = time.time()
                try:
                    if passages:
                        ans = call_llm_with_context(client, model, q, passages)
                    else:
                        # Fallback to LLM-only if no passages retrieved
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role":"system","content":"You are a concise, factual assistant. If the answer isn't clear, say you don't know."},
                                     {"role":"user","content":q}],
                            temperature=0.2,
                            max_tokens=256,
                        )
                        ans = resp.choices[0].message.content.strip()
                    
                    llm_latency = (time.time() - llm_start) * 1000
                    total_latency = retrieval_latency + llm_latency
                    
                    w.write(json.dumps({
                        "question_id": qid,
                        "question_text": q,
                        "model": model,
                        "retrieval_mode": mode,
                        "passages": passages,
                        "answer": ans,
                        "retrieval_latency_ms": round(retrieval_latency, 2),
                        "llm_latency_ms": round(llm_latency, 2),
                        "total_latency_ms": round(total_latency, 2)
                    }, ensure_ascii=False) + "\n")
                    w.flush()
                    
                except Exception as e:
                    w.write(json.dumps({
                        "question_id": qid, 
                        "question_text": q, 
                        "error": str(e),
                        "retrieval_latency_ms": round(retrieval_latency, 2) if 'retrieval_latency' in locals() else 0
                    }, ensure_ascii=False) + "\n")
                    w.flush()
                    
        except KeyboardInterrupt:
            print("\nInterrupted — partial results saved to", out_path)
    
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    app()
