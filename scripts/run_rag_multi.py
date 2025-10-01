#!/usr/bin/env python3
"""
Multi-backend RAG baseline with support for FAISS and vector databases.

Usage:
    # Local FAISS (default - Wikipedia)
    python scripts/run_rag_multi.py --input-path data/dev100.jsonl
    
    # Pinecone backend (Wikipedia)
    python scripts/run_rag_multi.py --input-path data/dev100.jsonl --backend pinecone
    
    # Custom documents with FAISS
    python scripts/run_rag_multi.py --input-path data/my_dev.jsonl --index-dir indexes/custom
    
    # Custom documents with Qdrant
    python scripts/run_rag_multi.py --input-path data/my_dev.jsonl --index-dir indexes/custom --backend qdrant
"""

import json
import time
import typer
import os
import sys
from typing import Optional, Literal
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.multi_backend import MultiBackendRetriever

app = typer.Typer(help="Multi-backend RAG baseline: retrieve passages and answer with LLM.")


def call_llm_with_context(client, model: str, question: str, passages: list, 
                          max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call LLM with retrieved context passages."""
    
    # Format context from passages
    context_parts = []
    for i, passage in enumerate(passages, 1):
        title = passage.get('title', 'Unknown')
        text = passage.get('chunk_text', '')
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
    out_path: str = "runs/rag_multi.jsonl",
    index_dir: str = "indexes",
    backend: Literal["faiss", "pinecone", "qdrant", "weaviate"] = "faiss",
    mode: Literal["bm25", "vector", "hybrid"] = "hybrid",
    k: int = 5,
    model: str = "gpt-4o-mini",
    max_items: int = 0,   # 0 = all
    # Pinecone settings
    pinecone_api_key: Optional[str] = None,
    pinecone_env: Optional[str] = None,
    pinecone_index: str = "typed-rag",
    # Qdrant settings
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    qdrant_collection: str = "typed-rag",
    # Weaviate settings
    weaviate_url: str = "http://localhost:8080",
    weaviate_class: str = "Document",
):
    """
    Run RAG baseline with configurable vector backend.
    """
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(input_path):
        input_path = str(project_root / input_path)
    if not os.path.isabs(out_path):
        out_path = str(project_root / out_path)
    if not os.path.isabs(index_dir):
        index_dir = str(project_root / index_dir)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    typer.echo(f"🚀 RAG Baseline with Multi-Backend Support")
    typer.echo(f"=" * 60)
    typer.echo(f"Input: {input_path}")
    typer.echo(f"Output: {out_path}")
    typer.echo(f"Index directory: {index_dir}")
    typer.echo(f"Backend: {backend}")
    typer.echo(f"Mode: {mode}")
    typer.echo(f"Top-k: {k}")
    typer.echo(f"Model: {model}")
    typer.echo(f"=" * 60)
    
    # Initialize retriever
    try:
        retriever = MultiBackendRetriever(
            bm25_path=f"{index_dir}/bm25_rank.pkl",
            meta_path=f"{index_dir}/meta.jsonl",
            vector_backend=backend,
            faiss_dir=f"{index_dir}/faiss_bge_small" if backend == "faiss" else "",
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_env,
            pinecone_index_name=pinecone_index,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            qdrant_collection_name=qdrant_collection,
            weaviate_url=weaviate_url,
            weaviate_class_name=weaviate_class,
        )
    except Exception as e:
        typer.echo(f"❌ Failed to initialize retriever: {e}", err=True)
        raise typer.Exit(code=1)
    
    # Initialize LLM client
    client = OpenAI()
    
    # Check retriever health
    health = retriever.health_check()
    typer.echo("\nRetrieval system health:")
    for key, value in health.items():
        status = "✅" if value else "❌"
        typer.echo(f"  {status} {key}: {value}")
    
    if not all(health.values()):
        typer.echo("⚠️  Warning: Some retrieval components are not loaded!")
    
    typer.echo()
    
    # Load questions
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
    
    typer.echo(f"Processing {total} questions (skipped {len(done)} already done)")
    typer.echo()
    
    with open(out_path, mode_file) as w:
        try:
            for r in tqdm(rows, total=total, desc=f"RAG-{backend}-{mode}"):
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
                            messages=[
                                {"role":"system","content":"You are a concise, factual assistant. If the answer isn't clear, say you don't know."},
                                {"role":"user","content":q}
                            ],
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
                        "retrieval_backend": backend,
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
                        "retrieval_backend": backend,
                        "retrieval_latency_ms": round(retrieval_latency, 2) if 'retrieval_latency' in locals() else 0
                    }, ensure_ascii=False) + "\n")
                    w.flush()
                    
        except KeyboardInterrupt:
            typer.echo("\nInterrupted — partial results saved to", out_path)
    
    typer.echo(f"\n✅ Wrote {out_path}")


if __name__ == "__main__":
    app()

