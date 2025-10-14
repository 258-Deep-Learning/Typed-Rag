#!/usr/bin/env python3
"""
Vanilla RAG baseline: Retrieve passages and use LLM to answer.
Uses hybrid retrieval (Pinecone + BM25).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import Retriever, Doc


def load_questions(input_path: Path) -> List[Dict[str, Any]]:
    """Load questions from JSONL."""
    questions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def format_passages(docs: List[Doc]) -> str:
    """Format retrieved passages for prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.title or "Unknown"
        text = doc.text or "[No text available]"
        formatted.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(formatted)


def call_llm_with_context(question: str, passages: str, model: str = "gemini-2.5-flash", seed: int = 42) -> str:
    """
    Call LLM API with retrieved context.
    Supports both Google Gemini and OpenAI.
    Falls back to a dummy response if API is not available.
    """
    try:
        prompt = f"""Answer the question based on the following passages:

{passages}

Question: {question}

Answer:"""
        
        # Try Gemini first (if GOOGLE_API_KEY is set)
        if os.getenv("GOOGLE_API_KEY") and model.startswith("gemini"):
            import google.generativeai as genai
            
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_model = genai.GenerativeModel(model)
            
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    candidate_count=1,
                )
            )
            
            return response.text.strip()
        
        # Fall back to OpenAI
        elif os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                seed=seed,
            )
            
            return response.choices[0].message.content.strip()
        
        else:
            raise ValueError("No API key found for Gemini or OpenAI")
    
    except Exception as e:
        logger.warning("LLM call failed, using dummy response", error=str(e))
        return f"[RAG baseline - API not available] Retrieved {passages.count('[') if passages else 0} passages but cannot generate answer."


def run_baseline(
    questions: List[Dict[str, Any]],
    retriever: Retriever,
    k: int = 5,
    model: str = "gemini-2.5-flash",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run RAG baseline on all questions."""
    results = []
    
    for i, q_data in enumerate(questions):
        question_id = q_data["question_id"]
        question = q_data["question"]
        
        logger.info("Processing question", idx=i+1, total=len(questions), qid=question_id)
        
        t0 = time.time()
        
        # Retrieve passages
        docs = retriever.retrieve(query=question, k=k, mode="hybrid", fuse="zscore")
        
        # Format passages
        passages_text = format_passages(docs)
        
        # Generate answer
        answer = call_llm_with_context(question, passages_text, model=model, seed=seed)
        
        latency_ms = int((time.time() - t0) * 1000)
        
        # Prepare passage info for logging
        passage_info = []
        for doc in docs:
            passage_info.append({
                "id": doc.id,
                "title": doc.title,
                "url": doc.url,
                "score": doc.score,
                "text": doc.text[:200] + "..." if doc.text and len(doc.text) > 200 else doc.text,
            })
        
        result = {
            "question_id": question_id,
            "question": question,
            "prompt": f"Passages:\n{passages_text}\n\nQuestion: {question}",
            "passages": passage_info,
            "answer": answer,
            "latency_ms": latency_ms,
            "seed": seed,
            "model": model,
            "retrieval_k": k,
        }
        
        results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save results to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info("Saved results", output=str(output_path), count=len(results))


def compute_stats(results: List[Dict[str, Any]]) -> None:
    """Compute and log statistics."""
    latencies = [r["latency_ms"] for r in results]
    if latencies:
        median_latency = sorted(latencies)[len(latencies) // 2]
        mean_latency = sum(latencies) / len(latencies)
        logger.info("Latency stats", median_ms=median_latency, mean_ms=int(mean_latency))


def main():
    parser = argparse.ArgumentParser(description="Vanilla RAG baseline")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input dev set JSONL")
    parser.add_argument("--out", type=str, required=True, help="Output results JSONL")
    parser.add_argument("--pinecone_index", type=str, default="typedrag-own", help="Pinecone index name")
    parser.add_argument("--pinecone_namespace", type=str, default="own_docs", help="Pinecone namespace")
    parser.add_argument("--bm25_index", type=str, default="typed_rag/indexes/lucene_own", help="BM25 index directory")
    parser.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", 
                        help="LLM model name (gemini-2.5-flash, gemini-1.5-flash, gpt-3.5-turbo, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device for embeddings (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist", input=str(input_path))
        return
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No LLM API key found (GOOGLE_API_KEY or OPENAI_API_KEY) - will use dummy responses")
    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set")
        return
    
    # Check BM25 index
    bm25_path = Path(args.bm25_index)
    if not bm25_path.exists():
        logger.error("BM25 index does not exist", path=str(bm25_path))
        logger.info("Please run build_bm25.py first")
        return
    
    logger.info("Loading questions", input=str(input_path))
    questions = load_questions(input_path)
    logger.info("Loaded questions", count=len(questions))
    
    if not questions:
        logger.warning("No questions to process")
        return
    
    # Initialize retriever
    logger.info("Initializing retriever", 
                pinecone_index=args.pinecone_index,
                pinecone_namespace=args.pinecone_namespace,
                bm25_index=args.bm25_index)
    
    retriever = Retriever(
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        bm25_index_dir=args.bm25_index,
        device=args.device,
    )
    
    logger.info("Running RAG baseline", model=args.model, k=args.k, seed=args.seed)
    results = run_baseline(questions, retriever, k=args.k, model=args.model, seed=args.seed)
    
    save_results(results, Path(args.out))
    compute_stats(results)
    logger.info("Baseline complete", total_results=len(results))


if __name__ == "__main__":
    main()

