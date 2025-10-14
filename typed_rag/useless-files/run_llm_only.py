#!/usr/bin/env python3
"""
LLM-only baseline: Answer questions without retrieval.
Uses OpenAI API (or compatible endpoint).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()


def load_questions(input_path: Path) -> List[Dict[str, Any]]:
    """Load questions from JSONL."""
    questions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def call_llm(question: str, model: str = "gemini-2.5-flash", seed: int = 42) -> str:
    """
    Call LLM API to answer a question.
    Supports both Google Gemini and OpenAI.
    Falls back to a dummy response if API is not available.
    """
    try:
        # Try Gemini first (if GOOGLE_API_KEY is set)
        if os.getenv("GOOGLE_API_KEY") and model.startswith("gemini"):
            import google.generativeai as genai
            
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_model = genai.GenerativeModel(model)
            
            prompt = f"Please answer the following question concisely:\n\n{question}"
            
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
            
            prompt = f"Please answer the following question concisely:\n\n{question}"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
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
        return f"[LLM-only baseline - API not available] Unable to answer: {question}"


def run_baseline(
    questions: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run LLM-only baseline on all questions."""
    results = []
    
    for i, q_data in enumerate(questions):
        question_id = q_data["question_id"]
        question = q_data["question"]
        
        logger.info("Processing question", idx=i+1, total=len(questions), qid=question_id)
        
        t0 = time.time()
        answer = call_llm(question, model=model, seed=seed)
        latency_ms = int((time.time() - t0) * 1000)
        
        result = {
            "question_id": question_id,
            "question": question,
            "prompt": f"Question: {question}",
            "passages": [],  # No retrieval
            "answer": answer,
            "latency_ms": latency_ms,
            "seed": seed,
            "model": model,
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
    parser = argparse.ArgumentParser(description="LLM-only baseline")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input dev set JSONL")
    parser.add_argument("--out", type=str, required=True, help="Output results JSONL")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", 
                        help="LLM model name (gemini-2.5-flash, gemini-1.5-flash, gpt-3.5-turbo, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist", input=str(input_path))
        return
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No API key found (GOOGLE_API_KEY or OPENAI_API_KEY) - will use dummy responses")
    
    logger.info("Loading questions", input=str(input_path))
    questions = load_questions(input_path)
    logger.info("Loaded questions", count=len(questions))
    
    if not questions:
        logger.warning("No questions to process")
        return
    
    logger.info("Running LLM-only baseline", model=args.model, seed=args.seed)
    results = run_baseline(questions, model=args.model, seed=args.seed)
    
    save_results(results, Path(args.out))
    compute_stats(results)
    logger.info("Baseline complete", total_results=len(results))


if __name__ == "__main__":
    main()

