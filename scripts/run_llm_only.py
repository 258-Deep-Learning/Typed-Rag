#!/usr/bin/env python3
"""
Baseline 1: LLM-Only (No Retrieval)

Generates answers using only the LLM without any retrieved context.
This establishes the lower bound for performance.

Usage:
    python scripts/run_llm_only.py
    python scripts/run_llm_only.py --input data/wiki_nfqa/dev6.jsonl
    python scripts/run_llm_only.py --model gemini-2.0-flash-lite
"""

from __future__ import annotations
import argparse
import json
import time
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.data.loaders import load_wiki_nfqa
from typed_rag.core.keys import get_fastest_model
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient


def is_huggingface_model(model_name: str) -> bool:
    """Check if model is from HuggingFace."""
    return "/" in model_name  # HF models have format "org/model-name"


def generate_answer_llm_only(question: str, llm, is_hf: bool = False) -> str:
    """Generate answer using only LLM, no retrieval."""
    prompt = f"""Answer the following question concisely and accurately.
Do not make up information. If you're unsure, say so.

Question: {question}

Answer:"""
    
    if is_hf:
        # HuggingFace API
        response = llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    else:
        # Gemini API
        response = llm.invoke(prompt)
        return response.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Run LLM-only baseline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/wiki_nfqa/dev6.jsonl"),
        help="Input questions file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/llm_only.jsonl"),
        help="Output answers file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: from get_fastest_model())"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("="*60)
    print("LLM-Only Baseline")
    print("="*60)
    
    # Load model
    model_name = args.model or get_fastest_model()
    print(f"\n📦 Loading model: {model_name}")
    
    is_hf = is_huggingface_model(model_name)
    
    if is_hf:
        # HuggingFace model
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN not set in .env file")
        
        llm = InferenceClient(
            model=model_name,
            token=hf_token
        )
        print(f"✓ Using HuggingFace Inference API")
    else:
        # Gemini model
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env file")
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.0  # Deterministic
        )
        print(f"✓ Using Google Gemini API")
    
    # Load questions
    print(f"📂 Loading questions from: {args.input}")
    
    if args.input.name.startswith("dev"):
        split = args.input.stem  # "dev6" or "dev100"
        from typed_rag.data.loaders import WikiNFQALoader
        loader = WikiNFQALoader()
        questions = loader.load_questions(split)
    else:
        with open(args.input, "r") as f:
            questions = []
            for line in f:
                data = json.loads(line)
                from typed_rag.data.loaders import WikiNFQAQuestion
                questions.append(WikiNFQAQuestion.from_dict(data))
    
    print(f"✓ Loaded {len(questions)} questions")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Process questions
    results = []
    total_time = 0
    
    print(f"\n🚀 Generating answers...")
    print("-"*60)
    
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q.category}: {q.question[:60]}...")
        
        start = time.time()
        answer = generate_answer_llm_only(q.question, llm, is_hf=is_hf)
        elapsed = time.time() - start
        total_time += elapsed
        
        result = {
            "question_id": q.question_id,
            "question": q.question,
            "category": q.category,
            "answer": answer,
            "system": "llm_only",
            "model": model_name,
            "latency": elapsed
        }
        results.append(result)
        
        print(f"  ✓ Generated in {elapsed:.2f}s")
    
    # Save results
    print(f"\n💾 Saving results to: {args.output}")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Questions processed: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {total_time/len(results):.2f}s per question")
    print(f"Output saved to: {args.output}")
    print("\n✅ LLM-only baseline complete!")


if __name__ == "__main__":
    main()