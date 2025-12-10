#!/usr/bin/env python3
"""
Baseline 3: Typed-RAG with Full Pipeline

This script runs the complete Typed-RAG pipeline:
1. Classify question type
2. Decompose into sub-questions
3. Retrieve evidence for each aspect
4. Generate aspect-level answers
5. Aggregate into final answer

Usage:
    python scripts/run_typed_rag.py
    python scripts/run_typed_rag.py --backend pinecone --source wikipedia
"""

from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.data.loaders import WikiNFQALoader, WikiNFQAQuestion
from typed_rag.rag_system import DataType, ask_typed_question
from typed_rag.core.keys import get_fastest_model


def main():
    parser = argparse.ArgumentParser(description="Run Typed-RAG baseline on Wiki-NFQA")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/wiki_nfqa/dev6.jsonl"),
        help="Input questions file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/typed_rag.jsonl"),
        help="Output answers file"
    )
    parser.add_argument(
        "--backend",
        choices=["faiss", "pinecone"],
        default="faiss",
        help="Vector store backend"
    )
    parser.add_argument(
        "--source",
        choices=["own_docs", "wikipedia"],
        default="own_docs",
        help="Document source"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM (use fallback)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (e.g., gemini-2.0-flash-lite or meta-llama/Llama-3.2-3B-Instruct)"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("="*60)
    print("Typed-RAG Baseline (Full Pipeline)")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"  Backend: {args.backend}")
    print(f"  Source: {args.source}")
    print(f"  Rerank: {'on' if args.rerank else 'off'}")
    print(f"  LLM: {'on' if not args.no_llm else 'fallback-only'}")
    print(f"\n📋 Pipeline Steps:")
    print("  1. Classify question type (Evidence/Comparison/Experience/Reason/Instruction/Debate)")
    print("  2. Decompose into aspect-level sub-questions")
    print("  3. Retrieve evidence for each aspect")
    print("  4. Generate aspect-level answers")
    print("  5. Aggregate into final answer\n")
    
    # Load questions
    print(f"📂 Loading questions from: {args.input}")
    
    if args.input.name.startswith("dev"):
        split = args.input.stem
        loader = WikiNFQALoader()
        questions = loader.load_questions(split)
    else:
        with open(args.input, "r") as f:
            questions = []
            for line in f:
                data = json.loads(line)
                questions.append(WikiNFQAQuestion.from_dict(data))
    
    print(f"✓ Loaded {len(questions)} questions")
    
    # Setup data type
    data_type = DataType(args.backend, args.source)
    model_name = args.model if args.model else get_fastest_model()
    is_hf = "/" in model_name
    print(f"📦 Using model: {model_name}")
    if is_hf:
        print(f"📦 Using HuggingFace Inference API")
    else:
        print(f"📦 Using Gemini API")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("output/batch_run")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Process questions
    results = []
    total_time = 0
    
    print(f"\n🚀 Running Typed-RAG pipeline...")
    print("-"*60)
    
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q.category}: {q.question[:60]}...")
        
        start = time.time()
        
        try:
            # Run full Typed-RAG pipeline
            result = ask_typed_question(
                q.question,
                data_type,
                model_name=model_name,
                rerank=args.rerank,
                use_llm=not args.no_llm,
                save_artifacts=True,
                output_dir=artifacts_dir / f"q{i:03d}_{q.question_id}"
            )
            
            elapsed = time.time() - start
            total_time += elapsed
            
            # Rate limit for Gemini API (15 RPM = 4 seconds between requests)
            # Note: Typed-RAG makes multiple LLM calls per question, so we need more conservative timing
            # Increased to 30s to avoid quota exhaustion (was 6.5s)
            if not is_hf and elapsed < 30.0:
                time.sleep(30.5 - elapsed)
            
            # Extract aspect answers if available
            aspects = []
            if result.aspects:
                for aspect in result.aspects:
                    aspects.append({
                        "aspect": aspect.get("aspect", ""),
                        "answer": aspect.get("answer", "")
                    })
            
            output = {
                "question_id": q.question_id,
                "question": q.question,
                "category": q.category,
                "answer": result.answer,
                "aspects": aspects,
                "classified_type": getattr(result, 'question_type', q.category),
                "system": "typed_rag",
                "model": model_name,
                "backend": args.backend,
                "source": args.source,
                "latency": elapsed
            }
            results.append(output)
            
            print(f"  ✓ Classified as: {output['classified_type']}")
            if aspects:
                print(f"  ✓ Decomposed into {len(aspects)} aspects")
            print(f"  ✓ Generated in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            elapsed = time.time() - start
            total_time += elapsed
            
            # Record failure
            output = {
                "question_id": q.question_id,
                "question": q.question,
                "category": q.category,
                "answer": f"ERROR: {str(e)}",
                "aspects": [],
                "classified_type": q.category,
                "system": "typed_rag",
                "model": model_name,
                "backend": args.backend,
                "source": args.source,
                "latency": elapsed,
                "error": str(e)
            }
            results.append(output)
    
    # Save results
    print(f"\n💾 Saving results to: {args.output}")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Questions processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    if successful:
        avg_latency = sum(r['latency'] for r in successful) / len(successful)
        print(f"Average latency: {avg_latency:.2f}s per question")
    print(f"Output saved to: {args.output}")
    print(f"Artifacts saved to: {artifacts_dir}")
    print("\n✅ Typed-RAG baseline complete!")
    
    # Show sample
    if successful:
        print("\n" + "="*60)
        print("Sample Result")
        print("="*60)
        sample = successful[0]
        print(f"Question: {sample['question']}")
        print(f"Classified Type: {sample['classified_type']}")
        if sample['aspects']:
            print(f"\nAspects ({len(sample['aspects'])}):")
            for i, asp in enumerate(sample['aspects'][:2], 1):
                print(f"  {i}. {asp['aspect']}: {asp['answer'][:100]}...")
        print(f"\nFinal Answer:")
        print(sample['answer'][:300] + "...")


if __name__ == "__main__":
    main()
