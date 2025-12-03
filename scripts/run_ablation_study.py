#!/usr/bin/env python3
"""
Ablation Study Script for Typed-RAG

Tests the impact of disabling individual components:
1. Full Typed-RAG (baseline)
2. Without Classification (force Evidence-based)
3. Without Decomposition (single aspect query)
4. Without Retrieval (pure LLM generation)

Usage:
    python scripts/run_ablation_study.py --input data/wiki_nfqa/dev6.jsonl --output results/ablation/
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.rag_system import ask_typed_question, DataType
from typed_rag.data.loaders import WikiNFQALoader, WikiNFQAQuestion


def run_ablation_variant(
    questions: list,
    variant: str,
    data_type: DataType,
    model_name: str = None,
) -> list:
    """
    Run a specific ablation variant.
    
    Args:
        questions: List of questions with metadata
        variant: One of ["full", "no_classification", "no_decomposition", "no_retrieval"]
        data_type: DataType configuration
        model_name: Model to use (defaults to gemini-2.0-flash-lite)
    
    Returns:
        List of results with answers and metadata
    """
    # Configure ablation flags
    config = {
        "full": {
            "use_classification": True,
            "use_decomposition": True,
            "use_retrieval": True,
        },
        "no_classification": {
            "use_classification": False,
            "use_decomposition": True,
            "use_retrieval": True,
        },
        "no_decomposition": {
            "use_classification": True,
            "use_decomposition": False,
            "use_retrieval": True,
        },
        "no_retrieval": {
            "use_classification": True,
            "use_decomposition": True,
            "use_retrieval": False,
        },
    }
    
    if variant not in config:
        raise ValueError(f"Unknown variant: {variant}. Must be one of {list(config.keys())}")
    
    flags = config[variant]
    results = []
    
    print(f"\n{'='*80}")
    print(f"Running Ablation Variant: {variant.upper()}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Classification: {'✓' if flags['use_classification'] else '✗'}")
    print(f"  - Decomposition: {'✓' if flags['use_decomposition'] else '✗'}")
    print(f"  - Retrieval: {'✓' if flags['use_retrieval'] else '✗'}")
    print()
    
    for i, q in enumerate(questions, 1):
        question_id = q.question_id
        question = q.question
        
        print(f"[{i}/{len(questions)}] Processing: {question[:60]}...")
        
        start_time = time.time()
        
        try:
            result = ask_typed_question(
                query=question,
                data_type=data_type,
                model_name=model_name,
                use_llm=True,
                save_artifacts=False,
                use_classification=flags["use_classification"],
                use_decomposition=flags["use_decomposition"],
                use_retrieval=flags["use_retrieval"],
            )
            
            elapsed = time.time() - start_time
            
            # Format result
            output = {
                "question_id": question_id,
                "question": question,
                "answer": result.answer,  # Changed from final_answer
                "question_type": result.question_type,
                "aspects": [aspect["aspect"] for aspect in result.aspects] if result.aspects else [],
                "num_aspects": len(result.aspects) if result.aspects else 0,
                "latency_seconds": round(elapsed, 2),
                "variant": variant,
                "config": flags,
            }
            
            # Add reference answers if available
            if hasattr(q, 'reference_answers') and q.reference_answers:
                output["reference_answers"] = q.reference_answers
            
            results.append(output)
            print(f"  ✓ Completed in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "question_id": question_id,
                "question": question,
                "answer": None,
                "error": str(e),
                "variant": variant,
            })
        
        # Rate limiting
        if i < len(questions):
            time.sleep(4.0)  # Respect API limits
    
    print(f"\n✓ Completed {variant} variant: {len([r for r in results if 'error' not in r])}/{len(questions)} successful")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Typed-RAG ablation study")
    parser.add_argument("--input", required=True, help="Input JSONL file with questions")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--output-dir", help="Output directory for results (alias for --output)")
    parser.add_argument("--source", default="wikipedia", choices=["wikipedia", "own_docs"], help="Data source")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "pinecone"], help="Vector store backend")
    parser.add_argument("--model", default=None, help="Model name (e.g., meta-llama/Llama-3.2-3B-Instruct or gemini-2.0-flash-lite)")
    parser.add_argument("--variants", nargs="+", 
                        default=["full", "no_classification", "no_decomposition", "no_retrieval"],
                        help="Variants to run")
    
    args = parser.parse_args()
    
    # Handle output directory (accept either --output or --output-dir)
    output_dir_str = args.output_dir or args.output
    if not output_dir_str:
        raise ValueError("Must specify either --output or --output-dir")
    
    # Load questions
    print(f"📂 Loading questions from: {args.input}")
    
    if Path(args.input).name.startswith("dev"):
        split = Path(args.input).stem  # "dev6" or "dev100"
        loader = WikiNFQALoader()
        questions = loader.load_questions(split)
    else:
        with open(args.input, "r") as f:
            questions = []
            for line in f:
                data = json.loads(line)
                questions.append(WikiNFQAQuestion.from_dict(data))
    
    print(f"✓ Loaded {len(questions)} questions\n")
    
    # Create output directory
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure data type
    data_type = DataType(type=args.backend, source=args.source)
    
    # Run each variant
    all_results = {}
    
    for variant in args.variants:
        results = run_ablation_variant(
            questions=questions,
            variant=variant,
            data_type=data_type,
            model_name=args.model,
        )
        
        # Save results
        output_file = output_dir / f"{variant}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"✓ Saved results to: {output_file}")
        all_results[variant] = results
    
    # Create summary
    summary = {
        "total_questions": len(questions),
        "variants": {},
    }
    
    for variant, results in all_results.items():
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        avg_latency = sum(r.get("latency_seconds", 0) for r in successful) / len(successful) if successful else 0
        
        summary["variants"][variant] = {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "avg_latency_seconds": round(avg_latency, 2),
        }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"\nSummary:")
    for variant, stats in summary["variants"].items():
        print(f"\n{variant}:")
        print(f"  Successful: {stats['successful']}/{stats['total']}")
        print(f"  Avg Latency: {stats['avg_latency_seconds']}s")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate with LINKAGE:")
    print(f"     python scripts/evaluate_linkage.py --systems {output_dir}/*.jsonl --output {output_dir}/linkage.json --no-llm")
    print(f"  2. Compare results to identify component contributions")


if __name__ == "__main__":
    main()
