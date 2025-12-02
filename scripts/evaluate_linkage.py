#!/usr/bin/env python3
"""
LINKAGE Evaluation Script

Evaluates system outputs against reference answers using LINKAGE methodology.
Computes MRR (Mean Reciprocal Rank) and MPR (Mean Percentile Rank).

Usage:
    python scripts/evaluate_linkage.py --systems runs/llm_only.jsonl runs/rag_baseline.jsonl runs/typed_rag.jsonl
    python scripts/evaluate_linkage.py --systems runs/*.jsonl --references data/wiki_nfqa/references.jsonl
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.eval.linkage import LinkageScorer, LinkageInstance, evaluate_linkage


def load_references(ref_file: Path) -> Dict[str, List[str]]:
    """Load reference answers by question_id."""
    references = {}
    with open(ref_file, "r") as f:
        for line in f:
            data = json.loads(line)
            references[data["question_id"]] = data["reference_answers"]
    return references


def load_system_output(output_file: Path) -> List[Dict[str, Any]]:
    """Load system output JSONL."""
    outputs = []
    with open(output_file, "r") as f:
        for line in f:
            outputs.append(json.loads(line))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate systems using LINKAGE")
    parser.add_argument(
        "--systems",
        nargs="+",
        type=Path,
        required=True,
        help="System output files (JSONL)"
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/wiki_nfqa/references.jsonl"),
        help="Reference answers file"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM for LINKAGE scoring (default: True)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use embedding fallback only"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/linkage_evaluation.json"),
        help="Output results file"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("="*70)
    print("LINKAGE EVALUATION")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Reference file: {args.references}")
    print(f"  Systems to evaluate: {len(args.systems)}")
    print(f"  Scoring method: {'LLM-based' if not args.no_llm else 'Embedding-based'}")
    
    # Load references
    print(f"\n📂 Loading reference answers...")
    references = load_references(args.references)
    print(f"✓ Loaded {len(references)} reference sets")
    
    # Initialize scorer
    use_llm = args.use_llm and not args.no_llm
    scorer = LinkageScorer(use_llm=use_llm)
    print(f"✓ Initialized LINKAGE scorer (mode: {'LLM' if use_llm else 'Embedding'})")
    
    # Evaluate each system
    all_results = {}
    
    for system_file in args.systems:
        system_name = system_file.stem  # e.g., "llm_only", "rag_baseline", "typed_rag"
        
        print(f"\n" + "="*70)
        print(f"Evaluating: {system_name}")
        print("="*70)
        
        # Load outputs
        outputs = load_system_output(system_file)
        print(f"✓ Loaded {len(outputs)} outputs")
        
        # Create LINKAGE instances
        instances = []
        skipped = 0
        
        for output in outputs:
            qid = output["question_id"]
            question = output["question"]
            answer = output["answer"]
            
            if qid not in references:
                print(f"⚠ Warning: No references for {qid}, skipping")
                skipped += 1
                continue
            
            refs = references[qid]
            instances.append(LinkageInstance(
                question=question,
                references=refs,
                candidate=answer
            ))
        
        print(f"✓ Created {len(instances)} evaluation instances")
        if skipped > 0:
            print(f"⚠ Skipped {skipped} questions without references")
        
        # Run evaluation
        print(f"\n🔍 Running LINKAGE evaluation...")
        mrr, mpr, results = evaluate_linkage(instances, scorer)
        
        # Compute per-category metrics
        category_metrics = defaultdict(lambda: {"ranks": [], "ref_sizes": []})
        
        for output, result in zip(outputs, results):
            if output["question_id"] in references:
                category = output.get("category", "Unknown")
                category_metrics[category]["ranks"].append(result.rank)
                category_metrics[category]["ref_sizes"].append(
                    len(references[output["question_id"]]) + 1
                )
        
        # Compute category-level MRR/MPR
        category_results = {}
        for category, data in category_metrics.items():
            cat_mrr = sum(1.0 / r for r in data["ranks"]) / len(data["ranks"]) if data["ranks"] else 0.0
            
            percentiles = []
            for rank, ref_size in zip(data["ranks"], data["ref_sizes"]):
                percentiles.append(1.0 - (rank - 1) / ref_size)
            cat_mpr = sum(percentiles) / len(percentiles) * 100.0 if percentiles else 0.0
            
            category_results[category] = {
                "mrr": cat_mrr,
                "mpr": cat_mpr,
                "count": len(data["ranks"])
            }
        
        # Store results
        all_results[system_name] = {
            "overall": {
                "mrr": mrr,
                "mpr": mpr,
                "questions": len(instances)
            },
            "by_category": category_results,
            "llm_scoring": use_llm
        }
        
        # Print results
        print(f"\n📊 Results for {system_name}:")
        print(f"  Overall MRR: {mrr:.4f}")
        print(f"  Overall MPR: {mpr:.2f}%")
        print(f"\n  Per-Category:")
        for category, metrics in sorted(category_results.items()):
            print(f"    {category:20s} MRR: {metrics['mrr']:.4f}  MPR: {metrics['mpr']:.2f}%  (n={metrics['count']})")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    
    # Comparison summary
    print(f"\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'System':<20} {'MRR':>10} {'MPR':>10} {'Questions':>10}")
    print("-"*70)
    
    for system_name in sorted(all_results.keys()):
        result = all_results[system_name]["overall"]
        print(f"{system_name:<20} {result['mrr']:>10.4f} {result['mpr']:>9.2f}% {result['questions']:>10}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
