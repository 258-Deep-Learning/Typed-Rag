#!/usr/bin/env python3
"""
Test suite for question classifier evaluation.

Tests classification accuracy against ground truth categories from Wiki-NFQA dataset.
Compares pattern-only vs pattern+LLM classification performance.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.classifier.classifier import QuestionClassifier, classify_question
from typed_rag.data.loaders import WikiNFQALoader, WikiNFQAQuestion


def load_test_questions(input_file: Path) -> List[WikiNFQAQuestion]:
    """Load questions with ground truth categories."""
    questions = []
    
    if input_file.name.startswith("dev"):
        # Load from WikiNFQALoader
        split = input_file.stem
        loader = WikiNFQALoader()
        questions = loader.load_questions(split)
    else:
        # Load from JSONL file
        with open(input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                questions.append(WikiNFQAQuestion.from_dict(data))
    
    return questions


def evaluate_classifier(questions: List[WikiNFQAQuestion], use_llm: bool = True) -> Dict[str, Any]:
    """
    Evaluate classifier accuracy against ground truth.
    
    Args:
        questions: List of questions with ground truth categories
        use_llm: Whether to use LLM fallback (True) or pattern-only (False)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Classifier (use_llm={use_llm})")
    print(f"{'='*60}\n")
    
    predictions = []
    correct = 0
    total = 0
    
    # Category-wise tracking
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    misclassified = []
    llm_used_count = 0
    
    classifier = QuestionClassifier(use_llm=use_llm)
    
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Classifying: {q.question[:60]}...")
        
        # Get prediction
        predicted = classifier.classify(q.question)
        ground_truth = q.category
        
        # Normalize category names to uppercase for comparison
        predicted_normalized = predicted.upper().replace("-", "_")
        ground_truth_normalized = ground_truth.upper().replace("-", "_")
        
        # Special handling for EVIDENCE-BASED variations
        if "EVIDENCE" in predicted_normalized:
            predicted_normalized = "EVIDENCE_BASED"
        if "EVIDENCE" in ground_truth_normalized:
            ground_truth_normalized = "EVIDENCE_BASED"
        
        is_correct = predicted_normalized == ground_truth_normalized
        
        # Track statistics
        predictions.append({
            "question_id": q.question_id,
            "question": q.question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct
        })
        
        if is_correct:
            correct += 1
        else:
            misclassified.append({
                "question": q.question,
                "ground_truth": ground_truth,
                "predicted": predicted
            })
        
        total += 1
        
        # Update category stats
        category_stats[ground_truth]["total"] += 1
        category_stats[ground_truth]["predictions"].append(predicted)
        if is_correct:
            category_stats[ground_truth]["correct"] += 1
        
        # Update confusion matrix
        confusion_matrix[ground_truth][predicted] += 1
        
        print(f"  Ground truth: {ground_truth}")
        print(f"  Predicted: {predicted}")
        print(f"  {'✓ Correct' if is_correct else '✗ Incorrect'}\n")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-category metrics
    per_category_metrics = {}
    for category, stats in category_stats.items():
        cat_total = stats["total"]
        cat_correct = stats["correct"]
        cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0.0
        
        per_category_metrics[category] = {
            "accuracy": cat_accuracy,
            "correct": cat_correct,
            "total": cat_total,
            "predictions": stats["predictions"]
        }
    
    # Build results
    results = {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "incorrect": total - correct,
        "use_llm": use_llm,
        "per_category": per_category_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "misclassified_examples": misclassified,
        "all_predictions": predictions
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Classification Results Summary")
    print(f"{'='*60}\n")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{total} correct)")
    print(f"Mode: {'Pattern + LLM Fallback' if use_llm else 'Pattern-Only'}\n")
    
    print("Per-Category Performance:")
    for category, metrics in sorted(per_category_metrics.items()):
        print(f"  {category:20s}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    
    if misclassified:
        print(f"\nMisclassified: {len(misclassified)} examples")
        print("\nCommon Error Patterns:")
        error_patterns = defaultdict(int)
        for example in misclassified:
            pattern = f"{example['ground_truth']} → {example['predicted']}"
            error_patterns[pattern] += 1
        
        for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:5]:
            print(f"  {pattern}: {count} cases")
    
    return results


def save_results(results: Dict[str, Any], output_file: Path):
    """Save evaluation results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


def save_misclassified_examples(results: Dict[str, Any], output_file: Path):
    """Save misclassified examples to text file for analysis."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("MISCLASSIFIED EXAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for i, example in enumerate(results["misclassified_examples"], 1):
            f.write(f"Example {i}:\n")
            f.write(f"Question: {example['question']}\n")
            f.write(f"Ground Truth: {example['ground_truth']}\n")
            f.write(f"Predicted: {example['predicted']}\n")
            f.write("-"*80 + "\n\n")
    
    print(f"✓ Misclassified examples saved to: {output_file}")


def main():
    """Run classifier evaluation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate question classifier")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/wiki_nfqa/dev100.jsonl"),
        help="Input questions file with ground truth categories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/classification"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare pattern-only vs pattern+LLM performance"
    )
    
    args = parser.parse_args()
    
    # Load questions
    print(f"📂 Loading questions from: {args.input}")
    questions = load_test_questions(args.input)
    print(f"✓ Loaded {len(questions)} questions\n")
    
    # Evaluate with pattern-only
    print("\n" + "="*60)
    print("TEST 1: Pattern-Only Classification")
    print("="*60)
    results_pattern_only = evaluate_classifier(questions, use_llm=False)
    
    # Save results
    output_file = args.output_dir / "classification_pattern_only.json"
    save_results(results_pattern_only, output_file)
    
    misclassified_file = args.output_dir / "misclassified_pattern_only.txt"
    save_misclassified_examples(results_pattern_only, misclassified_file)
    
    if args.compare:
        # Evaluate with pattern + LLM
        print("\n" + "="*60)
        print("TEST 2: Pattern + LLM Fallback Classification")
        print("="*60)
        results_with_llm = evaluate_classifier(questions, use_llm=True)
        
        # Save results
        output_file = args.output_dir / "classification_with_llm.json"
        save_results(results_with_llm, output_file)
        
        misclassified_file = args.output_dir / "misclassified_with_llm.txt"
        save_misclassified_examples(results_with_llm, misclassified_file)
        
        # Comparison summary
        print("\n" + "="*60)
        print("COMPARISON: Pattern-Only vs Pattern+LLM")
        print("="*60)
        print(f"Pattern-Only Accuracy:  {results_pattern_only['overall_accuracy']:.1%}")
        print(f"Pattern+LLM Accuracy:   {results_with_llm['overall_accuracy']:.1%}")
        
        improvement = results_with_llm['overall_accuracy'] - results_pattern_only['overall_accuracy']
        print(f"Improvement:            {improvement:+.1%}")
        
        print(f"\nLLM adds: {improvement * 100:.1f} percentage points")
        print(f"Cost: ~{len(questions) * 0.55:.0f} LLM calls (~55% of questions)")
    
    print("\n✅ Classification evaluation complete!")


if __name__ == "__main__":
    main()
