#!/usr/bin/env python3
"""
Setup script for Wiki-NFQA dataset.

This script will:
1. Download Wiki-NFQA from Hugging Face (or use local files)
2. Create test.jsonl with all questions
3. Create dev100.jsonl with stratified 100-question sample
4. Create references.jsonl with reference answers
5. Validate the dataset

Usage:
    python scripts/setup_wiki_nfqa.py
    python scripts/setup_wiki_nfqa.py --source huggingface
    python scripts/setup_wiki_nfqa.py --source local --input-dir ./raw_data
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.data.loaders import WikiNFQALoader, WikiNFQAQuestion


def download_from_huggingface(dataset_name: str = "wiki-nfqa") -> List[Dict[str, Any]]:
    """
    Download Wiki-NFQA from Hugging Face.
    
    Note: You'll need to find the actual dataset name on Hugging Face.
    Common patterns:
    - "your-org/wiki-nfqa"
    - "nfqa/wiki-nfqa"
    - Check the paper's GitHub for links
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)
    
    print(f"📥 Downloading {dataset_name} from Hugging Face...")
    
    try:
        # Try loading the dataset
        # TODO: Replace with actual dataset name once you find it
        dataset = load_dataset(dataset_name)
        
        # Convert to list of dicts
        test_split = dataset.get("test", dataset.get("validation", None))
        if test_split is None:
            print("❌ Error: No test/validation split found")
            print(f"Available splits: {list(dataset.keys())}")
            sys.exit(1)
        
        questions = []
        for item in test_split:
            questions.append({
                "question_id": item.get("id", item.get("question_id", "")),
                "question": item.get("question", item.get("question_text", "")),
                "category": item.get("category", item.get("type", "Unknown")),
                "reference_answers": item.get("reference_answers", item.get("reference_answer_list", []))
            })
        
        print(f"✓ Downloaded {len(questions)} questions")
        return questions
    
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print("\nTip: Check the paper's GitHub repo for dataset links")
        print("Or use --source local to load from local files")
        sys.exit(1)


def load_from_local(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Load Wiki-NFQA from local JSONL files.
    
    Expected format:
    - input_dir/wiki_nfqa.jsonl  OR
    - input_dir/test.jsonl
    
    Each line: {"question_id": "...", "question": "...", "category": "...", "reference_answers": [...]}
    """
    print(f"📂 Loading from local directory: {input_dir}")
    
    # Try different file names
    possible_files = [
        input_dir / "wiki_nfqa.jsonl",
        input_dir / "test.jsonl",
        input_dir / "questions.jsonl",
    ]
    
    input_file = None
    for file_path in possible_files:
        if file_path.exists():
            input_file = file_path
            break
    
    if input_file is None:
        print(f"❌ Error: No dataset file found in {input_dir}")
        print(f"Expected one of: {[f.name for f in possible_files]}")
        sys.exit(1)
    
    questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                questions.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping malformed line {line_num}: {e}")
                continue
    
    print(f"✓ Loaded {len(questions)} questions from {input_file.name}")
    return questions


def validate_questions(questions: List[WikiNFQAQuestion]) -> bool:
    """Validate that questions have required fields."""
    print("\n🔍 Validating dataset...")
    
    issues = []
    
    for i, q in enumerate(questions):
        if not q.question_id:
            issues.append(f"Question {i}: Missing question_id")
        if not q.question:
            issues.append(f"Question {i}: Missing question text")
        if not q.category:
            issues.append(f"Question {i}: Missing category")
        if not q.reference_answers:
            issues.append(f"Question {i}: No reference answers")
    
    if issues:
        print(f"❌ Found {len(issues)} validation issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    
    print("✓ All questions validated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup Wiki-NFQA dataset")
    parser.add_argument(
        "--source",
        choices=["huggingface", "local"],
        default="local",
        help="Source to load dataset from"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing local dataset files (for --source local)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wiki-nfqa",
        help="Hugging Face dataset name (for --source huggingface)"
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=100,
        help="Size of dev sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Wiki-NFQA Dataset Setup")
    print("="*60)
    
    # Step 1: Load raw data
    if args.source == "huggingface":
        raw_questions = download_from_huggingface(args.dataset_name)
    else:
        raw_questions = load_from_local(args.input_dir)
    
    # Step 2: Convert to WikiNFQAQuestion objects
    questions = [WikiNFQAQuestion.from_dict(q) for q in raw_questions]
    
    # Step 3: Validate
    if not validate_questions(questions):
        print("\n❌ Dataset validation failed. Please fix issues and retry.")
        sys.exit(1)
    
    # Step 4: Initialize loader
    loader = WikiNFQALoader()
    
    # Step 5: Save full test set
    print(f"\n💾 Saving test set...")
    loader.save_questions(questions, split="test")
    
    # Step 6: Create and save dev sample
    print(f"\n🎲 Creating stratified dev{args.dev_size} sample...")
    dev_questions = loader.create_stratified_sample(
        questions,
        n=args.dev_size,
        seed=args.seed
    )
    loader.save_questions(dev_questions, split=f"dev{args.dev_size}")
    
    # Step 7: Save references (for LINKAGE evaluation)
    print(f"\n💾 Saving reference answers...")
    ref_file = loader.data_dir / "references.jsonl"
    with open(ref_file, "w", encoding="utf-8") as f:
        for q in questions:
            ref_data = {
                "question_id": q.question_id,
                "question": q.question,
                "reference_answers": q.reference_answers
            }
            f.write(json.dumps(ref_data) + "\n")
    print(f"✓ Saved references to {ref_file}")
    
    # Step 8: Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    stats = loader.get_statistics(questions)
    print(f"\nTotal questions: {stats['total']}")
    print(f"\nBy category:")
    for category, count in sorted(stats['by_category'].items()):
        percentage = (count / stats['total']) * 100
        print(f"  {category:20s}: {count:4d} ({percentage:5.1f}%)")
    print(f"\nAverage reference answers: {stats['avg_references']:.1f}")
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print(f"\nDataset saved to: {loader.data_dir}")
    print(f"  - test.jsonl          : {stats['total']} questions")
    print(f"  - dev{args.dev_size}.jsonl       : {len(dev_questions)} questions")
    print(f"  - references.jsonl    : Reference answers for evaluation")
    
    print("\n📝 Next steps:")
    print("  1. Verify the dataset looks correct:")
    print(f"     head -n 1 {loader.data_dir}/test.jsonl | python -m json.tool")
    print("  2. Continue to Step 2: Create baseline scripts")


if __name__ == "__main__":
    main()