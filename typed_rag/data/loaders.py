"""Data loading utilities for Wiki-NFQA and other datasets."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import random


@dataclass
class WikiNFQAQuestion:
    """Single question from Wiki-NFQA dataset."""
    question_id: str
    question: str
    category: str  # Question type (Evidence-based, Comparison, etc.)
    reference_answers: List[str]
    
    @classmethod
    def from_dict(cls, data: dict) -> WikiNFQAQuestion:
        """Load from dictionary."""
        return cls(
            question_id=data.get("question_id", data.get("qid", "")),
            question=data.get("question", data.get("question_text", "")),
            category=data.get("category", data.get("category_prediction", "Unknown")),
            reference_answers=data.get("reference_answers", data.get("reference_answer_list", []))
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "reference_answers": self.reference_answers
        }


@dataclass
class WikiPassage:
    """Wikipedia passage for retrieval."""
    passage_id: str
    title: str
    text: str
    url: str = ""
    
    @classmethod
    def from_dict(cls, data: dict) -> WikiPassage:
        """Load from dictionary."""
        return cls(
            passage_id=data.get("passage_id", data.get("id", "")),
            title=data.get("title", ""),
            text=data.get("text", data.get("content", "")),
            url=data.get("url", "")
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passage_id": self.passage_id,
            "title": self.title,
            "text": self.text,
            "url": self.url
        }


class WikiNFQALoader:
    """Loader for Wiki-NFQA dataset."""
    
    def __init__(self, data_dir: Path | str = "data/wiki_nfqa"):
        """Initialize loader with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_questions(self, split: str = "test") -> List[WikiNFQAQuestion]:
        """
        Load questions from a split.
        
        Args:
            split: One of "test", "dev100", or "train" (if available)
        
        Returns:
            List of WikiNFQAQuestion objects
        """
        file_path = self.data_dir / f"{split}.jsonl"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Run: python scripts/setup_wiki_nfqa.py first"
            )
        
        questions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                questions.append(WikiNFQAQuestion.from_dict(data))
        
        return questions
    
    def save_questions(self, questions: List[WikiNFQAQuestion], split: str = "test") -> None:
        """Save questions to a split file."""
        file_path = self.data_dir / f"{split}.jsonl"
        
        with open(file_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q.to_dict()) + "\n")
        
        print(f"✓ Saved {len(questions)} questions to {file_path}")
    
    def create_stratified_sample(
        self,
        questions: List[WikiNFQAQuestion],
        n: int = 100,
        seed: int = 42
    ) -> List[WikiNFQAQuestion]:
        """
        Create stratified sample maintaining type distribution.
        
        Args:
            questions: Full question list
            n: Number of samples to take
            seed: Random seed for reproducibility
        
        Returns:
            Stratified sample of questions
        """
        random.seed(seed)

        if n >= len(questions):
            print(f"⚠️  Requested sample size ({n}) >= total questions ({len(questions)})")
            print(f"   Returning all {len(questions)} questions")
            return questions

        
        # Group by category
        by_category = {}
        for q in questions:
            category = q.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(q)
        
        # Calculate samples per category (proportional)
        total = len(questions)
        samples_per_category = {}
        remaining = n
        
        for category, cat_questions in by_category.items():
            proportion = len(cat_questions) / total
            samples = max(1, int(n * proportion))  # At least 1 per category
            samples_per_category[category] = min(samples, len(cat_questions))
            remaining -= samples_per_category[category]
        
        # Distribute remaining samples
        while remaining > 0:
            for category in by_category:
                if remaining > 0 and samples_per_category[category] < len(by_category[category]):
                    samples_per_category[category] += 1
                    remaining -= 1
        
        # Sample from each category
        sampled = []
        for category, count in samples_per_category.items():
            cat_sample = random.sample(by_category[category], count)
            sampled.extend(cat_sample)
        
        # Shuffle final sample
        random.shuffle(sampled)
        
        # Print distribution
        print(f"\n✓ Created stratified sample of {len(sampled)} questions:")
        for category, count in samples_per_category.items():
            print(f"  - {category}: {count}")
        
        return sampled
    
    def get_statistics(self, questions: List[WikiNFQAQuestion]) -> dict:
        """Get statistics about the dataset."""
        by_category = {}
        for q in questions:
            category = q.category
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total": len(questions),
            "by_category": by_category,
            "avg_references": sum(len(q.reference_answers) for q in questions) / len(questions) if questions else 0
        }


class WikiPassageLoader:
    """Loader for Wikipedia passages."""
    
    def __init__(self, data_dir: Path | str = "data"):
        """Initialize loader with data directory."""
        self.data_dir = Path(data_dir)
    
    def load_passages(self) -> List[WikiPassage]:
        """Load Wikipedia passages from passages.jsonl."""
        file_path = self.data_dir / "passages.jsonl"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Passages file not found: {file_path}\n"
                f"You need to create this file first."
            )
        
        passages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                passages.append(WikiPassage.from_dict(data))
        
        return passages
    
    def save_passages(self, passages: List[WikiPassage]) -> None:
        """Save passages to passages.jsonl."""
        file_path = self.data_dir / "passages.jsonl"
        
        with open(file_path, "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p.to_dict()) + "\n")
        
        print(f"✓ Saved {len(passages)} passages to {file_path}")


# Convenience functions
def load_wiki_nfqa(split: str = "test") -> List[WikiNFQAQuestion]:
    """Quick load Wiki-NFQA questions."""
    loader = WikiNFQALoader()
    return loader.load_questions(split)


def load_wiki_passages() -> List[WikiPassage]:
    """Quick load Wikipedia passages."""
    loader = WikiPassageLoader()
    return loader.load_passages()
