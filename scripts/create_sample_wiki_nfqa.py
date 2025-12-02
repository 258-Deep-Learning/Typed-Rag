#!/usr/bin/env python3
"""Create sample Wiki-NFQA data for testing."""

import json
from pathlib import Path

# Sample questions (one per type)
SAMPLE_QUESTIONS = [
    {
        "question_id": "sample_001",
        "question": "What is quantum computing?",
        "category": "Evidence-based",
        "reference_answers": [
            "Quantum computing uses quantum bits that can exist in superposition",
            "Computing paradigm based on quantum mechanical phenomena",
            "Technology leveraging quantum states for computation",
        ]
    },
    {
        "question_id": "sample_002",
        "question": "Python vs Java for web development?",
        "category": "Comparison",
        "reference_answers": [
            "Python: Django/Flask, simpler syntax. Java: Spring, enterprise-ready",
            "Python faster development, Java better performance and scalability",
            "Python more concise, Java more verbose but type-safe",
        ]
    },
    {
        "question_id": "sample_003",
        "question": "What are developers' experiences with Rust?",
        "category": "Experience",
        "reference_answers": [
            "Steep learning curve but worth it for memory safety guarantees",
            "Borrow checker frustrating initially but prevents bugs",
            "Great tooling and community support despite complexity",
        ]
    },
    {
        "question_id": "sample_004",
        "question": "Why is React popular for frontend development?",
        "category": "Reason",
        "reference_answers": [
            "Component-based architecture and virtual DOM for performance",
            "Large ecosystem, strong community, backed by Meta",
            "Declarative syntax and reusable components simplify development",
        ]
    },
    {
        "question_id": "sample_005",
        "question": "How to deploy a machine learning model to production?",
        "category": "Instruction",
        "reference_answers": [
            "1) Containerize model 2) Setup API endpoint 3) Configure scaling 4) Monitor",
            "Use Docker, create REST API, deploy to cloud, implement logging",
            "Package model, expose via Flask/FastAPI, use CI/CD, add monitoring",
        ]
    },
    {
        "question_id": "sample_006",
        "question": "Should companies adopt microservices architecture?",
        "category": "Debate",
        "reference_answers": [
            "Pro: Scalability and flexibility. Con: Operational complexity",
            "Depends on team size and requirements; not always beneficial",
            "Benefits large teams but adds overhead for small projects",
        ]
    },
]

def main():
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    output_file = output_dir / "wiki_nfqa.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for q in SAMPLE_QUESTIONS:
            f.write(json.dumps(q) + "\n")
    
    print(f"✓ Created sample dataset: {output_file}")
    print(f"  {len(SAMPLE_QUESTIONS)} sample questions")
    print("\nNow run: python scripts/setup_wiki_nfqa.py --source local")

if __name__ == "__main__":
    main()