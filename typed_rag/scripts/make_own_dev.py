#!/usr/bin/env python3
"""
Generate a dev set of ~100 questions from your documents.
Auto-derives questions from chunks and assigns type labels.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()


def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    """Load all chunks from JSONL."""
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def generate_question_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a question from a chunk.
    This is a simple heuristic-based approach.
    For production, you'd use an LLM to generate better questions.
    """
    text = chunk["text"][:500]  # Use first 500 chars
    title = chunk.get("title", "Document")
    
    # Simple question templates
    templates = [
        f"What does the document say about {title}?",
        f"Explain the content related to {title}.",
        f"What information is provided about {title}?",
        f"Summarize the key points in {title}.",
        f"What are the main ideas in {title}?",
    ]
    
    question = random.choice(templates)
    
    # Predict a rough type (simplified)
    text_lower = text.lower()
    if any(word in text_lower for word in ["how", "process", "step", "procedure"]):
        q_type = "procedure"
    elif any(word in text_lower for word in ["why", "reason", "because", "cause"]):
        q_type = "reason"
    elif any(word in text_lower for word in ["compare", "versus", "vs", "difference"]):
        q_type = "comparison"
    elif any(word in text_lower for word in ["define", "definition", "meaning", "what is"]):
        q_type = "definition"
    else:
        q_type = "concept"
    
    return {
        "question_id": chunk["id"].replace("::chunk_", "::q_"),
        "question": question,
        "type": q_type,
        "related_doc_id": chunk["doc_id"],
        "related_chunk_id": chunk["id"],
    }


def generate_dev_set(chunks: List[Dict[str, Any]], target_count: int = 100) -> List[Dict[str, Any]]:
    """
    Generate dev set questions.
    Samples diverse chunks and creates questions.
    """
    # Group chunks by doc_id for diversity
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)
    
    # Sample chunks to ensure diversity across documents
    sampled_chunks = []
    docs = list(chunks_by_doc.keys())
    random.shuffle(docs)
    
    # Round-robin sampling from different documents
    idx = 0
    while len(sampled_chunks) < target_count and idx < len(docs) * 10:
        doc_id = docs[idx % len(docs)]
        doc_chunks = chunks_by_doc[doc_id]
        if doc_chunks:
            chunk = doc_chunks.pop(0)
            sampled_chunks.append(chunk)
        idx += 1
    
    # If we still need more, just sample randomly
    if len(sampled_chunks) < target_count:
        remaining = [c for doc_chunks in chunks_by_doc.values() for c in doc_chunks]
        if remaining:
            sampled_chunks.extend(random.sample(remaining, min(target_count - len(sampled_chunks), len(remaining))))
    
    # Generate questions
    questions = []
    for chunk in sampled_chunks[:target_count]:
        q = generate_question_from_chunk(chunk)
        questions.append(q)
    
    return questions


def save_dev_set(questions: List[Dict[str, Any]], output_path: Path) -> None:
    """Save dev set to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    logger.info("Saved dev set", output=str(output_path), count=len(questions))


def main():
    parser = argparse.ArgumentParser(description="Generate dev set from documents")
    parser.add_argument("--root", type=str, required=True, help="Path to chunks.jsonl or data directory")
    parser.add_argument("--out", type=str, required=True, help="Output dev set JSONL file")
    parser.add_argument("--count", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Find chunks.jsonl
    root_path = Path(args.root)
    if root_path.is_file() and root_path.name.endswith(".jsonl"):
        chunks_file = root_path
    else:
        # Try to find chunks.jsonl in the directory
        chunks_file = root_path / "chunks.jsonl"
        if not chunks_file.exists():
            logger.error("Could not find chunks.jsonl", root=str(root_path))
            return
    
    logger.info("Loading chunks", file=str(chunks_file))
    chunks = load_chunks(chunks_file)
    logger.info("Loaded chunks", count=len(chunks))
    
    if not chunks:
        logger.warning("No chunks found")
        return
    
    logger.info("Generating dev set", target_count=args.count)
    questions = generate_dev_set(chunks, target_count=args.count)
    
    save_dev_set(questions, Path(args.out))
    logger.info("Dev set generation complete", total_questions=len(questions))


if __name__ == "__main__":
    main()

