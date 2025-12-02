#!/usr/bin/env python3
"""
Baseline 2: Vanilla RAG with In-Memory FAISS Index

This script demonstrates a complete RAG pipeline:
1. Creates synthetic passages from reference answers
2. Builds FAISS index in memory
3. Retrieves relevant passages for each question
4. Generates answers using retrieved context

Usage:
    python scripts/run_rag_baseline.py
"""

from __future__ import annotations
import argparse
import json
import time
import os
import hashlib
from pathlib import Path
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.data.loaders import WikiNFQALoader, WikiNFQAQuestion
from typed_rag.core.keys import get_fastest_model
from typed_rag.retrieval.pipeline import BGEEmbedder, LCBGEEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_passages_from_questions(questions: List[WikiNFQAQuestion]) -> List[Dict[str, Any]]:
    """Create synthetic Wikipedia-style passages from questions and reference answers."""
    passages = []
    
    for i, q in enumerate(questions, 1):
        # Generate passage ID
        passage_id = hashlib.md5(f"{q.question_id}_{i}".encode()).hexdigest()[:12]
        
        # Create title from question
        title = q.question.replace("?", "").strip()
        if len(title) > 60:
            title = title[:57] + "..."
        
        # Use reference answers as passage content
        text = " ".join(q.reference_answers[:3])
        
        passage = {
            "id": passage_id,
            "title": title,
            "text": text,
            "source": "wiki_nfqa_synthetic",
            "question_id": q.question_id,
            "category": q.category
        }
        passages.append(passage)
    
    return passages


def build_faiss_index(passages: List[Dict[str, Any]], embedder: BGEEmbedder) -> FAISS:
    """Build FAISS index from passages in memory."""
    # Prepare documents for LangChain
    docs: List[Document] = []
    for p in passages:
        text = f"{p['title']}: {p['text']}"
        metadata = {k: v for k, v in p.items() if k != "text"}
        docs.append(Document(page_content=text, metadata=metadata))
    
    # Create LangChain embeddings adapter
    lc_embeddings = LCBGEEmbeddings(embedder)
    
    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, lc_embeddings)
    
    return vectorstore


def retrieve_context(question: str, vectorstore: FAISS, top_k: int = 3) -> str:
    """Retrieve top-k relevant passages for a question."""
    # Search using LangChain FAISS
    results = vectorstore.similarity_search(question, k=top_k)
    
    # Format context
    context_parts = []
    for i, doc in enumerate(results, 1):
        context_parts.append(f"[{i}] {doc.page_content}")
    
    return "\n\n".join(context_parts)


def generate_answer_with_rag(question: str, context: str, llm) -> str:
    """Generate answer using retrieved context."""
    prompt = f"""Answer the question based on the provided context. Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Run vanilla RAG baseline with in-memory FAISS")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/wiki_nfqa/dev6.jsonl"),
        help="Input questions file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rag_baseline.jsonl"),
        help="Output answers file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of passages to retrieve"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("="*60)
    print("Vanilla RAG Baseline (In-Memory FAISS)")
    print("="*60)
    print("\n📋 Pipeline Steps:")
    print("  1. Create synthetic passages from reference answers")
    print("  2. Build FAISS index in memory")
    print("  3. Retrieve top-k passages for each question")
    print("  4. Generate answers using retrieved context\n")
    
    # Load model
    model_name = args.model or get_fastest_model()
    print(f"📦 Loading model: {model_name}")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env file")
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=0.0
    )
    
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
    
    # Step 1: Create passages
    print(f"\n📝 Step 1: Creating synthetic passages...")
    passages = create_passages_from_questions(questions)
    print(f"✓ Created {len(passages)} passages")
    
    # Step 2: Build FAISS index
    print(f"\n🔨 Step 2: Building FAISS index with BGE embeddings...")
    embedder = BGEEmbedder(device=None)
    vectorstore = build_faiss_index(passages, embedder)
    print(f"✓ FAISS index built with {len(passages)} documents")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 3 & 4: Retrieve and generate
    results = []
    total_time = 0
    
    print(f"\n🚀 Step 3 & 4: Retrieving context and generating answers...")
    print("-"*60)
    
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q.category}: {q.question[:60]}...")
        
        start = time.time()
        
        # Retrieve context
        context = retrieve_context(q.question, vectorstore, top_k=args.top_k)
        
        # Generate answer
        answer = generate_answer_with_rag(q.question, context, llm)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        result = {
            "question_id": q.question_id,
            "question": q.question,
            "category": q.category,
            "answer": answer,
            "context": context,
            "system": "rag_baseline",
            "model": model_name,
            "latency": elapsed,
            "top_k": args.top_k
        }
        results.append(result)
        
        print(f"  ✓ Retrieved {args.top_k} passages, generated in {elapsed:.2f}s")
    
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
    print(f"Passages created: {len(passages)}")
    print(f"Retrieval top-k: {args.top_k}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency: {total_time/len(results):.2f}s per question")
    print(f"Output saved to: {args.output}")
    print("\n✅ Vanilla RAG baseline complete!")
    
    # Show sample
    print("\n" + "="*60)
    print("Sample Result")
    print("="*60)
    sample = results[0]
    print(f"Question: {sample['question']}")
    print(f"\nRetrieved Context (top {args.top_k}):")
    print(sample['context'][:300] + "...")
    print(f"\nGenerated Answer:")
    print(sample['answer'][:200] + "...")


if __name__ == "__main__":
    main()
