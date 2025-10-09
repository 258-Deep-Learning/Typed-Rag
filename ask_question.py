#!/usr/bin/env python3
"""
Simple script to ask a single question to your RAG system
Usage: python3 ask_question.py "What is Amazon's revenue?"
"""

import os
import sys
import tyro
from dataclasses import dataclass
from typing import Optional
from pinecone import Pinecone
import google.generativeai as genai

# Add typed_rag to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "typed_rag"))

from retrieval.pipeline import BGEEmbedder, PineconeDenseStore


@dataclass
class Args:
    pinecone_index: str = "typedrag-own"
    """Pinecone index name"""
    
    pinecone_namespace: str = "own_docs"
    """Pinecone namespace"""
    
    k: int = 5
    """Number of passages to retrieve"""
    
    model: str = "gemini-2.5-flash"
    """Gemini model to use"""


def main(question: str, args: Args):
    """Ask a single question to your RAG system.
    
    Args:
        question: The question to ask
        args: Configuration options
    """
    print(f"🔍 Question: {question}")
    print()
    
    # Check API keys
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    
    if not pinecone_key:
        print("❌ PINECONE_API_KEY not set!")
        return
    
    if not google_key:
        print("❌ GOOGLE_API_KEY not set!")
        return
    
    # 1. Initialize Pinecone retrieval
    print("📚 Retrieving relevant passages...")
    embedder = BGEEmbedder()
    pc = Pinecone(api_key=pinecone_key)
    store = PineconeDenseStore(
        pc=pc,
        index_name=args.pinecone_index,
        namespace=args.pinecone_namespace,
        embedder=embedder
    )
    
    # Retrieve passages
    results = store.search(question, top_k=args.k)
    
    print(f"✓ Retrieved {len(results)} passages")
    print()
    
    # Display retrieved passages
    print("📄 Retrieved Passages:")
    print("-" * 80)
    for i, passage in enumerate(results, 1):
        print(f"\n[{i}] {passage.get('title', 'Untitled')}")
        print(f"Score: {passage.get('score', 0):.4f}")
        print(f"Text: {passage['text'][:200]}..." if len(passage['text']) > 200 else f"Text: {passage['text']}")
    print("-" * 80)
    print()
    
    # 2. Generate answer with Gemini
    print("💡 Generating answer...")
    
    # Build context from passages
    context = "\n\n".join([
        f"[{i+1}] {p['text']}"
        for i, p in enumerate(results)
    ])
    
    # Build prompt
    prompt = f"""Answer the question based on the following passages.

Passages:
{context}

Question: {question}

Answer (be concise and cite passage numbers):"""
    
    # Call Gemini
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel(args.model)
    response = model.generate_content(prompt)
    
    print("✓ Answer generated")
    print()
    
    # Display answer
    print("🎯 Answer:")
    print("=" * 80)
    print(response.text)
    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)

