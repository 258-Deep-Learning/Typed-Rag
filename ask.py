#!/usr/bin/env python3
"""
Simple script to ask a single question to your RAG system
Usage: python3 ask.py "What is Amazon's revenue?"
"""

import os
import sys

# Add typed_rag to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "typed_rag"))

import google.generativeai as genai
from retrieval.pipeline import BGEEmbedder, PineconeDenseStore


def main():
    # Get question from command line
    if len(sys.argv) < 2:
        print("Usage: python3 ask.py \"Your question here\"")
        print("Example: python3 ask.py \"What is Amazon's revenue?\"")
        sys.exit(1)
    
    question = sys.argv[1]
    k = 5  # Number of passages to retrieve
    
    # You can override these with environment variables or edit here
    pinecone_index = "typedrag-own"
    pinecone_namespace = "own_docs"
    model_name = "gemini-2.5-flash"
    
    print(f"🔍 Question: {question}")
    print()
    
    # Check API keys
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    
    if not pinecone_key:
        print("❌ PINECONE_API_KEY not set!")
        sys.exit(1)
    
    if not google_key:
        print("❌ GOOGLE_API_KEY not set!")
        sys.exit(1)
    
    # 1. Initialize Pinecone retrieval
    print("📚 Retrieving relevant passages...")
    embedder = BGEEmbedder()
    store = PineconeDenseStore(
        index_name=pinecone_index,
        namespace=pinecone_namespace,
        create_if_missing=False
    )
    
    # Retrieve passages
    query_vec = embedder.encode_queries([question])
    results = store.query(query_vec, top_k=k)
    
    print(f"✓ Retrieved {len(results)} passages")
    print()
    
    # Display retrieved passages
    print("📄 Retrieved Passages:")
    print("-" * 80)
    for i, passage in enumerate(results, 1):
        metadata = passage.get('metadata', {})
        print(f"\n[{i}] {metadata.get('title', 'Untitled')}")
        print(f"Score: {passage.get('score', 0):.4f}")
        text = metadata.get('text', '[No text available]')
        print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
    print("-" * 80)
    print()
    
    # 2. Generate answer with Gemini
    print("💡 Generating answer...")
    
    # Build context from passages
    context = "\n\n".join([
        f"[{i+1}] {p['metadata'].get('text', '')}"
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
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    print("✓ Answer generated")
    print()
    
    # Display answer
    print("🎯 Answer:")
    print("=" * 80)
    print(response.text)
    print("=" * 80)


if __name__ == "__main__":
    main()

