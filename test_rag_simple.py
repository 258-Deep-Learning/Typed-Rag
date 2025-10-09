#!/usr/bin/env python3
"""
Simple RAG test - Retrieve from Pinecone and answer with Gemini
"""

import sys
import os

sys.path.insert(0, 'typed_rag')

from retrieval.pipeline import BGEEmbedder, PineconeDenseStore
import google.generativeai as genai

def rag_answer(question: str, top_k: int = 3):
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from Pinecone
    2. Feed to Gemini for answer generation
    """
    
    print("🤖 RAG Pipeline Test")
    print("=" * 70)
    print(f"Question: {question}")
    print("=" * 70)
    print()
    
    # Check API keys
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not set!")
        return
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set!")
        return
    
    try:
        # Step 1: Retrieve from Pinecone
        print(f"📚 Step 1: Retrieving top {top_k} relevant chunks from Pinecone...")
        
        embedder = BGEEmbedder()
        store = PineconeDenseStore(
            index_name='typedrag-own',
            namespace='own_docs',
            create_if_missing=False
        )
        
        qv = embedder.encode_queries([question])
        results = store.query(qv, top_k=top_k)
        
        if not results:
            print("No results found!")
            return
        
        print(f"✓ Found {len(results)} relevant chunks")
        print()
        
        # Load the actual text from chunks
        import json
        passages = []
        for i, r in enumerate(results, 1):
            chunk_id = r['id']
            score = r['score']
            metadata = r['metadata']
            
            # Get the full text
            with open('typed_rag/data/chunks.jsonl', 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    if chunk['id'] == chunk_id:
                        passages.append({
                            'title': metadata.get('title', 'Unknown'),
                            'text': chunk['text'],
                            'score': score
                        })
                        print(f"  [{i}] {metadata.get('title', 'Unknown')} (score: {score:.4f})")
                        break
        
        print()
        
        # Step 2: Format context for Gemini
        print("📝 Step 2: Formatting context for LLM...")
        
        context = "\n\n".join([
            f"[Passage {i}] {p['title']}\n{p['text']}"
            for i, p in enumerate(passages, 1)
        ])
        
        prompt = f"""Answer the question based on the following passages from the documents:

{context}

Question: {question}

Please provide a comprehensive answer based on the passages above. If the passages don't contain enough information, say so."""
        
        print("✓ Context prepared")
        print()
        
        # Step 3: Get answer from Gemini
        print("🧠 Step 3: Generating answer with Gemini...")
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                candidate_count=1,
            )
        )
        
        answer = response.text.strip()
        
        print("✓ Answer generated")
        print()
        print("=" * 70)
        print("📖 ANSWER:")
        print("=" * 70)
        print(answer)
        print()
        print("=" * 70)
        print("📚 SOURCES:")
        print("=" * 70)
        for i, p in enumerate(passages, 1):
            print(f"[{i}] {p['title']} (relevance: {p['score']:.4f})")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def llm_only_answer(question: str):
    """
    LLM-only baseline (no retrieval)
    """
    
    print("🤖 LLM-Only Baseline (No Retrieval)")
    print("=" * 70)
    print(f"Question: {question}")
    print("=" * 70)
    print()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set!")
        return
    
    try:
        print("🧠 Generating answer without any context...")
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        prompt = f"Please answer the following question concisely:\n\n{question}"
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                candidate_count=1,
            )
        )
        
        answer = response.text.strip()
        
        print()
        print("=" * 70)
        print("📖 ANSWER (No Context):")
        print("=" * 70)
        print(answer)
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def compare(question: str):
    """Compare LLM-only vs RAG"""
    
    print("\n" + "🔬 COMPARISON: LLM-Only vs RAG" + "\n")
    print("=" * 70)
    
    # LLM-only first
    llm_only_answer(question)
    
    print("\n" + "vs" + "\n")
    
    # RAG
    rag_answer(question)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 test_rag_simple.py <question>")
        print("  python3 test_rag_simple.py --compare <question>")
        print()
        print("Examples:")
        print("  python3 test_rag_simple.py 'What experience does Indraneel have?'")
        print("  python3 test_rag_simple.py --compare 'Tell me about Toyota projects'")
        return
    
    if sys.argv[1] == '--compare':
        question = " ".join(sys.argv[2:])
        compare(question)
    else:
        question = " ".join(sys.argv[1:])
        rag_answer(question)


if __name__ == "__main__":
    main()

