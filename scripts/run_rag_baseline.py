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
from huggingface_hub import InferenceClient


def is_huggingface_model(model_name: str) -> bool:
    """Check if model is from HuggingFace."""
    return "/" in model_name  # HF models have format "org/model-name"


def create_passages_from_questions(questions: List[WikiNFQAQuestion], use_references: bool = False) -> List[Dict[str, Any]]:
    """
    Create synthetic Wikipedia-style passages from questions.
    
    Args:
        questions: List of questions to create passages from
        use_references: If True, uses reference answers (for upper-bound baseline).
                       If False, returns empty passages (requires external knowledge base).
    
    Note: Using reference answers creates an unrealistic upper-bound baseline
          because it retrieves the ground truth answers at test time.
    """
    passages = []
    
    for i, q in enumerate(questions, 1):
        # Generate passage ID
        passage_id = hashlib.md5(f"{q.question_id}_{i}".encode()).hexdigest()[:12]
        
        # Create title from question
        title = q.question.replace("?", "").strip()
        if len(title) > 60:
            title = title[:57] + "..."
        
        # Use reference answers as passage content (if enabled)
        if use_references:
            text = " ".join(q.reference_answers[:3])
        else:
            # Without references, this would need an external knowledge base
            text = ""
        
        passage = {
            "id": passage_id,
            "title": title,
            "text": text,
            "source": "wiki_nfqa_synthetic" if use_references else "empty",
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


def generate_answer_with_rag(question: str, context: str, llm, is_hf: bool = False) -> str:
    """Generate answer using retrieved context."""
    prompt = f"""Answer the question based on the provided context. Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""
    
    if is_hf:
        # HuggingFace API
        response = llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    else:
        # Gemini API
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
    parser.add_argument(
        "--backend",
        type=str,
        default="faiss",
        choices=["faiss", "references"],
        help="Backend to use: 'faiss' for Wikipedia index, 'references' for upper-bound baseline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="wikipedia",
        help="Source name (only used with --backend faiss)"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("="*60)
    if args.backend == "references":
        print("Vanilla RAG Baseline (Upper-Bound with Reference Answers)")
        print("⚠️  WARNING: Using reference answers as knowledge base!")
    else:
        print(f"Vanilla RAG Baseline (Wikipedia FAISS)")
    print("="*60)
    print("\n📋 Pipeline Steps:")
    if args.backend == "references":
        print("  1. Create synthetic passages from reference answers")
        print("     (⚠️  This uses ground truth - unrealistic baseline!)")
        print("  2. Build FAISS index in memory")
    else:
        print(f"  1. Load pre-built FAISS index from indexes/{args.source}/faiss/")
    print("  2. Retrieve top-k passages for each question")
    print("  3. Generate answers using retrieved context\n")
    
    # Load model
    model_name = args.model or get_fastest_model()
    print(f"📦 Loading model: {model_name}")
    
    is_hf = is_huggingface_model(model_name)
    
    if is_hf:
        # HuggingFace model
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN not set in .env file")
        
        llm = InferenceClient(
            model=model_name,
            token=hf_token
        )
        print(f"✓ Using HuggingFace Inference API")
    else:
        # Gemini model
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env file")
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.0
        )
        print(f"✓ Using Google Gemini API")
    
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
    
    # Load or build FAISS index
    if args.backend == "references":
        # Step 1: Create passages from references
        print(f"\n📝 Step 1: Creating synthetic passages from reference answers...")
        print(f"⚠️  Note: This creates an unrealistic upper-bound baseline")
        passages = create_passages_from_questions(questions, use_references=True)
        print(f"✓ Created {len(passages)} passages")
        
        # Step 2: Build FAISS index
        print(f"\n🔨 Step 2: Building FAISS index with BGE embeddings...")
        embedder = BGEEmbedder(device=None)
        vectorstore = build_faiss_index(passages, embedder)
        print(f"✓ FAISS index built with {len(passages)} documents")
    else:
        # Load pre-built Wikipedia FAISS
        from typed_rag.retrieval.pipeline import load_faiss_adapter
        
        print(f"\n📦 Step 1: Loading pre-built FAISS index...")
        faiss_dir = Path(f"indexes/{args.source}/faiss")
        if not faiss_dir.exists():
            print(f"❌ ERROR: FAISS index not found at {faiss_dir}")
            print(f"   Please build the index first using: python typed_rag/scripts/build_faiss.py")
            return
        
        embedder = BGEEmbedder(device=None)
        faiss_adapter = load_faiss_adapter(str(faiss_dir), embedder)
        
        # Use the adapter's underlying LangChain store
        vectorstore = faiss_adapter.store
        print(f"✓ Loaded FAISS index")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if resuming
    processed_ids = set()
    if args.output.exists():
        print(f"\n📂 Found existing output file, loading...")
        with open(args.output, "r") as f:
            for line in f:
                result = json.loads(line.strip())
                processed_ids.add(result["question_id"])
        print(f"   ✓ Already processed {len(processed_ids)} questions, resuming...")
    
    # Step 2/3: Retrieve and generate
    results = []
    total_time = 0
    skipped = 0
    
    print(f"\n🚀 Step 2: Retrieving context and generating answers...")
    print("-"*60)
    
    for i, q in enumerate(questions, 1):
        # Skip if already processed
        if q.question_id in processed_ids:
            skipped += 1
            print(f"[{i}/{len(questions)}] ⏭️  Skipping (already processed): {q.question[:60]}...")
            continue
            
        print(f"[{i}/{len(questions)}] {q.category}: {q.question[:60]}...")
        
        try:
            start = time.time()
            
            # Retrieve context
            context = retrieve_context(q.question, vectorstore, top_k=args.top_k)
            
            # Generate answer
            answer = generate_answer_with_rag(q.question, context, llm, is_hf=is_hf)
            
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
            
            # Save immediately (append mode)
            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            print(f"  ✓ Retrieved {args.top_k} passages, generated in {elapsed:.2f}s (saved)")
            
            # Rate limiting for Gemini (15 RPM limit)
            if not is_hf and elapsed < 4.0:
                time.sleep(4.5 - elapsed)
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
            print(f"  Progress saved. You can resume by running the same command.")
            raise
    
    # Summary message
    if skipped > 0:
        print(f"\n📊 Skipped {skipped} already-processed questions")
    print(f"💾 All results saved to: {args.output}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Questions processed: {len(results)}")
    if args.backend == "references":
        print(f"Passages created: {len(passages)}")
    else:
        print(f"Knowledge source: Wikipedia FAISS index")
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
