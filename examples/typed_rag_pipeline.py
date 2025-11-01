#!/usr/bin/env python3
"""
Complete Typed-RAG Integration Example

Demonstrates the full pipeline:
1. Question classification
2. Type-aware decomposition
3. Evidence retrieval with reranking
4. Saving outputs (typed_plan.json + evidence_bundle.json)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Typed-RAG components
from typed_rag.classifier import classify_question, QuestionType
from typed_rag.decompose import decompose_question
from typed_rag.retrieval.pipeline import BGEEmbedder
from typed_rag.retrieval.orchestrator import RetrievalOrchestrator


def run_typed_rag_example(
    question: str,
    vector_store,
    vector_store_type: str = "faiss",
    output_dir: Path = Path("./output"),
    use_cache: bool = True,
    rerank: bool = False,
):
    """
    Run the complete Typed-RAG pipeline.

    Args:
        question: Question to process
        vector_store: FAISS or Pinecone store instance
        vector_store_type: "faiss" or "pinecone"
        output_dir: Directory to save outputs
        use_cache: Whether to use caching
        rerank: Whether to use cross-encoder reranking
    """
    print("\n" + "=" * 80)
    print("TYPED-RAG PIPELINE")
    print("=" * 80)
    print(f"\n📝 Question: {question}\n")

    # Step 1: Classify question type
    print("🏷️  Step 1: Classifying question type...")
    qtype = classify_question(question, use_llm=True)
    print(f"   ✓ Type: {qtype}\n")

    # Step 2: Decompose into sub-queries
    print("🔀 Step 2: Decomposing into sub-queries...")
    plan = decompose_question(question, qtype, cache_dir=Path("./cache/decomposition"))
    print(f"   ✓ Generated {len(plan.sub_queries)} sub-queries:")
    for i, sq in enumerate(plan.sub_queries, 1):
        print(f"      [{i}] {sq.aspect}: {sq.query[:60]}...")
    print()

    # Step 3: Retrieve evidence
    print("🔍 Step 3: Retrieving evidence for sub-queries...")
    embedder = BGEEmbedder()
    orchestrator = RetrievalOrchestrator(
        embedder=embedder,
        vector_store=vector_store,
        vector_store_type=vector_store_type,
        cache_dir=Path("./cache/evidence"),
        rerank=rerank,
    )

    bundle = orchestrator.retrieve_evidence(
        plan,
        use_cache=use_cache,
        initial_top_k=50 if rerank else 5,
        final_top_k=5,
    )

    total_docs = sum(len(ev.documents) for ev in bundle.evidence)
    print(f"   ✓ Retrieved {total_docs} total documents\n")

    # Step 4: Save outputs
    print("💾 Step 4: Saving outputs...")
    orchestrator.save_plan_and_evidence(plan, bundle, output_dir)
    print()

    # Print summary
    print("=" * 80)
    print("EVIDENCE SUMMARY")
    print("=" * 80)
    for ev in bundle.evidence:
        print(f"\n{ev.aspect.upper()} ({len(ev.documents)} docs)")
        print(f"  Query: {ev.sub_query}")
        print(f"  Strategy: {ev.strategy}")
        if ev.documents:
            top_doc = ev.documents[0]
            print(f"  Top result: {top_doc.title} (score: {top_doc.score:.4f})")
            print(f"  Text: {top_doc.text[:100]}...")

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - {plan.question_id}_typed_plan.json")
    print(f"  - {plan.question_id}_evidence_bundle.json")
    print()

    return plan, bundle


def demo_all_question_types():
    """Run demo for all 6 question types."""

    # Example questions for each type
    examples = {
        QuestionType.EVIDENCE: "What is quantum computing?",
        QuestionType.COMPARISON: "Python vs Java for web development",
        QuestionType.EXPERIENCE: "Should I invest in stocks or bonds?",
        QuestionType.REASON: "Why is the sky blue?",
        QuestionType.INSTRUCTION: "How to install Python on Mac?",
        QuestionType.DEBATE: "Pros and cons of remote work",
    }

    print("\n" + "=" * 80)
    print("TYPED-RAG DEMO - ALL QUESTION TYPES")
    print("=" * 80)

    for qtype, question in examples.items():
        print(f"\n{'─' * 80}")
        print(f"Testing: {qtype}")
        print(f"{'─' * 80}")
        print(f"Question: {question}")

        # Classify
        classified = classify_question(question, use_llm=False)
        print(f"Classified as: {classified}")

        # Decompose
        plan = decompose_question(question, qtype, cache_dir=Path("./cache/demo"))
        print(f"Sub-queries: {len(plan.sub_queries)}")
        for i, sq in enumerate(plan.sub_queries, 1):
            print(f"  [{i}] {sq.aspect}: {sq.query[:50]}...")

        print()

    print("=" * 80)
    print("✅ Demo complete! All question types tested.")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Check for FAISS index or run demo
    faiss_dir = Path("./typed_rag/indexes/faiss")

    if not faiss_dir.exists():
        print("\n⚠️  No FAISS index found. Running decomposition-only demo...\n")
        demo_all_question_types()
    else:
        print("\n✓ FAISS index found. Running full pipeline...\n")

        # Load FAISS store
        from langchain_community.vectorstores import FAISS as LCFAISS
        from typed_rag.retrieval.pipeline import LCBGEEmbeddings

        embedder = BGEEmbedder()
        lc_embeddings = LCBGEEmbeddings(embedder)

        try:
            faiss_store = LCFAISS.load_local(
                str(faiss_dir),
                lc_embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Loaded FAISS index")

            # Run example
            question = "Python vs Java for web development"
            if len(sys.argv) > 1:
                question = sys.argv[1]

            # Create a wrapper for FAISS store to match expected interface
            class FAISSWrapper:
                def __init__(self, lc_store):
                    self.lc_store = lc_store

                def query(self, query_vec, top_k=5):
                    # Convert to compatible format
                    results = []
                    lc_results = self.lc_store.similarity_search_with_score("", k=top_k)
                    for doc, score in lc_results:
                        md = dict(doc.metadata or {})
                        text = doc.page_content or ""
                        if "text" not in md:
                            md["text"] = text
                        results.append({
                            "id": md.get("id", ""),
                            "score": float(score),
                            "metadata": md
                        })
                    return results

            wrapped_store = FAISSWrapper(faiss_store)

            plan, bundle = run_typed_rag_example(
                question=question,
                vector_store=wrapped_store,
                vector_store_type="faiss",
                output_dir=Path("./output"),
                use_cache=True,
                rerank=True,  # Enable cross-encoder reranking
            )

        except Exception as e:
            print(f"\n⚠️  Error loading FAISS: {e}")
            print("Running decomposition-only demo instead...\n")
            demo_all_question_types()
