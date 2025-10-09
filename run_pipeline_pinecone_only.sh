#!/bin/bash
# Pipeline runner - Pinecone only (skips BM25)
# Use this if you have Java version issues

set -e

echo "🚀 Running Typed-RAG Pipeline (Pinecone Only)"
echo "=============================================="
echo ""

# Build Pinecone
echo "Step 1/4: Building Pinecone vector index..."
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own \
  --namespace own_docs
echo "✓ Pinecone index built"
echo ""

# Generate dev set
echo "Step 2/3: Generating dev set..."
python3 typed_rag/scripts/make_own_dev.py \
  --root typed_rag/data/chunks.jsonl \
  --out typed_rag/data/dev_set.jsonl \
  --count 16
echo "✓ Dev set generated"
echo ""

# Run LLM-only baseline
echo "Step 3/3: Running LLM-only baseline..."
python3 typed_rag/scripts/run_llm_only.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/llm_only.jsonl \
  --model gemini-2.5-flash
echo "✓ LLM-only baseline complete"
echo ""

echo "✅ Pipeline complete!"
echo ""
echo "📊 Results:"
echo "   - typed_rag/runs/llm_only.jsonl (no retrieval)"
echo ""
echo "ℹ️  Note: Pinecone index built and ready for querying"
echo "   Use ask_question.py or integrate with your own LLM setup"

