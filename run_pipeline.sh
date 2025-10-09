#!/bin/bash
# Simple pipeline runner - run each step manually
# Make sure to: source venv/bin/activate && source .env first

set -e

echo "🚀 Running Typed-RAG Pipeline"
echo "=============================="
echo ""

# Step 2: Build BM25
echo "Step 1/4: Building BM25 index..."
python3 typed_rag/scripts/build_bm25.py \
  --in typed_rag/data/chunks.jsonl \
  --index typed_rag/indexes/lucene_own
echo "✓ BM25 index built"
echo ""

# Step 3: Build Pinecone
echo "Step 2/4: Building Pinecone vector index..."
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own \
  --namespace own_docs
echo "✓ Pinecone index built"
echo ""

# Step 4: Generate dev set
echo "Step 3/4: Generating dev set..."
python3 typed_rag/scripts/make_own_dev.py \
  --root typed_rag/data/chunks.jsonl \
  --out typed_rag/data/dev_set.jsonl \
  --count 100
echo "✓ Dev set generated"
echo ""

# Step 5: Run baselines
echo "Step 4/4: Running baselines..."
echo "  → LLM-only baseline..."
python3 typed_rag/scripts/run_llm_only.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/llm_only.jsonl \
  --model gemini-2.5-flash

echo "  → RAG baseline..."
python3 typed_rag/scripts/run_rag_baseline.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/rag_baseline.jsonl \
  --pinecone_index typedrag-own \
  --pinecone_namespace own_docs \
  --bm25_index typed_rag/indexes/lucene_own \
  --k 5 \
  --model gemini-2.5-flash
echo "✓ Baselines complete"
echo ""

echo "✅ Pipeline complete!"
echo ""
echo "📊 Results:"
echo "   - typed_rag/runs/llm_only.jsonl"
echo "   - typed_rag/runs/rag_baseline.jsonl"

