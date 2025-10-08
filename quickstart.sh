#!/bin/bash
# Quick start script for Typed-RAG Weekend 1

set -e

echo "🚀 Typed-RAG Weekend 1 Quick Start"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Check for API keys
if [[ -z "$PINECONE_API_KEY" ]]; then
    echo "❌ PINECONE_API_KEY not set!"
    echo "Please run: export PINECONE_API_KEY='your-key'"
    exit 1
fi

if [[ -z "$GOOGLE_API_KEY" ]] && [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠️  Neither GOOGLE_API_KEY nor OPENAI_API_KEY set (baselines will use dummy responses)"
fi

# Check for documents directory
DOCS_DIR="${1:-my_documents}"
if [[ ! -d "$DOCS_DIR" ]]; then
    echo "❌ Documents directory '$DOCS_DIR' not found!"
    echo "Usage: ./quickstart.sh [path_to_documents]"
    echo "Example: ./quickstart.sh my_documents"
    exit 1
fi

echo "📁 Using documents from: $DOCS_DIR"
echo ""

# Run the pipeline
echo "Step 1/6: Setting up directories..."
make setup

echo ""
echo "Step 2/6: Ingesting and chunking documents..."
make ingest DOCS_DIR="$DOCS_DIR"

echo ""
echo "Step 3/6: Building BM25 index..."
make build-bm25

echo ""
echo "Step 4/6: Building Pinecone vector index..."
make build-pinecone

echo ""
echo "Step 5/6: Generating dev set (100 questions)..."
make dev-set

echo ""
echo "Step 6/6: Running baselines..."
make baseline-llm
make baseline-rag

echo ""
echo "✅ All done!"
echo ""
echo "📊 Results saved to:"
echo "   - typed_rag/runs/llm_only.jsonl"
echo "   - typed_rag/runs/rag_baseline.jsonl"
echo ""
echo "🎯 Next steps:"
echo "   - Review results in typed_rag/runs/"
echo "   - Proceed to Weekend 2 (typed decomposition + reranking)"

