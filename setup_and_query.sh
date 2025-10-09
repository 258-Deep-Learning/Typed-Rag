#!/bin/bash
# Complete workflow: Ingest documents → Build Pinecone index → Query
# Usage: ./setup_and_query.sh

set -e

echo "🚀 Complete RAG Setup and Query Pipeline"
echo "=========================================="
echo ""

# Configuration
DOCS_DIR="my-documents"
CHUNKS_FILE="typed_rag/data/chunks.jsonl"
PINECONE_INDEX="typedrag-own"
PINECONE_NAMESPACE="own_docs"

# Check API keys
if [ -z "$PINECONE_API_KEY" ]; then
    echo "❌ PINECONE_API_KEY not set!"
    echo "Run: export PINECONE_API_KEY='your-key'"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set!"
    echo "Run: export GOOGLE_API_KEY='your-key'"
    exit 1
fi

# Step 1: Ingest documents
echo "Step 1/2: Ingesting documents from $DOCS_DIR..."
python3 typed_rag/scripts/ingest_own_docs.py \
  --root "$DOCS_DIR" \
  --out "$CHUNKS_FILE" \
  --chunk_tokens 200 \
  --stride_tokens 60
echo "✓ Documents ingested and chunked"
echo ""

# Step 2: Build Pinecone index
echo "Step 2/2: Building Pinecone index..."
python3 typed_rag/scripts/build_pinecone.py \
  --in "$CHUNKS_FILE" \
  --index "$PINECONE_INDEX" \
  --namespace "$PINECONE_NAMESPACE"
echo "✓ Pinecone index built"
echo ""

echo "✅ Setup complete!"
echo ""
echo "📚 Your documents are now indexed in Pinecone"
echo "   Index: $PINECONE_INDEX"
echo "   Namespace: $PINECONE_NAMESPACE"
echo ""
echo "💬 Now you can query using:"
echo "   python3 ask.py \"Your question here\""
echo ""
echo "Example:"
echo "   python3 ask.py \"What does the document say about Amazon?\""
echo ""

