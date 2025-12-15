#!/bin/bash
# Complete test suite runner for Typed-RAG
# Runs all tests in sequence: unit, integration, E2E, evaluation, and performance

set -e  # Exit on error

echo "🧪 Typed-RAG Complete Test Suite"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "README_TYPED_RAG.md" ]; then
    echo "❌ Error: Must run from Typed-Rag project root"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Section 1: Unit Tests
echo "📋 Section 1: Unit Tests"
echo "------------------------"
echo "Running pytest on tests/ directory..."
pytest tests/ -v --tb=short || {
    echo "❌ Unit tests failed"
    exit 1
}
echo "✓ Unit tests passed"
echo ""

# Section 2: Integration Tests
echo "📋 Section 2: Integration Tests"
echo "------------------------"
echo "Testing full pipeline with single question..."
python examples/typed_rag_pipeline.py "What is machine learning?" || {
    echo "❌ Integration test failed"
    exit 1
}
echo "✓ Integration test passed"
echo ""

# Section 3: E2E Tests
echo "📋 Section 3: End-to-End Tests"
echo "------------------------"
echo "Running E2E test with dev6 dataset..."
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_e2e.jsonl \
  --backend faiss \
  --source wikipedia || {
    echo "❌ E2E tests failed"
    exit 1
}
echo "✓ E2E tests passed"
echo ""

# Section 4: Evaluation Tests
echo "📋 Section 4: Evaluation Tests"
echo "------------------------"
echo "Running classifier evaluation..."
python -m typed_rag.classifier.classifier \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_classifier.json || {
    echo "❌ Classifier evaluation failed"
    exit 1
}
echo "✓ Classifier evaluation passed"
echo ""

# Section 5: Performance Tests
echo "📋 Section 5: Performance Tests"
echo "------------------------"
echo "Running performance profiling..."
python scripts/profile_performance.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_perf.json \
  --backend faiss \
  --source wikipedia || {
    echo "❌ Performance tests failed"
    exit 1
}
echo "✓ Performance tests passed"
echo ""

# Summary
echo "================================"
echo "✅ All tests passed successfully!"
echo "================================"
echo ""
echo "Test results saved to:"
echo "  - Unit tests: pytest output"
echo "  - E2E results: results/test_e2e.jsonl"
echo "  - Classifier: results/test_classifier.json"
echo "  - Performance: results/test_perf.json"
echo ""
echo "Next steps:"
echo "  - Review test results"
echo "  - Generate plots: python scripts/create_evaluation_plots.py"
echo "  - View in UI: streamlit run app.py"
