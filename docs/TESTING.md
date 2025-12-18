# Typed-RAG Complete Testing Guide

This guide provides comprehensive instructions for testing the Typed-RAG system end-to-end, from unit tests to full integration tests.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Section 1: Unit Tests](#section-1-unit-tests)
- [Section 2: Integration Tests](#section-2-integration-tests)
- [Section 3: End-to-End Pipeline Tests](#section-3-end-to-end-pipeline-tests)
- [Section 4: Evaluation Tests](#section-4-evaluation-tests)
- [Section 5: Performance Tests](#section-5-performance-tests)
- [Section 6: UI/Demo Tests](#section-6-uidemo-tests)
- [Section 7: Smoke Tests](#section-7-smoke-tests)
- [Section 8: Regression Tests](#section-8-regression-tests)
- [Section 9: Complete Test Suite](#section-9-complete-test-suite)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install pytest if not already installed
pip install pytest pytest-cov pytest-xdist
```

### Required Configuration

1. **API Keys**: Set up `.env` file with `GOOGLE_API_KEY`
2. **Data**: Download test dataset (optional for unit tests)
3. **Indexes**: Build vector store index (required for integration tests)

```bash
# Setup API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# Download test data (optional)
python scripts/setup_wiki_nfqa.py --split dev6

# Build FAISS index (for integration tests)
python rag_cli.py build --backend faiss --source wikipedia
```

---

## Quick Start

Run all tests in one command:

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=typed_rag --cov-report=html

# Run specific test file
pytest tests/test_classifier.py -v

# Run specific test
pytest tests/test_decomposition.py::TestQueryDecomposition::test_evidence_1 -v
```

---

## Section 1: Unit Tests

### 1.1 Classifier Unit Tests

**Purpose**: Test question type classification accuracy

**Test File**: `tests/test_classifier.py`

**Run Command**:
```bash
# Run all classifier tests
pytest tests/test_classifier.py -v

# Run with detailed output
pytest tests/test_classifier.py -v -s

# Run with coverage
pytest tests/test_classifier.py --cov=typed_rag.classifier --cov-report=term
```

**What It Tests**:
- Pattern-based classification accuracy
- LLM fallback classification
- Per-type precision, recall, F1 scores
- Confusion matrix generation

**Expected Output**:
```
tests/test_classifier.py::test_pattern_classification PASSED
tests/test_classifier.py::test_llm_classification PASSED
...
Pattern-based accuracy: 66.67%
Gemini accuracy: 100.00%
```

**Test Data**: Uses `data/wiki_nfqa/dev6.jsonl` (6 questions)

---

### 1.2 Decomposition Unit Tests

**Purpose**: Test query decomposition for all 6 question types

**Test File**: `tests/test_decomposition.py`

**Run Command**:
```bash
# Run all decomposition tests (30 tests total)
pytest tests/test_decomposition.py -v

# Run tests for specific question type
pytest tests/test_decomposition.py::TestQueryDecomposition::test_evidence_1 -v
pytest tests/test_decomposition.py::TestQueryDecomposition::test_comparison_1 -v
pytest tests/test_decomposition.py::TestQueryDecomposition::test_reason_1 -v
pytest tests/test_decomposition.py::TestQueryDecomposition::test_instruction_1 -v
pytest tests/test_decomposition.py::TestQueryDecomposition::test_experience_1 -v
pytest tests/test_decomposition.py::TestQueryDecomposition::test_debate_1 -v

# Run all tests for one type
pytest tests/test_decomposition.py -k "evidence" -v
pytest tests/test_decomposition.py -k "comparison" -v
```

**What It Tests**:
- **Evidence-based** (5 tests): Passthrough strategy, 1 sub-query
- **Comparison** (5 tests): Subject extraction, 3-5 sub-queries
- **Reason** (5 tests): Causal factors, 2-4 sub-queries
- **Instruction** (5 tests): Sequential steps, 3-5 sub-queries
- **Experience** (5 tests): Topical angles, 2-4 sub-queries
- **Debate** (5 tests): Opposing views, 2-3 sub-queries

**Expected Output**:
```
tests/test_decomposition.py::TestQueryDecomposition::test_evidence_1 PASSED
tests/test_decomposition.py::TestQueryDecomposition::test_evidence_2 PASSED
...
tests/test_decomposition.py::TestQueryDecomposition::test_debate_5 PASSED
======================== 30 passed in 2.34s ========================
```

**Test Coverage**: 30 tests total (5 per question type)

---

### 1.3 Generation Unit Tests

**Purpose**: Test answer generation and aggregation

**Test File**: `tests/test_generation.py`

**Run Command**:
```bash
# Run all generation tests
pytest tests/test_generation.py -v

# Run specific test
pytest tests/test_generation.py::test_generator_fallback_produces_citations -v
pytest tests/test_generation.py::test_aggregator_debate_structure -v
pytest tests/test_generation.py::test_instruction_aggregator_creates_numbered_steps -v
```

**What It Tests**:
- Fallback generator produces citations
- Debate aggregator creates proper structure
- Instruction aggregator creates numbered steps
- Type-aware answer formatting

**Expected Output**:
```
tests/test_generation.py::test_generator_fallback_produces_citations PASSED
tests/test_generation.py::test_aggregator_debate_structure PASSED
tests/test_generation.py::test_instruction_aggregator_creates_numbered_steps PASSED
======================== 3 passed in 0.45s ========================
```

---

### 1.4 Run All Unit Tests

```bash
# Run all unit tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=typed_rag --cov-report=term-missing

# Run in parallel (faster)
pytest tests/ -n auto

# Run with HTML coverage report
pytest tests/ --cov=typed_rag --cov-report=html
# Open htmlcov/index.html in browser
```

**Expected Result**: All 33+ unit tests pass

---

## Section 2: Integration Tests

### 2.1 Full Pipeline Integration Test

**Purpose**: Test complete pipeline from question to answer

**Test File**: `examples/typed_rag_pipeline.py`

**Run Command**:
```bash
# Test with a single question
python examples/typed_rag_pipeline.py "What is quantum computing?"

# Test with different question types
python examples/typed_rag_pipeline.py "Python vs Java for web development?"
python examples/typed_rag_pipeline.py "Why is React popular?"
python examples/typed_rag_pipeline.py "How to deploy a machine learning model?"
```

**What It Tests**:
- End-to-end pipeline execution
- Classification → Decomposition → Retrieval → Generation
- Output file generation (typed_plan.json, evidence_bundle.json)
- Error handling

**Expected Output**:
```
TYPED-RAG PIPELINE
================================================================================

📝 Question: What is quantum computing?

Step 1: Classifying question type...
✓ Classified as: Evidence-based

Step 2: Decomposing question...
✓ Generated 1 sub-queries

Step 3: Retrieving evidence...
✓ Retrieved 5 documents

Step 4: Generating answer...
✓ Generated answer

✓ Pipeline complete!
```

**Output Files**: `output/{question_id}_typed_plan.json`, `output/{question_id}_evidence_bundle.json`

---

### 2.2 Happy Flow Demo Test

**Purpose**: Test all 6 question types in one run

**Test File**: `examples/happy_flow_demo.py`

**Run Command**:
```bash
# Run demo for all question types
python examples/happy_flow_demo.py --backend faiss --source wikipedia

# Run with specific backend
python examples/happy_flow_demo.py --backend faiss --source own_docs
```

**What It Tests**:
- All 6 question types (Evidence, Comparison, Reason, Instruction, Experience, Debate)
- System handles different question types correctly
- Output generation for each type

**Expected Output**: 6 questions processed, one per type

---

### 2.3 CLI Integration Test

**Purpose**: Test command-line interface

**Run Command**:
```bash
# Test typed-ask command
python rag_cli.py typed-ask "What is machine learning?" --backend faiss --source wikipedia

# Test with different question types
python rag_cli.py typed-ask "Python vs Java?" --backend faiss --source wikipedia
python rag_cli.py typed-ask "Why use Docker?" --backend faiss --source wikipedia

# Test build command
python rag_cli.py build --backend faiss --source wikipedia
```

**What It Tests**:
- CLI argument parsing
- System initialization
- Error handling
- Output formatting

---

## Section 3: End-to-End Pipeline Tests

### 3.1 Single Question E2E Test

**Purpose**: Test complete system with one question

**Run Command**:
```bash
# Test with Typed-RAG system
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_typed_rag.jsonl \
  --backend faiss \
  --source wikipedia

# Verify output
cat results/test_typed_rag.jsonl | jq '.answer' | head -n 1
```

**What It Tests**:
- Complete pipeline execution
- Answer generation
- Output file creation
- Error handling

---

### 3.2 Multi-Question E2E Test

**Purpose**: Test system with multiple questions

**Run Command**:
```bash
# Process all questions in dev6
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_full_pipeline.jsonl \
  --backend faiss \
  --source wikipedia

# Check results
cat results/test_full_pipeline.jsonl | jq '.question, .answer' | head -n 20
```

**Expected Output**: All 6 questions processed successfully

---

### 3.3 System Comparison E2E Test

**Purpose**: Compare all three systems (LLM-Only, RAG Baseline, Typed-RAG)

**Run Command**:
```bash
# Test LLM-Only
python scripts/run_llm_only.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_llm_only.jsonl

# Test RAG Baseline
python scripts/run_rag_baseline.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_rag_baseline.jsonl \
  --backend faiss \
  --source wikipedia

# Test Typed-RAG
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_typed_rag.jsonl \
  --backend faiss \
  --source wikipedia

# Compare results
echo "LLM-Only answers:"
cat results/test_llm_only.jsonl | jq '.answer' | head -n 2

echo "RAG Baseline answers:"
cat results/test_rag_baseline.jsonl | jq '.answer' | head -n 2

echo "Typed-RAG answers:"
cat results/test_typed_rag.jsonl | jq '.answer' | head -n 2
```

**What It Tests**:
- All three systems work correctly
- Output format consistency
- Performance differences

---

## Section 4: Evaluation Tests

### 4.1 Classifier Evaluation Test

**Purpose**: Evaluate classifier accuracy

**Run Command**:
```bash
# Run classifier evaluation
python -m typed_rag.classifier.classifier \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_classifier_eval.json

# Check results
cat results/test_classifier_eval.json | jq '.pattern_only.accuracy'
cat results/test_classifier_eval.json | jq '.gemini.accuracy'
```

**Expected Output**:
- Pattern-based accuracy: ~66.67%
- Gemini accuracy: ~100%

---

### 4.2 Linkage Evaluation Test

**Purpose**: Evaluate answer quality (MRR/MPR)

**Run Command**:
```bash
# First, generate answers
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_for_linkage.jsonl \
  --backend faiss \
  --source wikipedia

# Then evaluate
python scripts/evaluate_linkage.py \
  --systems results/test_for_linkage.jsonl \
  --output results/test_linkage_eval.json

# Check results
cat results/test_linkage_eval.json | jq '.overall.mrr'
cat results/test_linkage_eval.json | jq '.overall.mpr'
```

**Expected Output**:
- MRR: 0.0-1.0 (depends on reference answers)
- MPR: 0-100% (depends on reference answers)

---

### 4.3 Ablation Study Test

**Purpose**: Test ablation study variants

**Run Command**:
```bash
# Run full ablation study
python scripts/run_ablation_study.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_ablation/ \
  --backend faiss \
  --source wikipedia

# Check summary
cat results/test_ablation/summary.json | jq '.'

# Test specific variant
python scripts/run_ablation_study.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_ablation/ \
  --variants full no_classification \
  --backend faiss \
  --source wikipedia
```

**What It Tests**:
- Full system (all components)
- No classification variant
- No decomposition variant
- No retrieval variant

**Expected Output**: Summary JSON with success rates and latencies

---

## Section 5: Performance Tests

### 5.1 Performance Profiling Test

**Purpose**: Measure system performance metrics

**Run Command**:
```bash
# Profile performance
python scripts/profile_performance.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_performance.json \
  --backend faiss \
  --source wikipedia

# Check results
cat results/test_performance.json | jq '.throughput'
cat results/test_performance.json | jq '.memory'
cat results/test_performance.json | jq '.component_latency'
```

**What It Tests**:
- Throughput (questions/second)
- Memory usage (peak, average)
- Component latency breakdown
- Success rates

**Expected Output**:
```json
{
  "throughput": {
    "questions_per_second": 0.26,
    "average_latency_seconds": 3.81
  },
  "memory": {
    "peak_mb": 450.2,
    "average_per_question_mb": 75.0
  }
}
```

---

### 5.2 Latency Benchmark Test

**Purpose**: Measure latency for different question types

**Run Command**:
```bash
# Test each question type
for qtype in "Evidence" "Comparison" "Reason" "Instruction" "Experience" "Debate"; do
  echo "Testing $qtype..."
  time python examples/typed_rag_pipeline.py "Test $qtype question"
done
```

**Expected Output**: Latency measurements for each type

---

### 5.3 Memory Leak Test

**Purpose**: Check for memory leaks during long runs

**Run Command**:
```bash
# Run multiple questions and monitor memory
python scripts/profile_performance.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_memory.json \
  --backend faiss \
  --source wikipedia

# Check if memory increases linearly
cat results/test_memory.json | jq '.per_question_metrics[].memory.delta_mb'
```

**Expected Output**: Memory delta should be consistent across questions

---

## Section 6: UI/Demo Tests

### 6.1 Streamlit App Test

**Purpose**: Test interactive UI

**Run Command**:
```bash
# Start Streamlit app
streamlit run app.py

# Then manually test:
# 1. Navigate to http://localhost:8501
# 2. Test Query Interface tab
# 3. Test Evaluation Results tab
# 4. Test Ablation Study tab
# 5. Try different question types
# 6. Test all three systems (LLM-Only, RAG Baseline, Typed-RAG)
```

**What To Test**:
- ✅ App loads without errors
- ✅ All three tabs are accessible
- ✅ Questions can be submitted
- ✅ Results are displayed correctly
- ✅ Metrics are shown
- ✅ No JavaScript errors in browser console

---

### 6.2 UI Functionality Test

**Manual Test Checklist**:

```bash
# 1. Test Query Interface
# - Enter question: "What is quantum computing?"
# - Select "Typed-RAG" system
# - Click "Ask Question"
# - Verify: Classification, Decomposition, Retrieval steps shown
# - Verify: Final answer displayed
# - Verify: Metrics shown (Question Type, Aspects, Total Docs)

# 2. Test System Comparison
# - Test "LLM-Only" system
# - Test "RAG Baseline" system
# - Test "Typed-RAG" system
# - Compare answers and latency

# 3. Test Evaluation Results Tab
# - Verify: Linkage evaluation table shown
# - Verify: Classifier metrics displayed
# - Verify: Per-type performance table visible

# 4. Test Ablation Study Tab
# - Verify: Performance summary table shown
# - Verify: Quality metrics (MRR/MPR) displayed
# - Verify: Key insights section visible
```

---

## Section 7: Smoke Tests

### 7.1 Quick Smoke Test

**Purpose**: Verify system is working (fastest test)

**Run Command**:
```bash
# Quick test - single question, no LLM
python -c "
from typed_rag.classifier import classify_question
from typed_rag.decompose import decompose_question

# Test classification (pattern-only, no API call)
qtype = classify_question('What is Python?', use_llm=False)
print(f'✓ Classification: {qtype}')

# Test decomposition (no API call)
plan = decompose_question('What is Python?', qtype, use_cache=False)
print(f'✓ Decomposition: {len(plan.sub_queries)} sub-queries')
print('✓ Smoke test passed!')
"
```

**Expected Output**: Quick verification that core components work

---

### 7.2 Import Test

**Purpose**: Verify all imports work

**Run Command**:
```bash
# Test all imports
python -c "
from typed_rag.classifier import classify_question, QuestionType
from typed_rag.decompose import decompose_question, QueryDecomposer
from typed_rag.retrieval.pipeline import BGEEmbedder
from typed_rag.retrieval.orchestrator import RetrievalOrchestrator
from typed_rag.generation.generator import TypedAnswerGenerator
from typed_rag.generation.aggregator import TypedAnswerAggregator
from typed_rag.rag_system import ask_typed_question, DataType
print('✓ All imports successful!')
"
```

**Expected Output**: No import errors

---

### 7.3 Configuration Test

**Purpose**: Verify configuration is correct

**Run Command**:
```bash
# Test API key
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
print(f'✓ API Key: {\"Set\" if api_key else \"Missing\"}')

# Test data directory
from pathlib import Path
data_dir = Path('data/wiki_nfqa')
print(f'✓ Data directory: {\"Exists\" if data_dir.exists() else \"Missing\"}')

# Test index
index_dir = Path('indexes/wikipedia/faiss')
print(f'✓ Index directory: {\"Exists\" if index_dir.exists() else \"Missing\"}')
"
```

---

## Section 8: Regression Tests

### 8.1 Regression Test Suite

**Purpose**: Ensure existing functionality still works

**Run Command**:
```bash
# Run all regression tests
pytest tests/ -v --tb=short

# Run with previous results comparison
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/regression_test.jsonl \
  --backend faiss \
  --source wikipedia

# Compare with baseline
# (Save baseline results first, then compare)
```

---

### 8.2 Backward Compatibility Test

**Purpose**: Ensure API changes don't break existing code

**Run Command**:
```bash
# Test old API still works
python -c "
# Old API style
from typed_rag.classifier import classify_question
from typed_rag.decompose import decompose_question

qtype = classify_question('Test', use_llm=False)
plan = decompose_question('Test', qtype)
print('✓ Backward compatibility: OK')
"
```

---

## Section 9: Complete Test Suite

### 9.1 Run All Tests Script

Create `scripts/run_all_tests.sh`:

```bash
#!/bin/bash
# Complete test suite runner

set -e  # Exit on error

echo "🧪 Typed-RAG Complete Test Suite"
echo "================================"

# Section 1: Unit Tests
echo ""
echo "📋 Section 1: Unit Tests"
echo "------------------------"
pytest tests/ -v --tb=short || exit 1

# Section 2: Integration Tests
echo ""
echo "📋 Section 2: Integration Tests"
echo "------------------------"
python examples/typed_rag_pipeline.py "What is machine learning?" || exit 1

# Section 3: E2E Tests
echo ""
echo "📋 Section 3: End-to-End Tests"
echo "------------------------"
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_e2e.jsonl \
  --backend faiss \
  --source wikipedia || exit 1

# Section 4: Evaluation Tests
echo ""
echo "📋 Section 4: Evaluation Tests"
echo "------------------------"
python -m typed_rag.classifier.classifier \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_classifier.json || exit 1

# Section 5: Performance Tests
echo ""
echo "📋 Section 5: Performance Tests"
echo "------------------------"
python scripts/profile_performance.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/test_perf.json \
  --backend faiss \
  --source wikipedia || exit 1

echo ""
echo "✅ All tests passed!"
```

**Run Command**:
```bash
chmod +x scripts/run_all_tests.sh
./scripts/run_all_tests.sh
```

---

### 9.2 Continuous Integration Test

**Purpose**: Tests suitable for CI/CD

**Run Command**:
```bash
# Fast CI tests (no API calls)
pytest tests/test_decomposition.py -v
pytest tests/test_generation.py -v

# With coverage
pytest tests/ --cov=typed_rag --cov-report=xml --cov-report=term

# Linting (if configured)
# flake8 typed_rag/
# black --check typed_rag/
# mypy typed_rag/
```

---

## Troubleshooting

### Issue: Tests Fail with Import Errors

**Solution**:
```bash
# Ensure you're in project root
cd /path/to/Typed-Rag

# Install dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

---

### Issue: API Quota Exceeded

**Solution**:
```bash
# Use pattern-only classification (no API calls)
pytest tests/test_decomposition.py -v  # No API calls

# Skip LLM-based tests
pytest tests/ -v -k "not llm" --ignore=tests/test_classifier.py
```

---

### Issue: Missing Test Data

**Solution**:
```bash
# Download test data
python scripts/setup_wiki_nfqa.py --split dev6

# Create sample data
python scripts/create_sample_wiki_nfqa.py --output data/test_sample.jsonl
```

---

### Issue: Index Not Found

**Solution**:
```bash
# Build FAISS index
python rag_cli.py build --backend faiss --source wikipedia

# Verify index exists
ls -lh indexes/wikipedia/faiss/
```

---

### Issue: Slow Tests

**Solution**:
```bash
# Run tests in parallel
pytest tests/ -n auto

# Skip slow tests
pytest tests/ -v -m "not slow"

# Use caching
# Tests will use cache if available
```

---

## Test Coverage Goals

| Component | Target Coverage | Current |
|-----------|----------------|---------|
| Classifier | 90%+ | ~85% |
| Decomposition | 95%+ | ~90% |
| Retrieval | 80%+ | ~75% |
| Generation | 85%+ | ~80% |
| Overall | 85%+ | ~82% |

**Check Coverage**:
```bash
pytest tests/ --cov=typed_rag --cov-report=html
open htmlcov/index.html
```

---

## Test Execution Time Estimates

| Test Section | Estimated Time | Notes |
|--------------|----------------|-------|
| Unit Tests | 5-10 seconds | Fast, no API calls |
| Integration Tests | 30-60 seconds | Single question |
| E2E Tests | 2-5 minutes | 6 questions |
| Evaluation Tests | 5-10 minutes | Full evaluation |
| Performance Tests | 3-5 minutes | Profiling overhead |
| Complete Suite | 15-25 minutes | All tests |

---

## Best Practices

1. **Run Unit Tests First**: Fastest feedback
2. **Use Caching**: Speeds up repeated tests
3. **Test Incrementally**: After each change
4. **Check Coverage**: Aim for 85%+ overall
5. **Document Failures**: Note expected vs actual
6. **Update Tests**: When adding features

---

## Quick Reference

```bash
# Most common commands
pytest tests/ -v                    # Run all unit tests
pytest tests/test_decomposition.py  # Run decomposition tests
python examples/typed_rag_pipeline.py "Question?"  # E2E test
python scripts/profile_performance.py --input data/wiki_nfqa/dev6.jsonl  # Performance
streamlit run app.py                # UI test
```

---

**Last Updated**: January 2025  
**Maintained By**: Typed-RAG Team  
**Questions**: See [EVALUATION.md](EVALUATION.md) or create an issue
