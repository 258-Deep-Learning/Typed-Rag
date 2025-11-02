# Typed-RAG: Type-Aware Query Decomposition

Complete implementation of the Typed-RAG system for decomposing and retrieving evidence for non-factoid questions.

## 🎯 Overview

Typed-RAG improves RAG performance on non-factoid questions by:

1. **Type Classification** - Classifies questions into 6 types (Evidence, Comparison, Experience, Reason, Instruction, Debate)
2. **Type-Aware Decomposition** - Decomposes questions into targeted sub-queries based on type
3. **Retrieval Orchestration** - Retrieves and reranks evidence for each sub-query
4. **Evidence Bundling** - Organizes retrieved evidence by aspect for better generation

## 📁 Project Structure

```
typed_rag/
├── classifier/
│   ├── classifier.py          # Question type classifier
│   └── benchmark_classifiers.py  # Comparison: Gemini vs RoBERTa
├── decompose/
│   └── query_decompose.py     # Type-aware decomposition
├── retrieval/
│   ├── pipeline.py            # BGE embedder + vector stores
│   └── orchestrator.py        # Retrieval orchestration with reranking
├── data/                      # Document chunks (chunks.jsonl)
└── indexes/                   # FAISS/Pinecone indexes

tests/
└── test_decomposition.py      # 30 unit tests (5 per type)

examples/
└── typed_rag_pipeline.py      # Full pipeline example

cache/
├── decomposition/             # Cached decomposition plans
└── evidence/                  # Cached evidence bundles

output/
├── {qid}_typed_plan.json      # Decomposition plan
└── {qid}_evidence_bundle.json # Retrieved evidence
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export GOOGLE_API_KEY="your-gemini-api-key"
export PINECONE_API_KEY="your-pinecone-key"  # Optional
```

### 2. Classify a Question

```python
from typed_rag.classifier import classify_question

question = "Python vs Java for web development?"
qtype = classify_question(question, use_llm=True)
print(qtype)  # Output: "Comparison"
```

### 3. Decompose a Question

```python
from typed_rag.decompose import decompose_question

plan = decompose_question(question, qtype)
print(f"Generated {len(plan.sub_queries)} sub-queries:")
for sq in plan.sub_queries:
    print(f"  - {sq.aspect}: {sq.query}")
```

### 4. Retrieve Evidence

```python
from typed_rag.retrieval.orchestrator import RetrievalOrchestrator
from typed_rag.retrieval.pipeline import BGEEmbedder

# Initialize
embedder = BGEEmbedder()
orchestrator = RetrievalOrchestrator(
    embedder=embedder,
    vector_store=your_vector_store,
    rerank=True  # Use cross-encoder reranking
)

# Retrieve evidence
bundle = orchestrator.retrieve_evidence(plan)

# Access results
for evidence in bundle.evidence:
    print(f"{evidence.aspect}: {len(evidence.documents)} docs")
```

### 5. Run Complete Pipeline (Programmatic)

```python
from examples.typed_rag_pipeline import run_typed_rag_example

plan, bundle = run_typed_rag_example(
    question="How to deploy a machine learning model?",
    vector_store=your_store,
    output_dir="./output",
    rerank=True
)
```

### 6. Run Type-Aware CLI Pipeline

```bash
# Build FAISS index over your documents (one-time)
python rag_cli.py build --backend faiss --source own_docs

# Ask with full Typed-RAG flow (classification → decomposition → retrieval → generation)
python rag_cli.py typed-ask "Pros and cons of remote work"
python rag_cli.py typed-ask --backend pinecone --source wikipedia --rerank \
  "Why were steam engines pivotal to the Industrial Revolution?"
```

Outputs for each run are written to `./output/{question_id}_*` unless `--no-save` is passed.

### 7. Demo All Question Types

```bash
python examples/happy_flow_demo.py --backend faiss --source own_docs
```

This script sends one question from each NFQ type through the full pipeline and prints the aspect answers plus the aggregated response.

## 🧠 Answer Generation & Aggregation

- `typed_rag/generation/generator.py` produces aspect-level answers (`AspectAnswer`) from evidence bundles.
- `typed_rag/generation/aggregator.py` merges aspect answers into a final, type-aware response with structured formatting.
- Caching lives under `cache/answers/` and `cache/final_answers/`.
- Fallbacks ensure deterministic answers when LLM access is unavailable.

## 📏 Evaluation (LINKAGE)

- `typed_rag/eval/linkage.py` wraps the LINKAGE prompt and provides heuristic fallbacks.
- `typed_rag/eval/references.py` can synthesise ordered reference answers when human annotations are unavailable.
- `typed_rag/eval/runner.py` evaluates a JSONL file of `{question, candidate, references}` records:

```bash
python -m typed_rag.eval.runner --input outputs.jsonl --use-llm
```

Outputs include MRR, MPR, and per-question ranks. Omit `--use-llm` to rely on the embedding-based heuristic scorer.

## 📊 Question Types & Decomposition Strategies

### 1. Evidence-based
- **Strategy**: 1 passthrough sub-query
- **Example**: "What is quantum computing?"
- **Sub-queries**: Direct factual query

### 2. Comparison
- **Strategy**: Extract subjects + 3-5 comparison axes
- **Example**: "Python vs Java for web development"
- **Sub-queries**:
  - Performance comparison
  - Ecosystem comparison
  - Safety features comparison
  - Learning curve comparison

### 3. Experience
- **Strategy**: 2-4 topical angles for diverse perspectives
- **Example**: "Should I invest in stocks?"
- **Sub-queries**:
  - Reliability/quality aspects
  - Cost/value (TCO) analysis
  - Maintainability considerations
  - Support/community feedback

### 4. Reason
- **Strategy**: 2-4 causal sub-questions
- **Example**: "Why is the sky blue?"
- **Sub-queries**:
  - Direct causes
  - Mechanisms (how it works)
  - Historical/contextual factors
  - Constraints/limitations

### 5. Instruction
- **Strategy**: 3-5 steps or phases
- **Example**: "How to install Python on Mac?"
- **Sub-queries**:
  - Prerequisites/requirements
  - Setup/preparation
  - Core implementation steps
  - Validation/testing
  - Common pitfalls/troubleshooting

### 6. Debate
- **Strategy**: 2-3 opposing stances + neutral synthesis
- **Example**: "Pros and cons of remote work"
- **Sub-queries**:
  - Arguments in favor (pro stance)
  - Arguments against (con stance)
  - Balanced/neutral analysis

## 🧪 Testing

Run all tests (decomposition + generation fallbacks):

```bash
pytest tests
```

## 📦 Output Formats

### Typed Plan (typed_plan.json)

```json
{
  "question_id": "abc123def456",
  "original_question": "Python vs Java for web development?",
  "type": "Comparison",
  "sub_queries": [
    {
      "aspect": "performance",
      "query": "Performance comparison of Python vs Java for web development",
      "strategy": "compare"
    },
    {
      "aspect": "ecosystem",
      "query": "Ecosystem and libraries comparison for Python vs Java web frameworks",
      "strategy": "compare"
    }
  ]
}
```

### Evidence Bundle (evidence_bundle.json)

```json
{
  "question_id": "abc123def456",
  "original_question": "Python vs Java for web development?",
  "type": "Comparison",
  "evidence": [
    {
      "aspect": "performance",
      "sub_query": "Performance comparison of Python vs Java",
      "strategy": "compare",
      "documents": [
        {
          "id": "doc_001",
          "title": "Python vs Java Performance Benchmarks",
          "text": "...",
          "score": 0.89,
          "url": "https://..."
        }
      ]
    }
  ]
}
```

## 🔧 Configuration

### Caching

Decomposition plans and evidence are automatically cached by question hash:

```python
# Disable caching
plan = decompose_question(question, qtype, use_cache=False)
bundle = orchestrator.retrieve_evidence(plan, use_cache=False)

# Custom cache directory
from pathlib import Path
decomposer = QueryDecomposer(cache_dir=Path("./my_cache"))
```

### Retrieval Parameters

```python
bundle = orchestrator.retrieve_evidence(
    plan,
    initial_top_k=50,  # Initial retrieval before reranking
    final_top_k=5,     # Final number after reranking
    use_cache=True
)
```

### Reranking

Enable cross-encoder reranking for better relevance:

```python
orchestrator = RetrievalOrchestrator(
    embedder=embedder,
    vector_store=store,
    rerank=True  # Requires: sentence-transformers
)
```

## 📈 Performance

### Classifier Performance

| Method | Speed | Accuracy | Offline | Cost |
|--------|-------|----------|---------|------|
| Pattern Matching | <1ms | ~60% coverage | ✓ | Free |
| Gemini (fallback) | 200-500ms | ~90-95% | ✗ | ~$0.001/query |
| RoBERTa (alternative) | 10-50ms | 85-95% | ✓ | Free (after training) |

**Current hybrid approach**: 60% pattern match (instant), 40% LLM fallback

### Retrieval Performance

- **Dense retrieval**: ~50-100ms (BGE-small + FAISS)
- **Reranking**: +50-100ms (cross-encoder on 50 docs)
- **Total per sub-query**: ~100-200ms
- **Full question (5 sub-queries)**: ~500ms-1s

## 🎓 Research Background

Based on the paper: **"Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions"**

Key improvements over vanilla RAG:
- ✅ Better coverage of multi-aspect questions
- ✅ More diverse perspectives (especially for Debate/Experience)
- ✅ Structured evidence organization
- ✅ Higher relevance through targeted retrieval

## 🔜 Future Enhancements

1. **Learned Decomposition** - Train a model for optimal facet splitting
2. **Adaptive Strategies** - Dynamic sub-query count based on complexity
3. **Hybrid Retrieval** - Combine dense + sparse (BM25)
4. **Answer Aggregation** - Type-aware answer synthesis from sub-answers
5. **Human Feedback** - Collect feedback for decomposition quality

## 📝 Example Usage

See `examples/typed_rag_pipeline.py` for a complete example demonstrating:
- Question classification
- Type-aware decomposition
- Evidence retrieval with reranking
- JSON output generation

```bash
python examples/typed_rag_pipeline.py "Your question here"
```

## 🤝 Contributing

When adding new question types or decomposition strategies:

1. Update `QuestionType` in `classifier/classifier.py`
2. Add patterns in `QuestionClassifier._patterns`
3. Implement decomposition method in `QueryDecomposer`
4. Add 5 test cases in `tests/test_decomposition.py`
5. Update this README

## 📄 License

MIT License - see LICENSE file for details.
