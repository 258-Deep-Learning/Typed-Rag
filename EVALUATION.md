# Typed-RAG Evaluation Guide

This guide provides step-by-step instructions to reproduce all evaluation results for the Typed-RAG system.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running Evaluations](#running-evaluations)
- [Expected Results](#expected-results)
- [Generating Visualizations](#generating-visualizations)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.9+ (tested on Python 3.13.7)
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: ~5GB for models and data
- **OS**: macOS, Linux, or Windows with WSL

### API Keys Required
- **Google Gemini API Key**: For LLM-based classification and generation
  - Get it from: https://makersuite.google.com/app/apikey
  - Free tier: 60 requests/minute, 200 requests/day
- **Pinecone API Key** (Optional): For cloud-based vector store
  - Get it from: https://www.pinecone.io/
  - Free tier available

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/Typed-Rag.git
cd Typed-Rag
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import sentence_transformers; print('✓ Dependencies installed')"
```

### Step 4: Configure API Keys

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << 'EOF'
# Required: Google Gemini API Key
GOOGLE_API_KEY=your-gemini-api-key-here

# Optional: Pinecone API Key (for cloud vector store)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=gcp-starter
EOF
```

Replace `your-gemini-api-key-here` with your actual API key.

**Important**: Never commit `.env` file to version control!

### Step 5: Verify Setup

```bash
# Test API connection
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
print('✓ API key configured' if api_key else '✗ API key not found')
"
```

---

## Running Evaluations

### Step 1: Download and Prepare Data

The evaluation uses the Wikipedia Non-Factoid QA (WikiNFQA) dataset.

```bash
# Download WikiNFQA dataset (6-question development set)
python scripts/setup_wiki_nfqa.py --split dev6

# Expected output:
# ✓ Downloaded and prepared 6 questions
# ✓ Saved to: data/wiki_nfqa/dev6.jsonl
```

**Dataset Structure:**
```json
{
  "question_id": "abc123",
  "question": "Why was the campus police established in 1958?",
  "type": "Reason",
  "reference_answers": ["answer1", "answer2", "answer3"]
}
```

### Step 2: Build Vector Store Index

Build a FAISS index over your document collection (one-time operation):

```bash
# Option A: Use Wikipedia data (recommended for evaluation)
python rag_cli.py build --backend faiss --source wikipedia

# Option B: Use your own documents
python rag_cli.py build --backend faiss --source own_docs

# Expected output:
# ✓ Loaded documents
# ✓ Created embeddings
# ✓ Built FAISS index
# ✓ Saved to: data/indexes/faiss_wikipedia.index
```

**Index Build Time**: ~2-5 minutes depending on corpus size

### Step 3: Run Ablation Study

The ablation study tests 4 variants to evaluate component impact:

```bash
python scripts/run_ablation_study.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/ablation/ \
  --backend faiss \
  --source wikipedia \
  --model gemini-2.0-flash-lite

# Expected runtime: ~5-10 minutes (6 questions × 4 variants)
```

**Variants Tested:**
1. **Full Typed-RAG**: All components enabled (baseline)
2. **No Classification**: Classification disabled, forced to Evidence-based
3. **No Decomposition**: Single aspect query (no decomposition)
4. **No Retrieval**: Pure LLM generation without retrieval

**Output Files:**
```
results/ablation/
├── full.jsonl                 # Full system results
├── no_classification.jsonl    # No classification variant
├── no_decomposition.jsonl     # No decomposition variant
├── no_retrieval.jsonl         # No retrieval variant
└── summary.json               # Summary statistics
```

### Step 4: Evaluate Quality Metrics (MRR/MPR)

Evaluate answer quality using LINKAGE metrics:

```bash
python scripts/evaluate_linkage.py \
  --systems results/ablation/*.jsonl \
  --output results/ablation_linkage_evaluation.json

# Note: Use --use-llm flag for LLM-based scoring (requires API calls)
# Without flag, uses embedding-based heuristic scoring
```

**Metrics Computed:**
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant answer
- **MPR (Mean Percentile Rank)**: Average percentile rank across all answers

### Step 5: Run Classifier Evaluation

Evaluate question type classification accuracy:

```bash
python -m typed_rag.classifier.classifier \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/classifier_evaluation.json

# Expected output:
# Pattern-based: 66.67% accuracy
# Gemini: 100% accuracy
```

### Step 6: Run System Comparison

Compare Typed-RAG against baselines:

```bash
# LLM-Only (no retrieval)
python scripts/run_llm_only.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/llm_only.jsonl

# RAG Baseline (simple retrieval + generation)
python scripts/run_rag_baseline.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/rag_baseline.jsonl \
  --backend faiss \
  --source wikipedia

# Typed-RAG (full system)
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/typed_rag.jsonl \
  --backend faiss \
  --source wikipedia

# Evaluate all systems
python scripts/evaluate_linkage.py \
  --systems results/llm_only.jsonl results/rag_baseline.jsonl results/typed_rag.jsonl \
  --output results/linkage_evaluation.json
```

---

## Expected Results

### Ablation Study Results

| Variant | Success Rate | Avg Latency (s) | Description |
|---------|--------------|-----------------|-------------|
| **Full** | 100% (6/6) | 3.81 | All components enabled |
| **No Classification** | 100% (6/6) | 2.81 | 26% faster, forced Evidence-based |
| **No Decomposition** | 0% (0/6) | 0.00 | Single aspect query (bug: requires fix) |
| **No Retrieval** | 100% (6/6) | 5.62 | 48% slower, pure LLM |

### Classifier Performance

| Method | Accuracy | Speed | Offline | Cost |
|--------|----------|-------|---------|------|
| **Pattern-Based** | 66.67% | <1ms | ✓ | Free |
| **Gemini (fallback)** | 100% | 200-500ms | ✗ | $0.001/query |
| **Hybrid** | ~90% | <100ms avg | Hybrid | $0.0004/query |

### Per-Type Performance (Pattern-Based)

| Question Type | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Evidence-based | 0.33 | 1.00 | 0.50 | 1 |
| Comparison | 1.00 | 1.00 | 1.00 | 1 |
| Experience | 0.00 | 0.00 | 0.00 | 1 |
| Reason | 1.00 | 1.00 | 1.00 | 1 |
| Instruction | 1.00 | 1.00 | 1.00 | 1 |
| Debate | 0.00 | 0.00 | 0.00 | 1 |

### System Comparison (Expected)

| System | MRR | MPR | Latency (s) | Description |
|--------|-----|-----|-------------|-------------|
| **LLM-Only** | ~0.35 | ~50% | 2.5 | No retrieval, relies on model knowledge |
| **RAG Baseline** | ~0.42 | ~65% | 3.2 | Simple retrieval + generation |
| **Typed-RAG** | ~0.47 | ~71% | 3.8 | Type-aware decomposition + retrieval |

**Note**: MRR/MPR values are illustrative. Actual values depend on reference answers and evaluation method.

---

## Generating Visualizations

### Generate All Plots

```bash
# Generate publication-ready plots
python scripts/create_evaluation_plots.py

# Expected output:
# ✓ Generated: results/plots/ablation_latency.png
# ✓ Generated: results/plots/ablation_success.png
# ✓ Generated: results/plots/ablation_combined.png
# ✓ Generated: results/plots/mrr_mpr_comparison.png
# ✓ Generated: results/plots/classifier_performance.png
# ✓ Generated: results/plots/confusion_matrix.png
```

**Custom Options:**
```bash
# Save as SVG (vector graphics)
python scripts/create_evaluation_plots.py --format svg

# Custom output directory
python scripts/create_evaluation_plots.py --output my_plots/
```

### Run Performance Profiling

```bash
# Profile system performance (memory, throughput)
python scripts/profile_performance.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/performance_profile.json

# Expected output:
# ✓ Throughput: 0.26 questions/second
# ✓ Peak memory: 450.2 MB
# ✓ Avg memory per question: 75.0 MB
```

### Launch Interactive Demo

```bash
# Start Streamlit UI
streamlit run app.py

# Access at: http://localhost:8501
```

The demo includes three tabs:
1. **Query Interface**: Interactive question answering
2. **Evaluation Results**: Metrics and performance tables
3. **Ablation Study**: Component impact analysis

---

## Complete Evaluation Pipeline

Run all evaluations in one go:

```bash
#!/bin/bash
# complete_evaluation.sh

echo "🔬 Starting Complete Evaluation Pipeline"

# Step 1: Setup data
echo "📂 Step 1/7: Setting up data..."
python scripts/setup_wiki_nfqa.py --split dev6

# Step 2: Build index
echo "🏗️  Step 2/7: Building FAISS index..."
python rag_cli.py build --backend faiss --source wikipedia

# Step 3: Run ablation study
echo "🧪 Step 3/7: Running ablation study..."
python scripts/run_ablation_study.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/ablation/ \
  --backend faiss \
  --source wikipedia

# Step 4: Evaluate linkage
echo "📊 Step 4/7: Evaluating quality metrics..."
python scripts/evaluate_linkage.py \
  --systems results/ablation/*.jsonl \
  --output results/ablation_linkage_evaluation.json

# Step 5: Evaluate classifier
echo "🎯 Step 5/7: Evaluating classifier..."
python -m typed_rag.classifier.classifier \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/classifier_evaluation.json

# Step 6: Generate plots
echo "📈 Step 6/7: Generating plots..."
python scripts/create_evaluation_plots.py

# Step 7: Profile performance
echo "⚡ Step 7/7: Profiling performance..."
python scripts/profile_performance.py \
  --input data/wiki_nfqa/dev6.jsonl \
  --output results/performance_profile.json

echo "✅ Evaluation complete!"
echo "📁 Results: results/"
echo "📊 Plots: results/plots/"
echo "📄 Reports: markdown_results/"
```

Make it executable and run:

```bash
chmod +x complete_evaluation.sh
./complete_evaluation.sh
```

---

## Troubleshooting

### Issue: API Quota Exceeded

**Error**: `429: Resource has been exhausted (e.g. check quota)`

**Solution**:
1. Gemini free tier: 200 requests/day limit
2. Wait 24 hours for quota reset
3. Use alternative API key
4. Disable LLM-based scoring: Remove `--use-llm` flag from evaluation commands

### Issue: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution**:
```bash
pip install sentence-transformers transformers torch
```

### Issue: FAISS Index Not Found

**Error**: `FileNotFoundError: Index file not found`

**Solution**:
```bash
# Rebuild FAISS index
python rag_cli.py build --backend faiss --source wikipedia

# Verify index exists
ls -lh data/indexes/
```

### Issue: Out of Memory

**Error**: `CUDA out of memory` or `MemoryError`

**Solution**:
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Reduce batch size in config
# Edit typed_rag/retrieval/pipeline.py:
# batch_size = 16  # Reduce from 32
```

### Issue: Slow Performance

**Problem**: Evaluation taking too long

**Solution**:
1. **Enable Caching**: Results are cached by default in `cache/` directory
2. **Reduce Test Set**: Use `dev6.jsonl` (6 questions) instead of `dev100.jsonl`
3. **Skip Reranking**: Remove `--rerank` flag to speed up retrieval
4. **Use Local Models**: Consider using local models instead of API calls

### Issue: Empty Results

**Error**: All metrics show 0.0 values

**Solution**:
```bash
# Check if evaluation ran successfully
cat results/ablation/full.jsonl | jq '.answer' | head -n 1

# If empty, check API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"

# Re-run evaluation with verbose logging
python scripts/run_ablation_study.py --input data/wiki_nfqa/dev6.jsonl --output results/ablation/ --backend faiss --source wikipedia 2>&1 | tee evaluation.log
```

---

## File Structure After Evaluation

```
Typed-Rag/
├── data/
│   ├── wiki_nfqa/
│   │   ├── dev6.jsonl              # Test questions
│   │   └── dev100.jsonl            # Larger test set (optional)
│   └── indexes/
│       └── faiss_wikipedia.index   # FAISS vector index
├── results/
│   ├── ablation/
│   │   ├── full.jsonl
│   │   ├── no_classification.jsonl
│   │   ├── no_decomposition.jsonl
│   │   ├── no_retrieval.jsonl
│   │   └── summary.json
│   ├── ablation_linkage_evaluation.json
│   ├── classifier_evaluation.json
│   ├── linkage_evaluation.json
│   ├── performance_profile.json
│   └── plots/                      # Generated charts
│       ├── ablation_latency.png
│       ├── ablation_success.png
│       ├── mrr_mpr_comparison.png
│       ├── classifier_performance.png
│       └── confusion_matrix.png
├── cache/                          # Cached results (auto-generated)
│   ├── decomposition/
│   ├── evidence/
│   ├── answers/
│   └── final_answers/
└── markdown_results/
    ├── ablation_study_results.md
    ├── linkage_evaluation_results.md
    └── happy_flow_demo_results.md
```

---

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@article{typed-rag-2025,
  title={Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions},
  author={Your Name},
  year={2025}
}
```

---

## Additional Resources

- **Architecture Diagram**: See [docs/architecture.md](docs/architecture.md)
- **Screenshot Guide**: See [docs/screenshots.md](docs/screenshots.md)
- **Main README**: See [README_TYPED_RAG.md](README_TYPED_RAG.md)
- **Paper**: See [paper.txt](paper.txt)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing GitHub issues
3. Create a new issue with:
   - Error message
   - Python version
   - Steps to reproduce
   - Evaluation logs

---

**Last Updated**: January 2025  
**Tested On**: Python 3.13.7, macOS/Linux  
**Estimated Total Time**: ~30-45 minutes for complete evaluation
