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
  - Free tier: 60 requests/minute, unlimited tokens/day
- **Groq API Key** (Recommended): For open-source Llama models
  - Get it from: https://console.groq.com/keys
  - Free tier: 30 requests/minute, 12K tokens/minute, 100K tokens/day
  - 10-20x faster inference than local models
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

# Required: Groq API Key (for Llama models)
GROQ_API_KEY=your-groq-api-key-here

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
# Download WikiNFQA dataset (full 97-question development set)
python scripts/setup_wiki_nfqa.py --split dev100

# Expected output:
# ✓ Downloaded and prepared 97 questions
# ✓ Saved to: data/wiki_nfqa/dev100.jsonl

# For quick testing, use smaller set:
python scripts/setup_wiki_nfqa.py --split dev6  # 6 questions only
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
  --input data/wiki_nfqa/dev100.jsonl \
  --output results/ablation_dev100/ \
  --backend faiss \
  --source wikipedia \
  --model gemini-2.5-flash

# Expected runtime: ~60-90 minutes (97 questions × 4 variants = 388 questions)
# For quick testing: use dev6.jsonl (~5-10 minutes, 6 questions × 4 variants)
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

### Step 6: Run System Comparison (3 Systems × 2 Models)

Compare Typed-RAG against baselines with both Gemini and Llama:

**6A. Gemini Evaluation (Commercial Model)**

```bash
# LLM-Only with Gemini
python scripts/run_llm_only.py \
  --input data/wiki_nfqa/dev100.jsonl \
  --model gemini-2.5-flash \
  --output runs/llm_only_gemini_dev100.jsonl

# RAG Baseline with Gemini
python scripts/run_rag_baseline.py \
  --input data/wiki_nfqa/dev100.jsonl \
  --model gemini-2.5-flash \
  --output runs/rag_baseline_gemini_dev100.jsonl \
  --backend faiss \
  --source wikipedia

# Typed-RAG with Gemini
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev100.jsonl \
  --model gemini-2.5-flash \
  --output runs/typed_rag_gemini_dev100.jsonl \
  --backend faiss \
  --source wikipedia
```

**6B. Llama Evaluation via Groq (Open-Source Model)**

```bash
# Export Groq API key
export GROQ_API_KEY=your-groq-api-key-here

# LLM-Only with Llama
python scripts/run_llm_only.py \
  --input data/wiki_nfqa/dev100.jsonl \
  --model groq/llama-3.3-70b-versatile \
  --output runs/llm_only_llama_dev100.jsonl

# RAG Baseline with Llama
python scripts/run_rag_baseline.py \
  --input data/wiki_nfqa/dev100.jsonl \
  --model groq/llama-3.3-70b-versatile \
  --output runs/rag_baseline_llama_dev100.jsonl \
  --backend faiss \
  --source wikipedia

# Typed-RAG with Llama (with rate limiting to avoid 100K token/day limit)
python scripts/run_typed_rag.py \
  --input data/wiki_nfqa/dev100.jsonl \
  --model groq/llama-3.3-70b-versatile \
  --output runs/typed_rag_llama_dev100.jsonl \
  --backend faiss \
  --source wikipedia \
  --rate-limit-delay 30
```

**6C. Evaluate All 6 Systems**

```bash
# Evaluate all systems together
python scripts/evaluate_linkage.py \
  --systems \
    runs/llm_only_gemini_dev100.jsonl \
    runs/rag_baseline_gemini_dev100.jsonl \
    runs/typed_rag_gemini_dev100.jsonl \
    runs/llm_only_llama_dev100.jsonl \
    runs/rag_baseline_llama_dev100.jsonl \
    runs/typed_rag_llama_dev100.jsonl \
  --references data/wiki_nfqa/references.jsonl \
  --output results/full_linkage_evaluation.json
```

**Expected Runtime:**
- Gemini: 3 systems × 97 questions = 291 answers (~20-30 minutes total)
- Llama: 3 systems × 97 questions = 291 answers (~40-60 minutes total with rate limiting)
- LINKAGE Evaluation: 6 systems × 97 questions = 582 answers (~5-10 minutes)

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

### System Comparison (Actual Results on dev100 - 97 Questions)

**Gemini 2.5 Flash (Commercial Model)**

| System | MRR | MPR | Questions | Description |
|--------|-----|-----|-----------|-------------|
| LLM-Only | 0.3332 | 61.89% | 97 | No retrieval, relies on model knowledge |
| RAG Baseline | 0.1878 | 43.95% | 97 | Simple retrieval + generation |
| **Typed-RAG** | 0.2280 | 49.12% | 97 | Type-aware decomposition + retrieval |

**Llama 3.3 70B via Groq (Open-Source Model)**

| System | MRR | MPR | Questions | Avg Latency | Description |
|--------|-----|-----|-----------|-------------|-------------|
| **LLM-Only** | **0.3726** | **71.90%** | 97 | 0.32s | ⭐ Best overall performance |
| RAG Baseline | 0.2905 | 59.00% | 97 | 3.59s | Simple retrieval + generation |
| Typed-RAG | 0.2880 | 62.89% | 97 | 5.03s | Type-aware decomposition + retrieval |

**Key Findings:**
- ⚠️ **Surprising Result**: LLM-Only outperformed both RAG systems for both models
- Llama 3.3 70B consistently outperforms Gemini 2.5 Flash across all system types
- Typed-RAG shows mixed results:
  - Better than RAG Baseline for Gemini (0.2280 vs 0.1878 MRR)
  - Slightly lower MRR but higher MPR for Llama (0.2880 vs 0.2905 MRR)
- Best performance by question type (Typed-RAG Llama):
  - DEBATE: 0.4896 MRR, 79.17% MPR (best)
  - REASON: 0.3329 MRR, 72.50% MPR
  - COMPARISON: 0.3889 MRR, 80.00% MPR
  - EVIDENCE-BASED: 0.2522 MRR, 57.50% MPR (shows retrieval quality issues)

**Performance Metrics:**
- Llama via Groq: 10-20x faster than local models
- Token usage: ~2,500 tokens per question (Typed-RAG)
- Rate limiting: 30s delay required to stay under 100K token/day limit
- Success rate: 100% (97/97 questions, 0 errors) across all systems

**Note**: These are actual results from completed evaluations. Source files:
- `runs/llm_only_gemini_dev100.jsonl` (97 questions, 35KB)
- `runs/rag_baseline_gemini_dev100.jsonl` (97 questions, 343KB)
- `runs/typed_rag_gemini_dev100.jsonl` (97 questions, 193KB)
- `runs/llm_only_llama_dev100.jsonl` (97 questions)
- `runs/rag_baseline_llama_dev100.jsonl` (97 questions, 355KB)
- `runs/typed_rag_llama_dev100.jsonl` (97 questions, 226KB)
- `results/full_linkage_evaluation.json` (6 systems evaluated)

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
