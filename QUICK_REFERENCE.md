# 📖 Quick Reference Card - Typed-RAG v2.0

Copy-paste commands for common tasks. **Keep this handy!**

---

## 🚀 Query Commands

### Query Wikipedia (Default)

```bash
# Hybrid search (BM25 + Vector)
python scripts/query_multi.py --q "Who discovered penicillin?" --k 5

# Keyword search only
python scripts/query_multi.py --q "penicillin discovery" --mode bm25

# Semantic search only  
python scripts/query_multi.py --q "antibiotic research" --mode vector

# Verbose output
python scripts/query_multi.py --q "your question" --verbose
```

### Query Custom Documents

```bash
# Query your custom index
python scripts/query_multi.py --q "your question" --index-dir indexes/custom

# Query with specific backend
python scripts/query_multi.py --q "your question" --index-dir indexes/custom --backend faiss
```

### Query with Vector Databases

```bash
# Pinecone
python scripts/query_multi.py --q "your question" --backend pinecone

# Qdrant
python scripts/query_multi.py --q "your question" --backend qdrant --qdrant-url localhost

# Weaviate
python scripts/query_multi.py --q "your question" --backend weaviate --weaviate-url http://localhost:8080
```

---

## 📁 Document Conversion

### Convert Text Files

```bash
# Basic conversion
python scripts/convert_txt_to_jsonl.py -i my_docs/ -o data/docs.jsonl

# With custom chunking
python scripts/convert_txt_to_jsonl.py -i my_docs/ -o data/docs.jsonl --chunk-size 300 --overlap 50

# Smart sentence-aware chunking
python scripts/convert_txt_to_jsonl.py -i my_docs/ -o data/docs.jsonl --smart
```

### Convert PDFs

```bash
# Install PDF support first
pip install PyPDF2

# Convert PDFs
python scripts/convert_pdf_to_jsonl.py -i my_pdfs/ -o data/docs.jsonl

# With options
python scripts/convert_pdf_to_jsonl.py -i my_pdfs/ -o data/docs.jsonl --chunk-size 500 --min-words 50
```

---

## 🔨 Index Building

### Build Indexes (Default Location)

```bash
# BM25 index
python scripts/build_bm25.py --input data/passages.jsonl

# FAISS index
python scripts/build_faiss.py --input data/passages.jsonl

# Both at once
python scripts/build_bm25.py --input data/passages.jsonl && \
python scripts/build_faiss.py --input data/passages.jsonl
```

### Build Indexes (Custom Location)

```bash
# BM25 in custom directory
python scripts/build_bm25.py \
  --input data/custom.jsonl \
  --out indexes/custom/bm25_rank.pkl \
  --meta-out indexes/custom/meta.jsonl

# FAISS in custom directory
python scripts/build_faiss.py \
  --input data/custom.jsonl \
  --index-dir indexes/custom/faiss_bge_small
```

### Build Pinecone Index

```bash
# Set credentials
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"

# Upload to Pinecone
python scripts/build_pinecone.py \
  --input data/passages.jsonl \
  --index-name typed-rag
```

---

## 🤖 RAG Pipeline

### Run RAG Baseline

```bash
# Wikipedia + FAISS
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag_results.jsonl

# Custom documents + FAISS
python scripts/run_rag_multi.py \
  --input-path data/my_questions.jsonl \
  --out-path runs/custom_rag.jsonl \
  --index-dir indexes/custom

# With Pinecone
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag_pinecone.jsonl \
  --backend pinecone

# Different retrieval modes
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --mode bm25 \
  --out-path runs/rag_bm25_only.jsonl
```

---

## 🏥 Health & Diagnostics

### Health Check

```bash
# Check FAISS backend
python scripts/query_multi.py health

# Check Pinecone backend
python scripts/query_multi.py health --backend pinecone

# Check custom documents
python scripts/query_multi.py health --index-dir indexes/custom
```

### Run Demo

```bash
# Demo with FAISS
python scripts/query_multi.py demo

# Demo with Pinecone
python scripts/query_multi.py demo --backend pinecone
```

---

## ⚙️ Configuration

### Set Environment Variables

```bash
# OpenAI (for RAG pipeline)
export OPENAI_API_KEY="sk-your-key"

# Pinecone
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"
export PINECONE_INDEX_NAME="typed-rag"

# Qdrant
export QDRANT_URL="localhost"
export QDRANT_PORT="6333"

# Weaviate
export WEAVIATE_URL="http://localhost:8080"
```

### Load Configuration File

```bash
# Create config file
cp config.env.example config.env
# Edit config.env with your credentials

# Load configuration
source config.env
```

---

## 🔄 Complete Workflows

### Workflow 1: Wikipedia QA (Default)

```bash
# 1. Query (indexes already built)
python scripts/query_multi.py --q "Who invented the telephone?"

# 2. Run RAG pipeline
python scripts/run_rag_multi.py --input-path data/dev100.jsonl --out-path runs/rag.jsonl
```

### Workflow 2: Custom Documents (New Setup)

```bash
# 1. Convert documents
python scripts/convert_txt_to_jsonl.py -i my_docs/ -o data/custom.jsonl

# 2. Build indexes
mkdir -p indexes/custom
python scripts/build_bm25.py --input data/custom.jsonl --out indexes/custom/bm25_rank.pkl --meta-out indexes/custom/meta.jsonl
python scripts/build_faiss.py --input data/custom.jsonl --index-dir indexes/custom/faiss_bge_small

# 3. Query
python scripts/query_multi.py --q "your question" --index-dir indexes/custom

# 4. Run RAG
python scripts/run_rag_multi.py --input-path data/my_questions.jsonl --index-dir indexes/custom --out-path runs/custom_rag.jsonl
```

### Workflow 3: Switch to Pinecone

```bash
# 1. Set credentials
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"

# 2. Upload data (one-time)
python scripts/build_pinecone.py --input data/passages.jsonl --index-name typed-rag

# 3. Query
python scripts/query_multi.py --q "your question" --backend pinecone

# 4. Run RAG
python scripts/run_rag_multi.py --input-path data/dev100.jsonl --backend pinecone --out-path runs/rag_pinecone.jsonl
```

### Workflow 4: Multiple Data Sources

```bash
# Setup different indexes
mkdir -p indexes/wikipedia indexes/company indexes/research

# Build Wikipedia indexes
python scripts/build_bm25.py --input data/wiki.jsonl --out indexes/wikipedia/bm25_rank.pkl --meta-out indexes/wikipedia/meta.jsonl
python scripts/build_faiss.py --input data/wiki.jsonl --index-dir indexes/wikipedia/faiss_bge_small

# Build company indexes
python scripts/build_bm25.py --input data/company.jsonl --out indexes/company/bm25_rank.pkl --meta-out indexes/company/meta.jsonl
python scripts/build_faiss.py --input data/company.jsonl --index-dir indexes/company/faiss_bge_small

# Build research indexes  
python scripts/build_bm25.py --input data/research.jsonl --out indexes/research/bm25_rank.pkl --meta-out indexes/research/meta.jsonl
python scripts/build_faiss.py --input data/research.jsonl --index-dir indexes/research/faiss_bge_small

# Query different sources
python scripts/query_multi.py --q "historical question" --index-dir indexes/wikipedia
python scripts/query_multi.py --q "company question" --index-dir indexes/company
python scripts/query_multi.py --q "technical question" --index-dir indexes/research
```

---

## 🐛 Troubleshooting

### Check What's Installed

```bash
# List installed packages
pip list | grep -E "faiss|sentence|bm25|pinecone|qdrant|weaviate"

# Check Python version
python --version

# Check if in virtual environment
which python
```

### Check Indexes

```bash
# List index files
ls -lh indexes/
ls -lh indexes/custom/

# Count documents in metadata
wc -l indexes/meta.jsonl

# Validate JSONL format
head -1 data/passages.jsonl | python -m json.tool
```

### Test Components

```bash
# Test BM25 loading
python -c "import joblib; data = joblib.load('indexes/bm25_rank.pkl'); print(f'BM25: {len(data[\"ids\"])} docs')"

# Test FAISS loading
python -c "import faiss; index = faiss.read_index('indexes/faiss_bge_small/index.flatip'); print(f'FAISS: {index.ntotal} vectors')"

# Test Pinecone connection
python -c "import pinecone; pinecone.init(api_key='$PINECONE_API_KEY', environment='$PINECONE_ENVIRONMENT'); print(pinecone.list_indexes())"
```

### Common Fixes

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Rebuild indexes
python scripts/build_bm25.py --input data/passages.jsonl
python scripts/build_faiss.py --input data/passages.jsonl

# Clear cache
rm -rf __pycache__ retrieval/__pycache__ scripts/__pycache__

# Check disk space
df -h
```

---

## 📊 Performance Tuning

### Reduce Memory Usage

```bash
# Smaller batch size for FAISS
python scripts/build_faiss.py --input data/passages.jsonl --batch-size 32

# Use fewer results
python scripts/query_multi.py --q "your question" --k 3
```

### Speed Up Queries

```bash
# Use BM25 only (fastest)
python scripts/query_multi.py --q "specific keywords" --mode bm25

# Use vector only (semantic)
python scripts/query_multi.py --q "conceptual question" --mode vector

# Reduce k
python scripts/query_multi.py --q "your question" --k 3
```

### Scale to Large Datasets

```bash
# Use Pinecone for >100K documents
export PINECONE_API_KEY="your-key"
python scripts/build_pinecone.py --input data/large_dataset.jsonl
python scripts/query_multi.py --q "your question" --backend pinecone
```

---

## 📋 File Locations

### Data Files

- `data/passages.jsonl` - Wikipedia passages (original)
- `data/dev100.jsonl` - Development questions (100)
- `data/custom.jsonl` - Your custom documents (you create this)

### Index Files

- `indexes/bm25_rank.pkl` - BM25 index
- `indexes/meta.jsonl` - Document metadata
- `indexes/faiss_bge_small/` - FAISS vector index
- `indexes/custom/` - Custom document indexes (you create this)

### Results Files

- `runs/llm_only.jsonl` - LLM-only baseline results
- `runs/rag.jsonl` - RAG baseline results  
- `runs/rag_multi.jsonl` - Multi-backend RAG results

### Scripts

- `scripts/query_multi.py` - Multi-backend query interface ⭐ NEW
- `scripts/run_rag_multi.py` - Multi-backend RAG pipeline ⭐ NEW
- `scripts/convert_txt_to_jsonl.py` - Text converter ⭐ NEW
- `scripts/convert_pdf_to_jsonl.py` - PDF converter ⭐ NEW
- `scripts/query.py` - Original query script (still works)
- `scripts/run_rag_baseline.py` - Original RAG script (still works)

---

## 🔑 Key Command Flags

### Common Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--q` | Query text | `--q "your question"` |
| `--k` | Number of results | `--k 5` |
| `--mode` | Search mode | `--mode hybrid` |
| `--backend` | Vector backend | `--backend pinecone` |
| `--index-dir` | Index directory | `--index-dir indexes/custom` |
| `--verbose` | Detailed output | `--verbose` |

### Backend-Specific Flags

| Flag | Backend | Purpose |
|------|---------|---------|
| `--pinecone-api-key` | Pinecone | API key |
| `--pinecone-env` | Pinecone | Environment |
| `--pinecone-index` | Pinecone | Index name |
| `--qdrant-url` | Qdrant | Host URL |
| `--qdrant-port` | Qdrant | Port number |
| `--qdrant-collection` | Qdrant | Collection name |
| `--weaviate-url` | Weaviate | URL |
| `--weaviate-class` | Weaviate | Class name |

---

## 🎓 Decision Trees

### Which Backend Should I Use?

```
Document Count?
├─ <50K → Use FAISS (local, fast, free)
├─ 50K-100K → Use FAISS, monitor performance
└─ >100K 
   ├─ Need managed service? → Pinecone
   └─ Want to self-host? → Qdrant or Weaviate
```

### Which Search Mode Should I Use?

```
Query Type?
├─ Specific keywords/names → BM25 (--mode bm25)
├─ Conceptual/semantic → Vector (--mode vector)
└─ Unsure or general → Hybrid (--mode hybrid) ⭐ RECOMMENDED
```

### Where Should I Store Indexes?

```
Use Case?
├─ Wikipedia only → indexes/ (default)
├─ Custom docs only → indexes/custom/
└─ Both → Keep separate: indexes/ and indexes/custom/
```

---

## 💡 Pro Tips

### Tip 1: Use Environment Variables

```bash
# Set once, use everywhere
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"

# No need to specify flags
python scripts/query_multi.py --q "question" --backend pinecone
```

### Tip 2: Create Aliases

```bash
# Add to ~/.zshrc or ~/.bashrc
alias qry='python scripts/query_multi.py'
alias qwiki='python scripts/query_multi.py --index-dir indexes'
alias qcustom='python scripts/query_multi.py --index-dir indexes/custom'

# Use
qry --q "your question"
qwiki --q "wikipedia question"
qcustom --q "custom docs question"
```

### Tip 3: Batch Processing

```bash
# Query multiple questions
while IFS= read -r question; do
  python scripts/query_multi.py --q "$question" >> results.jsonl
done < questions.txt
```

### Tip 4: Monitor Performance

```bash
# Time queries
time python scripts/query_multi.py --q "your question"

# Profile memory
/usr/bin/time -l python scripts/query_multi.py --q "your question"
```

---

**Print this page and keep it at your desk! 📎**

*Last updated: October 1, 2025 - Version 2.0*

