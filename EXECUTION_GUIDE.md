# 🚀 Complete Execution Guide - Typed-RAG Multi-Backend System

This guide covers **everything** you need to know to run the Typed-RAG system with:
- **Wikipedia data** (existing setup)
- **Your own custom documents** (new capability)
- **Multiple vector backends** (FAISS, Pinecone, Qdrant, Weaviate)

---

## 📋 Table of Contents

1. [Quick Start (Wikipedia + FAISS)](#quick-start-wikipedia--faiss)
2. [Using Your Own Documents](#using-your-own-documents)
3. [Switching Vector Backends](#switching-vector-backends)
4. [Complete Command Reference](#complete-command-reference)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start (Wikipedia + FAISS)

**Use the existing Wikipedia setup with local FAISS** (fastest way to get started).

### Prerequisites

```bash
# 1. Activate virtual environment
cd /Users/indraneelsarode/Desktop/Typed-Rag
source .venv/bin/activate

# 2. Ensure indexes are built (if not already done)
python scripts/build_bm25.py --input data/passages.jsonl
python scripts/build_faiss.py --input data/passages.jsonl
```

### Query Wikipedia Data

```bash
# Using the NEW multi-backend query script (recommended)
python scripts/query_multi.py --q "Who discovered penicillin?" --k 5 --mode hybrid

# Or using the ORIGINAL query script (still works)
python scripts/query.py --q "Who discovered penicillin?" --k 5 --mode hybrid
```

**Both scripts work!** The new `query_multi.py` adds backend selection capability.

### Run RAG Pipeline (Wikipedia)

```bash
# NEW multi-backend RAG script
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag_wikipedia.jsonl \
  --backend faiss \
  --mode hybrid

# ORIGINAL RAG script (still works)
python scripts/run_rag_baseline.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag.jsonl
```

**Result:** Answers to 100 questions using Wikipedia + FAISS hybrid retrieval.

---

## 📁 Using Your Own Documents

Add your custom documents **alongside** Wikipedia (or replace it entirely).

### Option 1: Keep Wikipedia + Add Custom Docs (Separate Indexes)

```bash
# 1. Create custom documents directory
mkdir -p my_documents

# 2. Add your text files to my_documents/

# 3. Convert to JSONL
python scripts/convert_txt_to_jsonl.py \
  --input-dir my_documents/ \
  --output data/custom_docs.jsonl

# 4. Build SEPARATE indexes for custom docs
python scripts/build_bm25.py \
  --input data/custom_docs.jsonl \
  --out indexes/custom/bm25_rank.pkl \
  --meta-out indexes/custom/meta.jsonl

python scripts/build_faiss.py \
  --input data/custom_docs.jsonl \
  --index-dir indexes/custom/faiss_bge_small

# 5. Query custom documents (specify custom index directory)
python scripts/query_multi.py \
  --q "question about your custom docs" \
  --index-dir indexes/custom \
  --backend faiss

# 6. Query Wikipedia (uses default indexes directory)
python scripts/query_multi.py \
  --q "question about wikipedia" \
  --index-dir indexes \
  --backend faiss
```

**Now you have BOTH:**
- `indexes/` → Wikipedia data
- `indexes/custom/` → Your custom documents

**Switch between them using `--index-dir`!**

### Option 2: Replace Wikipedia Entirely

```bash
# 1. Backup Wikipedia indexes (optional)
mv indexes indexes_wikipedia_backup

# 2. Convert your documents
python scripts/convert_txt_to_jsonl.py \
  --input-dir my_documents/ \
  --output data/my_docs.jsonl

# 3. Build indexes (replaces Wikipedia)
python scripts/build_bm25.py --input data/my_docs.jsonl
python scripts/build_faiss.py --input data/my_docs.jsonl

# 4. Query (automatically uses new indexes)
python scripts/query_multi.py \
  --q "your question" \
  --backend faiss
```

**Result:** System now uses your documents instead of Wikipedia.

### PDF Documents

```bash
# Install PDF support
pip install PyPDF2

# Convert PDFs
python scripts/convert_pdf_to_jsonl.py \
  --input-dir my_pdfs/ \
  --output data/my_docs.jsonl

# Build indexes
python scripts/build_bm25.py --input data/my_docs.jsonl
python scripts/build_faiss.py --input data/my_docs.jsonl
```

---

## 🔄 Switching Vector Backends

The system now supports **4 vector backends**. You can choose at runtime!

### Backend Options

| Backend | Type | Best For | Setup Difficulty |
|---------|------|----------|------------------|
| **faiss** | Local | <100K docs, fast, free | ⭐ Easy (default) |
| **pinecone** | Cloud | Managed service, scalable | ⭐ Easy |
| **qdrant** | Self-host/Cloud | Modern, high-performance | ⭐⭐ Medium |
| **weaviate** | Self-host/Cloud | Feature-rich, open source | ⭐⭐ Medium |

### Using Local FAISS (Default)

```bash
# Queries use local FAISS automatically
python scripts/query_multi.py \
  --q "your question" \
  --backend faiss \
  --mode hybrid
```

**No additional setup needed!** This is the default and works out of the box.

### Using Pinecone

#### Setup Pinecone

```bash
# 1. Install Pinecone client
pip install pinecone-client

# 2. Get credentials from https://www.pinecone.io/
# Sign up → Create project → Get API key and environment

# 3. Set environment variables
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"  # Your environment

# 4. Build Pinecone index (one-time setup)
python scripts/build_pinecone.py \
  --input data/passages.jsonl \
  --index-name typed-rag
```

#### Query with Pinecone

```bash
# Query using Pinecone backend
python scripts/query_multi.py \
  --q "Who discovered penicillin?" \
  --backend pinecone \
  --pinecone-index typed-rag

# Or set env vars and omit flags
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="your-env"

python scripts/query_multi.py \
  --q "Who discovered penicillin?" \
  --backend pinecone
```

#### RAG with Pinecone

```bash
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag_pinecone.jsonl \
  --backend pinecone \
  --pinecone-index typed-rag
```

### Using Qdrant

#### Setup Qdrant (Docker)

```bash
# 1. Install Qdrant client
pip install qdrant-client

# 2. Run Qdrant with Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. Build Qdrant collection (you need to create this script or use API)
# For now, you can use the Pinecone script as a template
```

#### Query with Qdrant

```bash
python scripts/query_multi.py \
  --q "your question" \
  --backend qdrant \
  --qdrant-url localhost \
  --qdrant-port 6333 \
  --qdrant-collection typed-rag
```

### Using Weaviate

#### Setup Weaviate (Docker)

```bash
# 1. Install Weaviate client
pip install weaviate-client

# 2. Run Weaviate with Docker
docker run -d \
  -p 8080:8080 \
  --name weaviate \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest

# 3. Build Weaviate schema and upload data
# (You'll need to create a build script for Weaviate)
```

#### Query with Weaviate

```bash
python scripts/query_multi.py \
  --q "your question" \
  --backend weaviate \
  --weaviate-url http://localhost:8080 \
  --weaviate-class Document
```

---

## 📚 Complete Command Reference

### Query Commands

#### Multi-Backend Query (NEW)

```bash
# Basic usage
python scripts/query_multi.py --q "your question" --k 5 --mode hybrid

# With specific backend
python scripts/query_multi.py --q "your question" --backend pinecone

# With custom documents
python scripts/query_multi.py --q "your question" --index-dir indexes/custom

# Verbose output
python scripts/query_multi.py --q "your question" --verbose

# Different search modes
python scripts/query_multi.py --q "your question" --mode bm25     # Keyword only
python scripts/query_multi.py --q "your question" --mode vector   # Semantic only
python scripts/query_multi.py --q "your question" --mode hybrid   # Both (default)
```

**All Options:**
```bash
--q TEXT                Query string [required]
--k INTEGER             Number of results [default: 5]
--mode [bm25|vector|hybrid]  Search mode [default: hybrid]
--backend [faiss|pinecone|qdrant|weaviate]  Vector backend [default: faiss]
--index-dir TEXT        Index directory [default: indexes]
--verbose              Show detailed information
--pinecone-api-key     Pinecone API key
--pinecone-env         Pinecone environment
--pinecone-index       Pinecone index name [default: typed-rag]
--qdrant-url           Qdrant host [default: localhost]
--qdrant-port          Qdrant port [default: 6333]
--qdrant-collection    Qdrant collection [default: typed-rag]
--weaviate-url         Weaviate URL [default: http://localhost:8080]
--weaviate-class       Weaviate class [default: Document]
```

#### Original Query Script (Still Works)

```bash
# Basic usage
python scripts/query.py --q "your question" --k 5 --mode hybrid

# Different modes
python scripts/query.py --q "your question" --mode bm25
python scripts/query.py --q "your question" --mode faiss
python scripts/query.py --q "your question" --mode hybrid

# Health check
python scripts/query.py health

# Demo
python scripts/query.py demo
```

### RAG Pipeline Commands

#### Multi-Backend RAG (NEW)

```bash
# Basic usage (FAISS)
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag_results.jsonl

# With Pinecone
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag_pinecone.jsonl \
  --backend pinecone

# With custom documents
python scripts/run_rag_multi.py \
  --input-path data/my_questions.jsonl \
  --out-path runs/rag_custom.jsonl \
  --index-dir indexes/custom \
  --backend faiss

# Different retrieval modes
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --mode bm25 \
  --out-path runs/rag_bm25_only.jsonl
```

**All Options:**
```bash
--input-path TEXT       Input questions JSONL [default: data/dev100.jsonl]
--out-path TEXT         Output results JSONL [default: runs/rag_multi.jsonl]
--index-dir TEXT        Index directory [default: indexes]
--backend [faiss|pinecone|qdrant|weaviate]  Backend [default: faiss]
--mode [bm25|vector|hybrid]  Retrieval mode [default: hybrid]
--k INTEGER             Top-k passages [default: 5]
--model TEXT            LLM model [default: gpt-4o-mini]
--max-items INTEGER     Limit questions (0=all) [default: 0]
```

#### Original RAG Script (Still Works)

```bash
python scripts/run_rag_baseline.py \
  --input-path data/dev100.jsonl \
  --out-path runs/rag.jsonl \
  --mode hybrid \
  --k 5
```

### Document Conversion Commands

#### Text Files

```bash
# Basic
python scripts/convert_txt_to_jsonl.py \
  --input-dir my_documents/ \
  --output data/my_docs.jsonl

# With custom chunking
python scripts/convert_txt_to_jsonl.py \
  --input-dir my_documents/ \
  --output data/my_docs.jsonl \
  --chunk-size 300 \
  --overlap 50 \
  --smart  # Use smart sentence-aware chunking
```

#### PDF Files

```bash
# Basic
python scripts/convert_pdf_to_jsonl.py \
  --input-dir my_pdfs/ \
  --output data/my_docs.jsonl

# With options
python scripts/convert_pdf_to_jsonl.py \
  --input-dir my_pdfs/ \
  --output data/my_docs.jsonl \
  --chunk-size 500 \
  --min-words 50 \
  --recursive
```

### Index Building Commands

#### BM25 Index

```bash
# Default location (Wikipedia)
python scripts/build_bm25.py --input data/passages.jsonl

# Custom location
python scripts/build_bm25.py \
  --input data/my_docs.jsonl \
  --out indexes/custom/bm25_rank.pkl \
  --meta-out indexes/custom/meta.jsonl
```

#### FAISS Index

```bash
# Default location (Wikipedia)
python scripts/build_faiss.py --input data/passages.jsonl

# Custom location
python scripts/build_faiss.py \
  --input data/my_docs.jsonl \
  --index-dir indexes/custom/faiss_bge_small

# With custom batch size (for memory constraints)
python scripts/build_faiss.py \
  --input data/passages.jsonl \
  --batch-size 64
```

### Health Check Commands

```bash
# Check FAISS backend (default)
python scripts/query_multi.py health

# Check Pinecone backend
python scripts/query_multi.py health --backend pinecone

# Check custom documents
python scripts/query_multi.py health --index-dir indexes/custom

# Original health check
python scripts/query.py health
```

### Demo Commands

```bash
# Demo with FAISS
python scripts/query_multi.py demo

# Demo with Pinecone
python scripts/query_multi.py demo --backend pinecone

# Demo with custom documents
python scripts/query_multi.py demo --index-dir indexes/custom
```

---

## ⚙️ Configuration

### Environment Variables

Create `config.env` from the template:

```bash
cp config.env.example config.env
# Edit config.env with your credentials
```

Load configuration:

```bash
source config.env
```

### What to Configure

**Required for LLM:**
```bash
export OPENAI_API_KEY="sk-your-key"
```

**Required for Pinecone:**
```bash
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"
```

**Optional (for other backends):**
```bash
# Qdrant
export QDRANT_URL="localhost"
export QDRANT_PORT="6333"

# Weaviate
export WEAVIATE_URL="http://localhost:8080"
```

---

## 🎯 Common Use Cases

### Use Case 1: Wikipedia QA (Default)

```bash
# Query Wikipedia
python scripts/query_multi.py \
  --q "Who invented the telephone?" \
  --mode hybrid

# Full RAG pipeline
python scripts/run_rag_multi.py \
  --input-path data/dev100.jsonl \
  --out-path runs/wikipedia_rag.jsonl
```

### Use Case 2: Company Knowledge Base

```bash
# 1. Prepare documents
mkdir company_docs
# Add your .txt/.md files

# 2. Convert
python scripts/convert_txt_to_jsonl.py \
  --input-dir company_docs/ \
  --output data/company.jsonl

# 3. Build indexes
python scripts/build_bm25.py \
  --input data/company.jsonl \
  --out indexes/company/bm25_rank.pkl \
  --meta-out indexes/company/meta.jsonl

python scripts/build_faiss.py \
  --input data/company.jsonl \
  --index-dir indexes/company/faiss_bge_small

# 4. Query
python scripts/query_multi.py \
  --q "What is our vacation policy?" \
  --index-dir indexes/company
```

### Use Case 3: Research Papers

```bash
# 1. Convert PDFs
python scripts/convert_pdf_to_jsonl.py \
  --input-dir research_papers/ \
  --output data/papers.jsonl

# 2. Build indexes
python scripts/build_bm25.py --input data/papers.jsonl
python scripts/build_faiss.py --input data/papers.jsonl

# 3. Query
python scripts/query_multi.py \
  --q "What are transformer architectures?" \
  --k 10
```

### Use Case 4: Hybrid Wikipedia + Custom Docs

```bash
# Keep both index sets:
# - indexes/              (Wikipedia)
# - indexes/custom/       (Your docs)

# Query Wikipedia
python scripts/query_multi.py \
  --q "historical question" \
  --index-dir indexes

# Query custom docs
python scripts/query_multi.py \
  --q "company question" \
  --index-dir indexes/custom
```

### Use Case 5: Large Scale with Pinecone

```bash
# 1. Set up Pinecone
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="your-env"

# 2. Upload to Pinecone (one-time)
python scripts/build_pinecone.py \
  --input data/large_dataset.jsonl \
  --index-name my-large-index

# 3. Query (fast, scalable)
python scripts/query_multi.py \
  --q "your question" \
  --backend pinecone \
  --pinecone-index my-large-index
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"

```bash
# Solution: Activate virtual environment
source .venv/bin/activate

# Or reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Index not found"

```bash
# Check if indexes exist
ls -lh indexes/

# Rebuild if missing
python scripts/build_bm25.py --input data/passages.jsonl
python scripts/build_faiss.py --input data/passages.jsonl
```

### Issue: "Pinecone connection failed"

```bash
# Check credentials
echo $PINECONE_API_KEY
echo $PINECONE_ENVIRONMENT

# Test connection
python -c "import pinecone; pinecone.init(api_key='$PINECONE_API_KEY', environment='$PINECONE_ENVIRONMENT'); print(pinecone.list_indexes())"
```

### Issue: "Out of memory"

```bash
# Reduce batch size for FAISS
python scripts/build_faiss.py --input data/passages.jsonl --batch-size 32

# Or use vector database instead of local FAISS
python scripts/query_multi.py --q "your question" --backend pinecone
```

### Issue: "Slow queries"

**Solutions:**
1. Use BM25 only for keyword queries: `--mode bm25`
2. Use vector only for semantic queries: `--mode vector`
3. Reduce k: `--k 3`
4. Use vector database for large datasets: `--backend pinecone`

---

## 📊 Performance Expectations

### Query Latency

| Backend | Document Count | Query Time |
|---------|----------------|------------|
| FAISS | <50K | 50-150ms |
| FAISS | 50K-100K | 150-300ms |
| Pinecone | Any | 100-200ms |
| Qdrant | Any | 80-150ms |

### Index Build Time

| Document Count | BM25 | FAISS (local) |
|----------------|------|---------------|
| 1K | 2s | 30s |
| 10K | 15s | 5min |
| 50K | 1min | 20min |
| 100K | 2min | 40min |

---

## ✅ Quick Reference Cheat Sheet

```bash
# ============================================
# QUERY WIKIPEDIA (Default Setup)
# ============================================
python scripts/query_multi.py --q "your question"

# ============================================
# QUERY CUSTOM DOCUMENTS
# ============================================
# 1. Convert
python scripts/convert_txt_to_jsonl.py -i docs/ -o data/docs.jsonl

# 2. Index
python scripts/build_bm25.py --input data/docs.jsonl --out indexes/custom/bm25_rank.pkl --meta-out indexes/custom/meta.jsonl
python scripts/build_faiss.py --input data/docs.jsonl --index-dir indexes/custom/faiss_bge_small

# 3. Query
python scripts/query_multi.py --q "your question" --index-dir indexes/custom

# ============================================
# USE PINECONE INSTEAD OF FAISS
# ============================================
# Setup
export PINECONE_API_KEY="key"
export PINECONE_ENVIRONMENT="env"
python scripts/build_pinecone.py --input data/docs.jsonl

# Query
python scripts/query_multi.py --q "your question" --backend pinecone

# ============================================
# RUN FULL RAG PIPELINE
# ============================================
python scripts/run_rag_multi.py --input-path data/dev100.jsonl --out-path runs/results.jsonl

# ============================================
# HEALTH CHECK
# ============================================
python scripts/query_multi.py health
```

---

## 🎉 Summary

You now have a **fully flexible system** that supports:

✅ **Wikipedia data** (original setup, still works)  
✅ **Custom documents** (via conversion scripts)  
✅ **Multiple backends** (FAISS, Pinecone, Qdrant, Weaviate)  
✅ **Runtime selection** (choose backend when you query)  
✅ **Backward compatibility** (old scripts still work)  
✅ **Complete documentation** (this guide!)

**Choose your path:**
- Start simple: Use Wikipedia + FAISS
- Add your docs: Build custom indexes
- Scale up: Switch to Pinecone/Qdrant
- Mix and match: Keep both Wikipedia and custom docs

---

**Happy querying! 🚀**

*Last updated: October 1, 2025*

