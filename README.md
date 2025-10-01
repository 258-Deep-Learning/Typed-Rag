# Typed-RAG: Multi-Backend Retrieval-Augmented Generation

A flexible hybrid retrieval system with **pluggable vector database backends**. Supports local FAISS, Pinecone, Qdrant, and Weaviate. Works with Wikipedia data or your own custom documents.

**🆕 Version 2.0 - Multi-Backend Support Released!** See [WHATS_NEW.md](WHATS_NEW.md) for details.

## ✨ Key Features

- 🔄 **Multi-Backend Support**: Choose FAISS, Pinecone, Qdrant, or Weaviate at runtime
- 📚 **Custom Documents**: Use your own documents alongside or instead of Wikipedia
- 🎯 **Hybrid Search**: Combines BM25 (keyword) + Vector (semantic) search
- ⚡ **Fast**: 150-250ms query latency with local FAISS
- 🍎 **Apple Silicon Optimized**: Pure Python, no C++ compilation issues
- 🔌 **Backward Compatible**: All original scripts still work

## 🚀 Quick Start

### Option 1: Wikipedia + Local FAISS (Default)

```bash
# 1. Setup environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Build indexes (if not already built)
python scripts/build_bm25.py --input data/passages.jsonl
python scripts/build_faiss.py --input data/passages.jsonl

# 3. Query the system
python scripts/query_multi.py --q "Who discovered penicillin?" --k 5 --mode hybrid
```

### Option 2: Your Own Documents

```bash
# 1. Convert your documents
python scripts/convert_txt_to_jsonl.py --input-dir my_documents/ --output data/my_docs.jsonl

# 2. Build indexes
python scripts/build_bm25.py --input data/my_docs.jsonl --out indexes/custom/bm25_rank.pkl --meta-out indexes/custom/meta.jsonl
python scripts/build_faiss.py --input data/my_docs.jsonl --index-dir indexes/custom/faiss_bge_small

# 3. Query your documents
python scripts/query_multi.py --q "your question" --index-dir indexes/custom
```

### Option 3: Use Pinecone (Cloud Vector Database)

```bash
# 1. Set credentials
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"

# 2. Upload to Pinecone
python scripts/build_pinecone.py --input data/passages.jsonl --index-name typed-rag

# 3. Query with Pinecone
python scripts/query_multi.py --q "your question" --backend pinecone
```

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [File Structure](#file-structure)
- [Documentation](#documentation)
- [Development Notes](#development-notes)

## 🏗️ System Architecture

```
        ┌──────────────┐
        │passages.jsonl│   (your corpus)
        └──────┬───────┘
               │
     ┌─────────▼─────────┐             ┌─────────────────────────┐
     │ build_bm25.py      │            │ build_faiss.py          │
     │  tokenize & cache  │            │  embed & index vectors  │
     └──────┬─────────────┘            └──────────┬──────────────┘
            │                                      │
  bm25_rank.pkl + meta.jsonl              faiss/index.flatip + meta.jsonl
            │                                      │
            └───────────────┬──────────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │ HybridRetriever   │  ← hybrid = z(BM25) + z(Vector)
                  └─────────┬─────────┘
                            │
                       top-k docs
                            │
                  (next: feed to an LLM)
```

### Key Components:
- **BM25**: Classic keyword scoring for exact term matches
- **Vector Search**: Semantic similarity using BGE-small embeddings
- **Hybrid Scoring**: Z-score normalization + combination for robust retrieval
- **FAISS**: Fast approximate nearest neighbor search for vectors

## 🛠️ Installation

### Prerequisites
- Python 3.11+ 
- Java 17+ (for some optional dependencies)
- macOS with Apple Silicon support

### Step-by-Step Setup

1. **Install Python 3.11**
   ```bash
   brew install python@3.11
   ```

2. **Create Virtual Environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade Build Tools**
   ```bash
   python -m pip install --upgrade pip wheel setuptools
   ```

4. **Set Java Environment** (if needed)
   ```bash
   export JAVA_HOME=$(/usr/libexec/java_home -v 17)
   echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Prepare Your Data

Create `data/passages.jsonl` with your corpus. Each line should be a JSON object:

```json
{"id":"doc_001","title":"Document Title","url":"https://example.com","chunk_text":"Your passage text here..."}
```

### 2. Build Indexes

**BM25 Index** (keyword search):
```bash
python scripts/build_bm25.py --input data/passages.jsonl
# Creates: indexes/bm25_rank.pkl, indexes/meta.jsonl
```

**FAISS Index** (vector search):
```bash
python scripts/build_faiss.py --input data/passages.jsonl
# Creates: indexes/faiss_bge_small/index.flatip, meta.jsonl, model.txt
```

### 3. Query the System

```bash
# Hybrid search (recommended)
python scripts/query.py --q "Who discovered penicillin?" --k 3 --mode hybrid

# BM25 only
python scripts/query.py --q "penicillin discovery" --k 3 --mode bm25

# Vector search only
python scripts/query.py --q "antibiotic research" --k 3 --mode faiss
```

### Sample Output
```json
[
  {
    "id": "doc_001",
    "title": "Penicillin Discovery",
    "url": "https://en.wikipedia.org/wiki/Penicillin",
    "chunk_text": "Penicillin was discovered by Alexander Fleming in 1928...",
    "score": 0.85
  }
]
```

## 🔧 Components

### Core Libraries

| Library | Purpose | Why We Use It |
|---------|---------|---------------|
| `rank-bm25` | Pure-Python BM25 implementation | Avoids Pyserini's nmslib issues on Apple Silicon |
| `sentence-transformers` | Text embeddings | BGE-small-en-v1.5 for semantic similarity |
| `faiss-cpu` | Fast vector search | Efficient similarity search at scale |
| `joblib` | Object serialization | Caching tokenized texts for BM25 |
| `typer` | CLI framework | Clean command-line interfaces |

### Scripts

- **`build_bm25.py`**: Tokenizes passages and builds BM25 index
- **`build_faiss.py`**: Encodes passages to vectors and builds FAISS index  
- **`query.py`**: CLI interface for querying the hybrid retrieval system
- **`healthcheck.py`**: System health and dependency verification

### Retrieval Module

- **`retrieval/hybrid.py`**: Core `HybridRetriever` class with three modes:
  - `bm25`: Keyword-based retrieval only
  - `faiss`: Vector-based retrieval only  
  - `hybrid`: Combined scoring with z-score normalization

## 📁 File Structure

```
Typed-Rag/
├── data/
│   └── passages.jsonl          # Input corpus (JSONL format)
├── indexes/
│   ├── bm25_rank.pkl          # BM25 tokenized cache
│   ├── meta.jsonl             # Document metadata lookup
│   └── faiss_bge_small/       # FAISS vector index
│       ├── index.flatip       # Vector index file
│       ├── meta.jsonl         # Metadata aligned to vectors
│       └── model.txt          # Embedding model name
├── retrieval/
│   └── hybrid.py              # HybridRetriever implementation
├── scripts/
│   ├── build_bm25.py         # Build BM25 index
│   ├── build_faiss.py        # Build FAISS index
│   ├── query.py              # Query interface
│   └── healthcheck.py        # System verification
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🧪 Testing

### Quick Sanity Check

Create sample data and test the system:

```bash
# Add 3 sample documents
python - <<'PY'
import json, os
docs = [
  {"id":"1","title":"Penicillin","url":"https://en.wikipedia.org/wiki/Penicillin",
   "chunk_text":"Penicillin was discovered by Alexander Fleming in 1928 at St Mary's Hospital in London."},
  {"id":"2","title":"Albert Einstein","url":"https://en.wikipedia.org/wiki/Albert_Einstein", 
   "chunk_text":"Albert Einstein developed the theory of relativity and won the 1921 Nobel Prize in Physics."},
  {"id":"3","title":"Basketball","url":"https://en.wikipedia.org/wiki/Basketball",
   "chunk_text":"Basketball is a team sport where two teams try to score by shooting a ball through a hoop."}
]
os.makedirs("data", exist_ok=True)
with open("data/passages.jsonl","w") as f:
    for d in docs: f.write(json.dumps(d)+"\n")
print("wrote 3 docs")
PY

# Rebuild indexes and test
python scripts/build_bm25.py --input data/passages.jsonl
python scripts/build_faiss.py --input data/passages.jsonl
python scripts/query.py --q "Who discovered penicillin?" --k 3 --mode hybrid
```

### Health Check

```bash
python scripts/healthcheck.py
```

## 🎯 Next Steps (Weekend Goals)

1. **✅ Working retrieval system** with hybrid scoring
2. **🔄 Baseline comparisons**:
   - LLM-only (no retrieval)
   - Vanilla RAG (question + top passages)
3. **📊 Evaluation** on ~100 question dev set
4. **🚀 Production features**:
   - API endpoints
   - Batch processing
   - Performance monitoring

## 🐛 Troubleshooting

### Common Issues

**Query returns nothing**: 
- Ensure `retrieval/hybrid.py` exists and implements `HybridRetriever`
- Check that indexes were built successfully
- Verify your corpus has data in `data/passages.jsonl`

**Build failures on Apple Silicon**:
- We specifically avoid Pyserini/nmslib for this reason
- Use `faiss-cpu` instead of `faiss-gpu`
- Ensure you're using Python 3.11+

**Memory issues with large corpora**:
- Adjust `batch_size` in `build_faiss.py`
- Consider using `IndexIVFFlat` instead of `IndexFlatIP` for >100k docs

## 📚 Documentation

### Getting Started

- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - **Start here!** Complete guide to running the system
  - Wikipedia + FAISS setup
  - Custom documents workflow
  - Backend selection guide
  - Complete command reference
  - Troubleshooting

- **[WHATS_NEW.md](WHATS_NEW.md)** - Version 2.0 release notes
  - What's new in multi-backend support
  - Migration guide
  - Usage examples

- **[QUICKSTART_CUSTOM_DOCS.md](QUICKSTART_CUSTOM_DOCS.md)** - 5-minute quick start for custom documents

### Advanced Topics

- **[CUSTOM_DOCUMENTS_GUIDE.md](CUSTOM_DOCUMENTS_GUIDE.md)** - Complete guide for using your own documents
  - Format requirements
  - Conversion scripts (text, PDF, CSV)
  - Chunking best practices
  - Real-world use cases

- **[VECTOR_DATABASE_GUIDE.md](VECTOR_DATABASE_GUIDE.md)** - Vector database integration
  - When to use vector databases
  - Full Pinecone integration code
  - Qdrant, Weaviate, Milvus examples
  - Performance comparisons

- **[CUSTOMIZATION_SUMMARY.md](CUSTOMIZATION_SUMMARY.md)** - Quick reference for customization options

### Development History

- **[weekend1.md](weekend1.md)** - Weekend 1 implementation log
  - System implementation journey
  - Performance benchmarks
  - Evaluation results

- **[DEV_SET_DOCUMENTATION.md](DEV_SET_DOCUMENTATION.md)** - Development set creation
- **[WIKIPEDIA_MIGRATION_DOCS.md](WIKIPEDIA_MIGRATION_DOCS.md)** - Dataset migration notes

## 📝 Development Notes

- **No Conda**: Using pure Python virtual environments for simplicity
- **Apple Silicon Optimized**: All dependencies tested on M1/M2 Macs
- **Pure Python BM25**: Avoids C++ compilation issues
- **Normalized Embeddings**: Using cosine similarity via inner product
- **Z-score Combination**: Robust score fusion across different scales
- **Pluggable Backends**: Runtime selection of vector databases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Add your license here]

---

*Built with ❤️ for robust, scalable retrieval-augmented generation*