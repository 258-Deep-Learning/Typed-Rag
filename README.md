# Typed-RAG: Weekend 1 Implementation

A production-ready RAG (Retrieval-Augmented Generation) system using **Pinecone** for vector search and **BM25** for lexical search, with hybrid fusion for optimal retrieval quality.

This is **Weekend 1** of the Typed-RAG project: building the foundational retrieval pipeline with your own documents.

## 🎯 What This Does

- **Ingest** documents (PDF, DOCX, MD, TXT, HTML) and chunk them into 200-token segments
- **Index** with both BM25 (Pyserini/Lucene) and dense vectors (Pinecone + BGE embeddings)
- **Retrieve** using hybrid search (z-score fusion)
- **Generate** dev set questions from your documents
- **Run baselines**: LLM-only and Vanilla RAG

## 📋 Prerequisites

- Python 3.11+
- Pinecone account (free tier works)
- Google API key for Gemini (recommended - fast & cost-effective) OR OpenAI API key

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Typed-Rag
source venv/bin/activate  # Activate your virtual environment

# Dependencies already installed:
# pip3 install pyserini pinecone-client transformers sentencepiece datasets faiss-cpu tyro typer structlog

# Optional (for better PDF/DOCX/HTML parsing):
pip3 install PyPDF2 python-docx beautifulsoup4
```

### 2. Set Up Environment Variables

```bash
# Add your API keys (choose Gemini OR OpenAI)
export PINECONE_API_KEY="your-pinecone-api-key"

# Option 1: Google Gemini (Recommended - Fast & Cost-effective)
export GOOGLE_API_KEY="your-google-api-key"

# Option 2: OpenAI (Alternative)
# export OPENAI_API_KEY="your-openai-api-key"
```

Get your Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Prepare Your Documents

Create a folder with your documents (PDF, DOCX, MD, TXT, HTML):

```bash
mkdir -p my_documents
# Add your documents to my_documents/
```

### 4. Run the Pipeline

#### Step 1: Ingest and Chunk Documents

```bash
python3 typed_rag/scripts/ingest_own_docs.py \
  --root my_documents \
  --out typed_rag/data/chunks.jsonl \
  --chunk_tokens 200 \
  --stride_tokens 60
```

This creates `chunks.jsonl` with 200-token chunks (60-token overlap).

#### Step 2: Build BM25 Index

```bash
python3 typed_rag/scripts/build_bm25.py \
  --in typed_rag/data/chunks.jsonl \
  --index typed_rag/indexes/lucene_own
```

#### Step 3: Build Pinecone Vector Index

```bash
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own \
  --namespace own_docs
```

This creates a Pinecone index with 384-dim BGE embeddings.

#### Step 4: Generate Dev Set (100 questions)

```bash
python3 typed_rag/scripts/make_own_dev.py \
  --root typed_rag/data/chunks.jsonl \
  --out typed_rag/data/dev_set.jsonl \
  --count 100
```

#### Step 5: Run Baselines

**LLM-Only Baseline** (no retrieval):

```bash
python3 typed_rag/scripts/run_llm_only.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/llm_only.jsonl \
  --model gemini-2.0-flash-exp
```

**Vanilla RAG Baseline** (with retrieval):

```bash
python3 typed_rag/scripts/run_rag_baseline.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/rag_baseline.jsonl \
  --pinecone_index typedrag-own \
  --pinecone_namespace own_docs \
  --bm25_index typed_rag/indexes/lucene_own \
  --k 5 \
  --model gemini-2.0-flash-exp
```

**Alternative: Use OpenAI models** by setting `OPENAI_API_KEY` and using `--model gpt-3.5-turbo` or `--model gpt-4`

## 📁 Project Structure

```
Typed-Rag/
├── typed_rag/
│   ├── data/                    # Raw and processed data
│   │   ├── chunks.jsonl         # Chunked documents
│   │   └── dev_set.jsonl        # Dev set questions
│   ├── retrieval/               # Retrieval components
│   │   ├── __init__.py
│   │   └── pipeline.py          # Core retrieval (Pinecone + BM25 + Hybrid)
│   ├── scripts/                 # Executable scripts
│   │   ├── ingest_own_docs.py   # Document ingestion & chunking
│   │   ├── build_bm25.py        # BM25 index builder
│   │   ├── build_pinecone.py    # Pinecone index builder
│   │   ├── make_own_dev.py      # Dev set generator
│   │   ├── run_llm_only.py      # LLM-only baseline
│   │   └── run_rag_baseline.py  # RAG baseline
│   ├── indexes/                 # Search indexes
│   │   └── lucene_own/          # BM25 index
│   ├── runs/                    # Experiment outputs
│   │   ├── llm_only.jsonl
│   │   └── rag_baseline.jsonl
│   └── config.env.example       # Environment config template
├── venv/                        # Virtual environment
└── README.md                    # This file
```

## 🔧 Configuration

### Retrieval Pipeline

The retrieval pipeline (`typed_rag/retrieval/pipeline.py`) supports:

- **Dense search** (Pinecone): BGE-small-en-v1.5 embeddings (384-dim, cosine similarity)
- **Lexical search** (BM25): Pyserini with standard BM25 parameters
- **Hybrid fusion**: Z-score normalization with stable tie-breaking

### Chunking Parameters

- `chunk_tokens`: 200 (default) - Size of each chunk
- `stride_tokens`: 60 (default) - Overlap between chunks

### Supported Document Types

- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF (requires PyPDF2)
- `.docx` - Word documents (requires python-docx)
- `.html`, `.htm` - HTML (requires beautifulsoup4)

## 📊 Output Format

### Chunks JSONL

```json
{
  "id": "doc123::chunk_0007",
  "doc_id": "doc123",
  "title": "My Document",
  "url": "file:///path/to/doc.pdf",
  "section": "",
  "chunk_idx": 7,
  "text": "...",
  "token_len": 198,
  "source": "internal"
}
```

### Dev Set JSONL

```json
{
  "question_id": "doc123::q_0007",
  "question": "What does the document say about...",
  "type": "concept",
  "related_doc_id": "doc123",
  "related_chunk_id": "doc123::chunk_0007"
}
```

### Baseline Results JSONL

```json
{
  "question_id": "doc123::q_0007",
  "question": "What does...",
  "prompt": "...",
  "passages": [...],
  "answer": "...",
  "latency_ms": 1234,
  "seed": 42,
  "model": "gpt-3.5-turbo"
}
```

## 🎯 Acceptance Criteria (Weekend 1)

- ✅ `retrieve()` returns stable top-20 under fixed seed
- ✅ Baselines complete end-to-end on 100-item dev set
- ✅ Median top-1 latency ≤ 2s per query (laptop)
- ✅ Every passage has title + URL/path metadata

## 🐛 Troubleshooting

### Pinecone Connection Issues

```bash
# Check your API key
echo $PINECONE_API_KEY

# List your Pinecone indexes
python3 -c "from pinecone import Pinecone; pc = Pinecone(api_key='YOUR_KEY'); print(pc.list_indexes())"
```

### BM25 Index Issues

Make sure Java is installed (required by Pyserini):

```bash
java -version
```

### Embedding Model Download

First run will download the BGE model (~133MB). Subsequent runs use cached model.

### LLM API Issues

If no LLM API is available (Gemini or OpenAI), scripts will log warnings and return dummy responses. The pipeline still works for testing retrieval.

**Supported Models:**
- **Gemini**: `gemini-2.0-flash-exp`, `gemini-1.5-flash`, `gemini-1.5-pro` (recommended)
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`

## 🔮 What's Next: Weekend 2 & 3

- **Weekend 2**: Typed decomposition + reranking
  - Query decomposition by question type
  - Cross-encoder reranking
  - Type-aware fusion

- **Weekend 3**: Evaluation + citations
  - Automated answer quality metrics
  - Citation accuracy
  - Comparison with baselines

## 📚 Key Components

### BGEEmbedder

Uses `BAAI/bge-small-en-v1.5` with L2 normalization:
- Query prefix: `"query: {text}"`
- Passage prefix: (none)
- Dimension: 384
- Batch size: 64 (default)

### PineconeDenseStore

Pinecone wrapper with:
- Serverless index creation
- Batch upserts (200 vectors/batch)
- Metadata filtering support
- Cosine similarity metric

### BM25Lexical

Pyserini/Lucene wrapper with:
- BM25 parameters: k1=0.9, b=0.4
- JSON document storage
- Metadata preservation

### Hybrid Fusion

Z-score normalization:
- Normalizes dense and lexical scores
- Combines with α (dense) and β (lexical) weights
- Stable tie-breaking by document ID

## 📝 Notes

- **Determinism**: All scripts use seed=42 by default for reproducible results
- **Metadata**: Titles and URLs are preserved throughout the pipeline for citations
- **Scalability**: Pinecone serverless scales automatically; BM25 is local
- **Deduplication**: Hybrid fusion automatically deduplicates by chunk ID

## 🤝 Contributing

This is the foundation for Weekends 2-3. Keep the `retrieve()` contract stable and log everything for downstream components.

## 📄 License

MIT License - Feel free to use for your own projects!

---

**Questions?** Check the logs—all scripts use `structlog` for detailed logging.

**Issues?** Make sure you've set up your API keys and installed optional dependencies for your document types.

