# Wikipedia Dataset Migration & Index Building Documentation

## 🚨 Problem: Dataset Scripts Deprecation

### What Happened?
On September 25, 2025, we encountered a critical error when running `scripts/make_passages_wikipedia.py`:

```
RuntimeError: Dataset scripts are no longer supported, but found wikipedia.py
```

### Root Cause Analysis
- **Hugging Face Datasets Library Change**: The Hugging Face `datasets` library deprecated script-based dataset loaders
- **Old Approach**: Our script was using `load_dataset("wikipedia", name, ...)` which relied on a Python script (`wikipedia.py`)  
- **Impact**: The old Wikipedia dataset loader became incompatible with newer versions of the `datasets` library
- **Timeline**: This change occurred as part of Hugging Face's move away from custom Python scripts for security and maintenance reasons

## ✅ Solution: Migration to Parquet Format

### The Fix
We migrated from the deprecated script-based Wikipedia dataset to the official parquet-based dataset:

**Before (Broken):**
```python
ds = load_dataset("wikipedia", name, split="train", streaming=streaming)
```

**After (Working):**
```python
ds = load_dataset("wikimedia/wikipedia", f"{snapshot}.{lang}", split="train", streaming=streaming)
```

### Why This Works
1. **Official Parquet Format**: `wikimedia/wikipedia` is the official parquet-based Wikipedia dump on Hugging Face
2. **Same Data Structure**: Maintains identical schema with fields: `id`, `url`, `title`, `text`
3. **Better Performance**: Parquet format is more efficient than script-based loading
4. **Future-Proof**: No dependency on deprecated Python scripts
5. **Multiple Languages**: Supports both `en` (English) and `simple` (Simple English) variants

### Available Configurations
- **20231101.en**: Full English Wikipedia (~6.41M articles)
- **20231101.simple**: Simple English Wikipedia (~242k articles)  
- **20220301.simple**: Older Simple English snapshot
- **Other snapshots**: Various historical snapshots available

## 🔧 Implementation Details

### File Modified
- **Location**: `scripts/make_passages_wikipedia.py`
- **Line Changed**: Line 34
- **Change Type**: Single line replacement
- **Impact**: Zero breaking changes to existing functionality

### Testing Results
```bash
# Test command used
python scripts/make_passages_wikipedia.py --lang simple --snapshot 20231101 --max-pages 100 --out data/test_passages.jsonl

# Results
✅ Successfully processed 100 pages
✅ Generated 531 text chunks
✅ All existing functionality preserved
```

## 📊 Index Building Process

After successfully generating the Wikipedia passages, we rebuilt the retrieval indexes:

### 1. BM25 Index Building

**Command Executed:**
```bash
python scripts/build_bm25.py --input data/passages.jsonl
```

**What This Script Does:**
- **Input Processing**: Reads `data/passages.jsonl` line by line
- **Tokenization**: Extracts tokens using regex `\b\w+\b` (word boundaries, Unicode-aware)
- **Data Extraction**: Parses each JSON line to extract:
  - `id`: Document identifier
  - `title`: Document title  
  - `url`: Source URL
  - `chunk_text`: Text content for indexing
- **Token Processing**: Converts text to lowercase and splits into tokens
- **Serialization**: Uses `joblib` to pickle the processed data for fast loading

**Outputs Generated:**
- `indexes/bm25_rank.pkl`: Serialized BM25 data containing:
  - `ids`: List of document IDs
  - `titles`: List of document titles
  - `urls`: List of source URLs
  - `texts`: List of original text chunks
  - `tokens`: List of tokenized text (for BM25 scoring)
- `indexes/meta.jsonl`: Metadata file with one JSON object per line

**Results:**
```
Saved: indexes/bm25_rank.pkl and indexes/meta.jsonl  (docs=33268)
```

### 2. FAISS Vector Index Building

**Command Executed:**
```bash
python scripts/build_faiss.py --input data/passages.jsonl
```

**What This Script Does:**
- **Text Extraction**: Reads passage text from JSONL file
- **Embedding Generation**: Uses `BAAI/bge-small-en-v1.5` model to create 384-dimensional vectors
- **Batch Processing**: Processes texts in batches of 128 for memory efficiency
- **Normalization**: Normalizes embeddings for cosine similarity via inner product
- **FAISS Indexing**: Creates `IndexFlatIP` (Inner Product) index for exact search
- **Persistence**: Saves index and metadata to disk

**Processing Details:**
- **Model**: `BAAI/bge-small-en-v1.5` (BGE = Beijing General Embeddings)
- **Embedding Dimension**: 384
- **Batch Size**: 128 passages per batch
- **Processing Time**: ~14 minutes for 33,268 passages (260 batches)
- **Normalization**: `normalize_embeddings=True` enables cosine similarity via inner product

**Outputs Generated:**
- `indexes/faiss_bge_small/index.flatip`: FAISS index file
- `indexes/faiss_bge_small/meta.jsonl`: Metadata aligned to vector positions
- `indexes/faiss_bge_small/model.txt`: Records the embedding model used

**Results:**
```
Encoding: 100%|█████████████████████████████████| 260/260 [14:01<00:00,  3.23s/it]
FAISS index built: 33268 vectors, dim=384
```

## 📁 Final Index Structure

After successful completion, the `indexes/` directory contains:

```
indexes/
├── bm25_rank.pkl           # BM25 tokenized data (33,268 docs)
├── meta.jsonl              # Shared metadata file  
└── faiss_bge_small/        # FAISS vector index directory
    ├── index.flatip        # Vector index (33,268 x 384 float32)
    ├── meta.jsonl          # Vector-aligned metadata
    └── model.txt           # "BAAI/bge-small-en-v1.5"
```

## 🎯 What Happens Next: Retrieval System

### Hybrid Retrieval Architecture

The built indexes enable a hybrid retrieval system combining:

1. **BM25 (Keyword-based)**:
   - Uses `indexes/bm25_rank.pkl` for exact term matching
   - Implements BM25 scoring algorithm
   - Excellent for queries with specific keywords

2. **FAISS (Semantic Vector Search)**:
   - Uses `indexes/faiss_bge_small/index.flatip` for semantic similarity
   - Leverages BGE embeddings for meaning-based matching
   - Great for conceptual queries

3. **Hybrid Scoring**:
   - Combines both approaches using z-score normalization
   - Provides robust retrieval across different query types

### Query Interface

Users can now query the system using:

```bash
# Hybrid search (recommended)
python scripts/query.py --q "Who discovered penicillin?" --k 5 --mode hybrid

# BM25 only (keyword-focused)
python scripts/query.py --q "penicillin discovery" --k 5 --mode bm25  

# Vector search only (semantic)
python scripts/query.py --q "antibiotic research" --k 5 --mode faiss
```

## 📈 Performance Metrics

### Dataset Statistics
- **Total Documents**: 33,268 passages
- **Source**: Simple English Wikipedia (20231101 snapshot)
- **Average Processing**: ~3.23 seconds per batch (128 passages)
- **Storage Efficiency**: Parquet format reduces download time and storage

### Index Characteristics
- **BM25 Index**: Tokenized text cache for fast keyword scoring
- **Vector Index**: 33,268 × 384 normalized float32 vectors
- **Index Type**: `IndexFlatIP` for exact nearest neighbor search
- **Search Complexity**: O(n) for exact search, suitable for <100K documents

## 🔄 Maintenance & Updates

### Future Wikipedia Updates
To update to newer Wikipedia snapshots:

```bash
# Update to newer snapshot (when available)
python scripts/make_passages_wikipedia.py --lang simple --snapshot 20240301 --max-pages 8000 --out data/passages.jsonl

# Rebuild indexes
python scripts/build_bm25.py --input data/passages.jsonl
python scripts/build_faiss.py --input data/passages.jsonl
```

### Scaling Considerations
- **For >100K documents**: Consider `IndexIVFFlat` instead of `IndexFlatIP`
- **Memory optimization**: Adjust `batch_size` in `build_faiss.py`
- **Storage**: Current setup uses ~150MB for 33K documents

## 🚀 Benefits of the Migration

1. **Reliability**: No more dependency on deprecated dataset scripts
2. **Performance**: Parquet format loads faster than script-based approach  
3. **Maintainability**: Official dataset reduces maintenance burden
4. **Compatibility**: Works with latest `datasets` library versions
5. **Scalability**: Parquet format handles large datasets efficiently

## 📝 Lessons Learned

1. **Stay Updated**: Monitor dependency deprecations proactively
2. **Official Sources**: Prefer official datasets over community scripts
3. **Testing**: Always test with small samples before full processing
4. **Documentation**: Document breaking changes and solutions
5. **Future-Proofing**: Choose stable, well-maintained alternatives

---

**Migration Date**: September 25, 2025  
**Status**: ✅ Complete and Tested  
**Impact**: Zero breaking changes to existing functionality  
**Next Steps**: System ready for production RAG queries
