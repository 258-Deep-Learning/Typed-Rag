# Using Your Own Documents with Typed-RAG

## 🚀 Quick Start: Replace Wikipedia with Your Documents

The Typed-RAG system is completely document-agnostic. Follow these steps to use your own documents.

---

## Step 1: Prepare Your Documents

### Format: JSONL (JSON Lines)

Create a file `data/my_documents.jsonl` with one JSON object per line:

```json
{"id":"doc001","title":"Introduction to Machine Learning","url":"https://example.com/ml-intro","chunk_text":"Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data..."}
{"id":"doc002","title":"Neural Networks Basics","url":"https://example.com/nn-basics","chunk_text":"Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes..."}
{"id":"doc003","title":"Deep Learning Applications","url":"https://example.com/dl-apps","chunk_text":"Deep learning has revolutionized computer vision, natural language processing, and speech recognition..."}
```

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier for the chunk | `"doc001"`, `"article_123_chunk_2"` |
| `title` | string | Document/section title | `"Introduction to ML"` |
| `url` | string | Source URL or reference | `"https://example.com"` or `"file:///path/to/doc.pdf"` |
| `chunk_text` | string | The actual content to search | Your document text (200-500 tokens recommended) |

---

## Step 2: Create JSONL from Your Data

### Option A: From Text Files

```python
# scripts/convert_txt_to_jsonl.py
import json
import os
from pathlib import Path

def convert_text_files_to_jsonl(input_dir: str, output_file: str):
    """Convert all .txt files in a directory to JSONL format."""
    
    documents = []
    
    for filepath in Path(input_dir).rglob("*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create chunks (simple approach - you can make this more sophisticated)
        chunks = chunk_text(content, chunk_size=500)
        
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{filepath.stem}_{i}",
                "title": filepath.stem.replace("_", " ").title(),
                "url": f"file://{filepath}",
                "chunk_text": chunk
            }
            documents.append(doc)
    
    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(documents)} chunks to {output_file}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(' '.join(chunk_words))
    
    return chunks


if __name__ == "__main__":
    # Example usage
    convert_text_files_to_jsonl(
        input_dir="my_documents/",
        output_file="data/my_documents.jsonl"
    )
```

### Option B: From PDFs

```python
# scripts/convert_pdf_to_jsonl.py
import json
from pathlib import Path
import PyPDF2  # pip install PyPDF2

def convert_pdfs_to_jsonl(input_dir: str, output_file: str):
    """Convert all PDF files to JSONL format."""
    
    documents = []
    
    for pdf_path in Path(input_dir).rglob("*.pdf"):
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            # Extract text from all pages
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text()
            
            # Create chunks
            chunks = chunk_text(full_text, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": f"{pdf_path.stem}_page_{i}",
                    "title": pdf_path.stem.replace("_", " ").title(),
                    "url": f"file://{pdf_path}",
                    "chunk_text": chunk
                }
                documents.append(doc)
    
    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(documents)} chunks from PDFs to {output_file}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Skip empty chunks
            chunks.append(' '.join(chunk_words))
    
    return chunks


if __name__ == "__main__":
    convert_pdfs_to_jsonl(
        input_dir="my_pdfs/",
        output_file="data/my_documents.jsonl"
    )
```

### Option C: From CSV/Excel

```python
# scripts/convert_csv_to_jsonl.py
import json
import pandas as pd

def convert_csv_to_jsonl(csv_file: str, output_file: str, 
                         id_col: str = "id",
                         title_col: str = "title", 
                         text_col: str = "content",
                         url_col: str = "url"):
    """Convert CSV/Excel to JSONL format."""
    
    # Read CSV or Excel
    if csv_file.endswith('.csv'):
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_excel(csv_file)
    
    # Convert to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            doc = {
                "id": str(row[id_col]) if id_col in df.columns else f"doc_{idx}",
                "title": str(row[title_col]) if title_col in df.columns else f"Document {idx}",
                "url": str(row[url_col]) if url_col in df.columns else f"https://example.com/{idx}",
                "chunk_text": str(row[text_col])
            }
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {len(df)} rows to {output_file}")


if __name__ == "__main__":
    # Example usage
    convert_csv_to_jsonl(
        csv_file="my_data.csv",
        output_file="data/my_documents.jsonl",
        id_col="document_id",
        title_col="doc_title",
        text_col="document_text",
        url_col="source_url"
    )
```

---

## Step 3: Build Indexes for Your Documents

Once you have your JSONL file, build the indexes:

```bash
# Navigate to project directory
cd /Users/indraneelsarode/Desktop/Typed-Rag

# Build BM25 index (keyword search)
python scripts/build_bm25.py --input data/my_documents.jsonl

# Build FAISS index (semantic search)
python scripts/build_faiss.py --input data/my_documents.jsonl

# This will create:
# - indexes/bm25_rank.pkl
# - indexes/meta.jsonl
# - indexes/faiss_bge_small/
```

**Expected output:**
```
✅ BM25 loaded: 1234 docs
Encoding: 100%|████████| 10/10 [00:15<00:00]
✅ FAISS loaded: 1234 vectors, model: BAAI/bge-small-en-v1.5
```

---

## Step 4: Query Your Documents

Now you can query your own documents!

```bash
# Hybrid search (recommended)
python scripts/query.py --q "your question about your documents" --k 5 --mode hybrid

# Keyword-only search
python scripts/query.py --q "specific keywords" --k 5 --mode bm25

# Semantic search
python scripts/query.py --q "conceptual question" --k 5 --mode faiss
```

**Example:**
```bash
python scripts/query.py --q "What are neural networks?" --k 3 --mode hybrid
```

**Output:**
```json
[
  {
    "id": "doc002",
    "title": "Neural Networks Basics",
    "url": "https://example.com/nn-basics",
    "chunk_text": "Neural networks are computing systems inspired by biological neural networks...",
    "score": 2.45,
    "bm25_score": 8.32,
    "faiss_score": 0.87
  },
  ...
]
```

---

## 📏 Best Practices for Chunking

### Recommended Chunk Sizes by Document Type

| Document Type | Chunk Size (tokens) | Overlap (tokens) | Rationale |
|---------------|---------------------|------------------|-----------|
| **Academic Papers** | 300-500 | 100 | Preserve paragraph context |
| **Code Documentation** | 200-300 | 50 | Keep function/class together |
| **News Articles** | 400-600 | 100 | Maintain story flow |
| **Technical Manuals** | 300-400 | 80 | Preserve step-by-step instructions |
| **Legal Documents** | 400-600 | 150 | Keep clauses together |
| **Chat/Email** | 200-300 | 50 | Short messages, less overlap needed |

### Advanced Chunking Strategy

```python
def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Intelligent chunking that respects sentence boundaries.
    """
    import re
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        if current_length + sentence_length <= chunk_size:
            current_chunk.extend(sentence_words)
            current_length += sentence_length
        else:
            # Save current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + sentence_words
            current_length = len(current_chunk)
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

---

## 🔍 Testing Your Setup

### Health Check

```bash
python scripts/query.py health
```

**Expected output:**
```
🏥 Retrieval System Health Check
========================================
✅ Bm25 Loaded: True
✅ Faiss Loaded: True
✅ Metadata Loaded: True
✅ Embedding Model Loaded: True
========================================
🎉 All systems operational!
```

### Quick Test Query

```bash
python scripts/query.py --q "test query about your content" --k 3 --mode hybrid --verbose
```

---

## 📊 Performance Considerations

### Index Building Time Estimates

| Document Count | BM25 Build Time | FAISS Build Time | Total Storage |
|----------------|-----------------|------------------|---------------|
| 1,000 | ~2 seconds | ~30 seconds | ~5 MB |
| 10,000 | ~15 seconds | ~5 minutes | ~50 MB |
| 50,000 | ~1 minute | ~20 minutes | ~200 MB |
| 100,000 | ~2 minutes | ~40 minutes | ~400 MB |

### Query Performance

- **BM25 only**: 50-100ms
- **FAISS only**: 100-150ms
- **Hybrid**: 150-250ms

---

## 🎯 Common Use Cases

### 1. Company Knowledge Base

```json
{"id":"kb001","title":"Employee Onboarding Process","url":"https://intranet.company.com/hr/onboarding","chunk_text":"New employees should complete the following steps within their first week..."}
{"id":"kb002","title":"IT Support - Password Reset","url":"https://intranet.company.com/it/password-reset","chunk_text":"To reset your password, navigate to the IT portal and click on 'Forgot Password'..."}
```

### 2. Research Paper Database

```json
{"id":"paper001","title":"Attention Is All You Need","url":"https://arxiv.org/abs/1706.03762","chunk_text":"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."}
{"id":"paper002","title":"BERT: Pre-training of Deep Bidirectional Transformers","url":"https://arxiv.org/abs/1810.04805","chunk_text":"We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations..."}
```

### 3. Product Documentation

```json
{"id":"api001","title":"Authentication API","url":"https://docs.company.com/api/auth","chunk_text":"To authenticate API requests, include your API key in the Authorization header..."}
{"id":"api002","title":"User Management Endpoints","url":"https://docs.company.com/api/users","chunk_text":"The Users API provides endpoints for creating, updating, and deleting user accounts..."}
```

---

## 🚨 Troubleshooting

### Issue: "No results returned"

**Check:**
1. Verify JSONL format is correct: `head -1 data/my_documents.jsonl | python -m json.tool`
2. Ensure indexes were built: `ls -lh indexes/`
3. Run health check: `python scripts/query.py health`

### Issue: "Out of memory during index building"

**Solution:**
Reduce batch size in FAISS building:

```bash
python scripts/build_faiss.py --input data/my_documents.jsonl --batch-size 64
```

### Issue: "Slow queries"

**Solutions:**
1. For >100K documents, use approximate search (see Vector Database section)
2. Reduce `k` (number of results)
3. Use single mode (bm25 or faiss) instead of hybrid

---

## 📚 Next Steps

After setting up your documents:

1. **Create a dev set** for your domain:
   ```bash
   python scripts/make_devset_quick.py --meta-path indexes/meta.jsonl --out-path data/my_dev_set.jsonl --n 100
   ```

2. **Run RAG baseline**:
   ```bash
   python scripts/run_rag_baseline.py --input-path data/my_dev_set.jsonl --out-path runs/my_rag_results.jsonl
   ```

3. **Evaluate performance** on your domain

---

## 💡 Tips

✅ **DO:**
- Use descriptive titles and URLs
- Keep chunks between 200-500 tokens
- Use overlap (50-150 tokens) for context
- Test with a small sample first

❌ **DON'T:**
- Mix languages in the same index (use separate indexes)
- Make chunks too small (<100 tokens) or too large (>1000 tokens)
- Forget to rebuild indexes after updating documents
- Include non-textual content without preprocessing

---

**You're all set!** Your custom documents are now searchable with the Typed-RAG system. 🎉

