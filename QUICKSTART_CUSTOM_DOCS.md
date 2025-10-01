# 🚀 Quick Start: Using Your Own Documents

This guide will get you up and running with your own documents in **5 minutes**.

---

## Option 1: Text Files (Easiest)

### Step 1: Put your text files in a folder

```bash
mkdir my_documents
# Add your .txt or .md files to this folder
```

### Step 2: Convert to JSONL

```bash
python scripts/convert_txt_to_jsonl.py \
  --input-dir my_documents/ \
  --output data/my_docs.jsonl
```

### Step 3: Build indexes

```bash
python scripts/build_bm25.py --input data/my_docs.jsonl
python scripts/build_faiss.py --input data/my_docs.jsonl
```

### Step 4: Query!

```bash
python scripts/query.py --q "your question" --k 5 --mode hybrid
```

**Done! 🎉**

---

## Option 2: PDF Files

### Step 1: Install PDF support

```bash
pip install PyPDF2
```

### Step 2: Convert PDFs

```bash
python scripts/convert_pdf_to_jsonl.py \
  --input-dir my_pdfs/ \
  --output data/my_docs.jsonl
```

### Step 3-4: Same as above

Build indexes and query as shown in Option 1.

---

## Option 3: Manual JSONL (Full Control)

Create `data/my_docs.jsonl` manually:

```json
{"id":"1","title":"My First Doc","url":"https://example.com","chunk_text":"This is my content..."}
{"id":"2","title":"My Second Doc","url":"https://example.com/2","chunk_text":"More content here..."}
```

Then build indexes and query.

---

## 🧪 Test with Sample Data

Want to try it first? Create a test file:

```bash
# Create sample documents
cat > data/test.jsonl << 'EOF'
{"id":"1","title":"Machine Learning Basics","url":"https://example.com/ml","chunk_text":"Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."}
{"id":"2","title":"Neural Networks","url":"https://example.com/nn","chunk_text":"Artificial neural networks are computing systems inspired by biological neural networks. They are comprised of node layers containing an input layer, one or more hidden layers, and an output layer. Each node connects to another and has an associated weight and threshold."}
{"id":"3","title":"Deep Learning","url":"https://example.com/dl","chunk_text":"Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to fields including computer vision and natural language processing."}
EOF

# Build indexes
python scripts/build_bm25.py --input data/test.jsonl
python scripts/build_faiss.py --input data/test.jsonl

# Test query
python scripts/query.py --q "What is deep learning?" --k 2 --mode hybrid
```

**Expected output:**
```json
[
  {
    "id": "3",
    "title": "Deep Learning",
    "url": "https://example.com/dl",
    "chunk_text": "Deep learning is part of a broader family...",
    "score": 2.45,
    "bm25_score": 12.3,
    "faiss_score": 0.89
  }
]
```

---

## 📝 Common Commands

### Convert different file types

```bash
# Text files (.txt, .md)
python scripts/convert_txt_to_jsonl.py -i my_docs/ -o data/docs.jsonl

# PDFs
python scripts/convert_pdf_to_jsonl.py -i my_pdfs/ -o data/docs.jsonl

# Both with custom chunk size
python scripts/convert_txt_to_jsonl.py \
  -i my_docs/ \
  -o data/docs.jsonl \
  --chunk-size 300 \
  --overlap 50
```

### Build indexes

```bash
# Build both indexes at once
python scripts/build_bm25.py --input data/docs.jsonl
python scripts/build_faiss.py --input data/docs.jsonl

# With custom batch size (if memory constrained)
python scripts/build_faiss.py --input data/docs.jsonl --batch-size 64
```

### Query different modes

```bash
# Hybrid (best for most cases)
python scripts/query.py --q "your question" --k 5 --mode hybrid

# Keyword search only (for exact terms)
python scripts/query.py --q "specific keyword" --k 5 --mode bm25

# Semantic search only (for concepts)
python scripts/query.py --q "conceptual question" --k 5 --mode faiss
```

### Check system health

```bash
python scripts/query.py health
```

---

## 🎯 Real-World Example

Let's say you have company documentation:

```bash
# 1. Organize your docs
my_company_docs/
├── engineering/
│   ├── api_docs.md
│   └── setup_guide.md
├── hr/
│   ├── onboarding.md
│   └── benefits.md
└── sales/
    └── playbook.md

# 2. Convert to JSONL
python scripts/convert_txt_to_jsonl.py \
  --input-dir my_company_docs/ \
  --output data/company_docs.jsonl \
  --recursive

# 3. Build indexes (takes ~2-5 minutes)
python scripts/build_bm25.py --input data/company_docs.jsonl
python scripts/build_faiss.py --input data/company_docs.jsonl

# 4. Ask questions!
python scripts/query.py --q "How do I set up the development environment?" --k 3

python scripts/query.py --q "What are the employee benefits?" --k 3

python scripts/query.py --q "What is our sales process?" --k 3
```

---

## 🔧 Troubleshooting

### "No results returned"

**Check:**
1. Do you have documents? `wc -l data/my_docs.jsonl`
2. Are indexes built? `ls -lh indexes/`
3. Run health check: `python scripts/query.py health`

### "Out of memory"

**Solution:**
```bash
# Reduce batch size
python scripts/build_faiss.py --input data/docs.jsonl --batch-size 32

# Or process fewer documents at once
head -1000 data/docs.jsonl > data/docs_small.jsonl
```

### "PyPDF2 not found"

**Solution:**
```bash
pip install PyPDF2
```

### "Slow queries"

**Check:**
- Are you querying the right index size? <100K docs is fine
- Try single mode: `--mode bm25` or `--mode faiss`
- Reduce k: `--k 3` instead of `--k 10`

---

## 📊 Performance Expectations

| Document Count | Index Build Time | Query Time | Memory Usage |
|----------------|------------------|------------|--------------|
| 100 | 30 seconds | 50ms | ~50MB |
| 1,000 | 2 minutes | 100ms | ~100MB |
| 10,000 | 15 minutes | 200ms | ~500MB |
| 50,000 | 1 hour | 300ms | ~2GB |

---

## 🎉 Next Steps

Once you have your documents indexed:

1. **Use with RAG**: See `scripts/run_rag_baseline.py` for full RAG pipeline
2. **Create dev set**: Generate evaluation questions with `scripts/make_devset_quick.py`
3. **Scale up**: See `VECTOR_DATABASE_GUIDE.md` for handling >100K documents
4. **Production**: Deploy as API with FastAPI (coming soon!)

---

## 💡 Pro Tips

✅ **DO:**
- Start small (100-1000 docs) to test
- Use descriptive titles in your documents
- Keep chunk sizes between 200-500 words
- Test queries before full indexing

❌ **DON'T:**
- Index binary files without conversion
- Mix multiple languages (use separate indexes)
- Forget to rebuild indexes when updating docs
- Use very large chunk sizes (>1000 words)

---

**That's it! You're ready to search your own documents.** 🚀

Questions? Check the main README.md or open an issue.

