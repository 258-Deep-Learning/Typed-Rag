# 🌟 Complete Guide: Using Pinecone with Typed-RAG

**Last Updated:** October 2, 2025  
**Version:** 2.0  
**Status:** ✅ Tested and Working

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Common Errors & Solutions](#common-errors--solutions)
- [Complete Workflow](#complete-workflow)
- [Step-by-Step Guide](#step-by-step-guide)
- [Querying Pinecone](#querying-pinecone)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [FAQ](#faq)

---

## 🎯 Overview

### What is Pinecone?

**Pinecone** is a managed vector database service that allows you to:
- Store millions/billions of vector embeddings in the cloud
- Query them with sub-second latency
- Scale automatically without managing infrastructure
- Access your data from anywhere

### Why Use Pinecone vs Local FAISS?

| Feature | Local FAISS | Pinecone Cloud |
|---------|-------------|----------------|
| **Setup** | Simple, no signup | Requires account |
| **Storage** | Your disk | Cloud (unlimited) |
| **Scalability** | <100K docs | Millions of docs ✨ |
| **Access** | Single machine only | From anywhere 🌍 |
| **Speed** | Super fast (local) | Very fast (network) |
| **Cost** | Free | Free tier + paid |
| **Updates** | Rebuild indexes | Real-time updates |
| **Multi-user** | No | Yes ✨ |

### When to Use Pinecone?

✅ **Use Pinecone when:**
- You have >100K documents
- Need to access from multiple machines
- Want real-time index updates
- Building a production application
- Need to scale to millions of vectors

❌ **Stick with FAISS when:**
- You have <50K documents
- Working on a single machine
- Cost is a primary concern
- Batch updates are acceptable

---

## 🔧 Prerequisites

### 1. System Requirements
- Python 3.11+ installed
- Virtual environment activated
- Internet connection (for cloud access)

### 2. Pinecone Account Setup

**Step 1: Create Account**
1. Go to: https://www.pinecone.io/
2. Click "Sign Up" (free tier available)
3. Verify your email
4. Login to dashboard

**Step 2: Get API Key**
1. In Pinecone dashboard: https://app.pinecone.io/
2. Click on "API Keys" in left sidebar
3. Copy your API key (starts with `pcsk_...`)
4. Save it securely - you'll need this!

**Important Notes:**
- ⚠️ **DO NOT** commit API keys to git
- ⚠️ Store in environment variables, not in code
- ⚠️ Free tier has limits (check current quotas)

### 3. Required Python Packages

You need these installed in your virtual environment:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Install required packages
pip install pinecone         # NOT pinecone-client (old package)
pip install sentence-transformers
pip install PyPDF2           # For PDF conversion
pip install typer tqdm numpy
```

---

## ⚠️ Common Errors & Solutions

### Error 1: Wrong Pinecone Package

**Error Message:**
```
Exception: The official Pinecone python package has been renamed from 
`pinecone-client` to `pinecone`. Please remove `pinecone-client` from 
your project dependencies and add `pinecone` instead.
```

**Why This Happens:**
Pinecone renamed their package from `pinecone-client` to `pinecone` in 2024.

**Solution:**
```bash
# Remove old package
pip uninstall -y pinecone-client

# Install new package
pip install pinecone
```

**What Changed:**
- Old import: `import pinecone` from `pinecone-client` package
- New import: `from pinecone import Pinecone` from `pinecone` package

---

### Error 2: Old API Methods Not Working

**Error Message:**
```
AttributeError: module 'pinecone' has no attribute 'init'
```

**Why This Happens:**
The new Pinecone SDK (v3.0+) uses a different API structure.

**Old API (doesn't work):**
```python
import pinecone
pinecone.init(api_key="...", environment="...")
index = pinecone.Index("index-name")
```

**New API (correct):**
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("index-name")
```

**Solution:**
Update your code to use the new Pinecone object-oriented API.

---

### Error 3: Missing Environment Parameter

**Error Message:**
```
ValueError: Pinecone environment required. Set PINECONE_ENVIRONMENT or use --environment
```

**Why This Happens:**
Old Pinecone required both `api_key` and `environment`. The new version **doesn't need environment**!

**Solution:**
With new Pinecone SDK, you only need the API key:

```python
# Old way (don't use)
pinecone.init(api_key="...", environment="us-east1-gcp")

# New way (correct)
pc = Pinecone(api_key="...")
```

The environment is handled automatically based on your API key.

---

### Error 4: Virtual Environment Not Activated

**Error Message:**
```
source: no such file or directory: venv/bin/activate
```

**Why This Happens:**
- Virtual environment doesn't exist
- Using wrong path (`venv` vs `.venv`)

**Solution:**
```bash
# Check if .venv exists
ls -la | grep venv

# If it doesn't exist, create it
python3.11 -m venv .venv

# Activate (note the dot before venv)
source .venv/bin/activate

# Verify activation (should see (.venv) in prompt)
which python
```

---

### Error 5: PyPDF2 Not Installed

**Error Message:**
```
ModuleNotFoundError: No module named 'PyPDF2'
```

**Why This Happens:**
PDF conversion requires PyPDF2 package.

**Solution:**
```bash
source .venv/bin/activate
pip install PyPDF2
```

---

### Error 6: Index Not Ready Immediately

**Error Message:**
```
Index stats show 0 vectors immediately after upload
```

**Why This Happens:**
Pinecone takes a few seconds to process and index vectors after upload.

**Solution:**
Wait 5-10 seconds and check again:

```python
import time
time.sleep(5)  # Wait 5 seconds
stats = index.describe_index_stats()
print(stats)  # Should show vectors now
```

---

## 🚀 Complete Workflow

Here's the end-to-end process from PDF to querying in Pinecone:

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: YOUR PDF FILES                                  │
├─────────────────────────────────────────────────────────┤
│  dl.pdf, research.pdf, etc.                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: CONVERT TO JSONL                                │
├─────────────────────────────────────────────────────────┤
│  python scripts/convert_pdf_to_jsonl.py                  │
│  → data/my_docs.jsonl                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: CONVERT TO VECTORS & UPLOAD                     │
├─────────────────────────────────────────────────────────┤
│  python scripts/build_pinecone.py                        │
│  → Pinecone Cloud (29 vectors uploaded)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: QUERY FROM ANYWHERE                             │
├─────────────────────────────────────────────────────────┤
│  python scripts/query_pinecone.py --q "your question"    │
│  → Get answers from cloud!                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📖 Step-by-Step Guide

### Step 1: Initial Setup (One-Time)

#### 1.1 Activate Virtual Environment

```bash
# Navigate to project
cd /Users/indraneelsarode/Desktop/Typed-Rag

# Activate virtual environment
source .venv/bin/activate

# Verify activation (should see (.venv) prefix)
which python
# Output: /Users/indraneelsarode/Desktop/Typed-Rag/.venv/bin/python
```

#### 1.2 Install Required Packages

```bash
# Install Pinecone (NEW package name)
pip install pinecone

# Install PDF support
pip install PyPDF2

# Install other dependencies (if not already installed)
pip install sentence-transformers typer tqdm numpy
```

#### 1.3 Set Pinecone API Key

```bash
# Set as environment variable (lasts for this terminal session)
export PINECONE_API_KEY="your-api-key-here"

# Verify it's set
echo $PINECONE_API_KEY
```

**For Permanent Setup (Optional):**

Add to your shell config file (`~/.zshrc` or `~/.bashrc`):

```bash
# Add this line to ~/.zshrc
echo 'export PINECONE_API_KEY="your-api-key-here"' >> ~/.zshrc

# Reload shell config
source ~/.zshrc
```

---

### Step 2: Convert PDFs to JSONL

#### 2.1 Prepare Your PDFs

Place your PDF files in the project directory (or any folder):

```bash
# Example structure
/Users/indraneelsarode/Desktop/Typed-Rag/
├── dl.pdf
├── deep_learning_paper.pdf
└── research.pdf
```

#### 2.2 Convert PDFs to JSONL Format

```bash
# Convert all PDFs in current directory
python scripts/convert_pdf_to_jsonl.py \
  --input-dir . \
  --output data/my_docs.jsonl

# Or convert PDFs from specific folder
python scripts/convert_pdf_to_jsonl.py \
  --input-dir /path/to/pdfs/ \
  --output data/my_docs.jsonl
```

**What This Does:**
- Scans directory for all `.pdf` files
- Extracts text from each PDF
- Chunks text into manageable pieces (default: 500 words)
- Saves to JSONL format with metadata

**Expected Output:**
```
Found 2 PDF files
Processing: dl.pdf...
✅ Processed: dl.pdf (4 chunks)
Processing: 258 deep learning paper.pdf...
✅ Processed: 258 deep learning paper.pdf (25 chunks)

✅ Conversion complete!
   PDFs processed: 2
   Total chunks: 29
   Output: data/my_docs.jsonl
```

#### 2.3 Verify JSONL File

```bash
# Check the file was created
ls -lh data/my_docs.jsonl

# View first record
head -1 data/my_docs.jsonl | python -m json.tool
```

**Expected Format:**
```json
{
  "id": "dl_chunk_0",
  "title": "Dl",
  "url": "file:///path/to/dl.pdf#page=0",
  "chunk_text": "Deep Learning MCQ Rapid Notes..."
}
```

---

### Step 3: Upload to Pinecone Cloud

#### 3.1 Create Upload Script (Already Created)

The script `scripts/build_pinecone.py` is ready to use. It:
- Loads your JSONL documents
- Converts text to vectors (384-dimensional embeddings)
- Creates Pinecone index (if doesn't exist)
- Uploads vectors to cloud in batches

#### 3.2 Upload Your Documents

```bash
# Basic upload
python scripts/build_pinecone.py \
  --input data/my_docs.jsonl \
  --index-name dl-rag

# With custom settings
python scripts/build_pinecone.py \
  --input data/my_docs.jsonl \
  --index-name my-custom-index \
  --batch-size 50 \
  --cloud aws \
  --region us-east-1
```

**Parameters Explained:**
- `--input`: Path to your JSONL file
- `--index-name`: Name for your Pinecone index (lowercase, hyphens ok)
- `--batch-size`: Upload batch size (default: 100)
- `--cloud`: Cloud provider (aws, gcp, azure)
- `--region`: Cloud region (us-east-1, us-west-2, etc.)

**Expected Output:**
```
🔌 Connecting to Pinecone...
📦 Loading embedding model: BAAI/bge-small-en-v1.5
✨ Creating new index: dl-rag
⏳ Waiting for index to be ready...
📄 Loading documents from data/my_docs.jsonl...
✅ Loaded 29 documents
🚀 Encoding and uploading to Pinecone...
Uploading: 100%|██████████| 1/1 [00:02<00:00,  2.78s/it]

✅ Pinecone index built successfully!
📊 Total vectors: 29
🏷️  Index name: dl-rag

🎯 Next: Query with --backend pinecone
```

#### 3.3 Verify Upload

```bash
# Check index stats
python -c "
from pinecone import Pinecone
import os
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index('dl-rag')
stats = index.describe_index_stats()
print(f'Total vectors: {stats[\"total_vector_count\"]}')
print(f'Dimension: {stats[\"dimension\"]}')
"
```

**Expected Output:**
```
Total vectors: 29
Dimension: 384
```

---

### Step 4: Query Pinecone Index

#### 4.1 Basic Query

```bash
python scripts/query_pinecone.py \
  --q "What is deep learning?" \
  --k 3
```

**Parameters:**
- `--q`: Your question (required)
- `--k`: Number of results to return (default: 5)
- `--index-name`: Index to query (default: dl-rag)
- `--pretty`: Pretty print output (default: true)

**Expected Output:**
```
🔌 Connecting to Pinecone index: dl-rag...
📦 Loading embedding model...
🔍 Searching for: What is deep learning?

============================================================
[
  {
    "id": "dl_chunk_0",
    "score": 0.664905548,
    "metadata": {
      "text": "Deep Learning MCQ Rapid Notes...",
      "title": "Dl",
      "url": "file:///Users/.../dl.pdf#page=0"
    }
  },
  ...
]

✅ Found 3 results from Pinecone cloud
```

#### 4.2 Query Different Topics

```bash
# About activation functions
python scripts/query_pinecone.py \
  --q "What are activation functions?" \
  --k 3

# About optimization
python scripts/query_pinecone.py \
  --q "Explain Adam optimizer" \
  --k 5

# About neural networks
python scripts/query_pinecone.py \
  --q "How do CNNs work?" \
  --k 3

# Save results to file
python scripts/query_pinecone.py \
  --q "What is backpropagation?" \
  --k 5 > results.json
```

---

## 🔍 Querying Pinecone

### Query Modes

#### 1. Vector-Only Search (Current Implementation)

This is what we've implemented - pure semantic search using vector embeddings.

```python
# How it works:
1. Your question → Convert to vector (384 dimensions)
2. Search Pinecone for similar vectors
3. Return top-k most similar documents
```

**Pros:**
- ✅ Finds semantically similar content
- ✅ Works with synonyms and paraphrasing
- ✅ Language-agnostic (to some extent)

**Cons:**
- ❌ May miss exact keyword matches
- ❌ Less control over search behavior

#### 2. Hybrid Search (Can Be Implemented)

Combine keyword (BM25) + vector search for best results.

**Current Workaround:**
Since we have both local BM25 and Pinecone vectors:

```bash
# Use local hybrid search (BM25 + local FAISS)
python scripts/query.py search \
  --q "your question" \
  --bm25-path indexes/dl/bm25_rank.pkl \
  --faiss-dir indexes/dl/faiss_bge_small \
  --meta-path indexes/dl/meta.jsonl
```

### Advanced Query Options

#### Filter by Metadata (Pinecone Feature)

You can filter results by metadata fields:

```python
# Example (would need to modify query_pinecone.py)
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "title": {"$eq": "Dl"},  # Only from "Dl" document
        # or
        "url": {"$contains": "page=0"}  # Only from page 0
    },
    include_metadata=True
)
```

#### Namespace Queries (Multi-Tenant)

Organize vectors in namespaces for different users/categories:

```python
# Upload to namespace
index.upsert(vectors=vectors, namespace="user_123")

# Query from namespace
results = index.query(
    vector=query_embedding,
    namespace="user_123",
    top_k=5
)
```

---

## 🛠️ Troubleshooting

### Issue 1: "Index not found"

**Problem:**
```
Index 'dl-rag' not found
```

**Solutions:**
1. Check index exists in dashboard: https://app.pinecone.io/
2. List all indexes:
   ```bash
   python -c "from pinecone import Pinecone; import os; pc = Pinecone(api_key=os.environ['PINECONE_API_KEY']); print([idx.name for idx in pc.list_indexes()])"
   ```
3. Create index if missing:
   ```bash
   python scripts/build_pinecone.py --input data/my_docs.jsonl --index-name dl-rag
   ```

---

### Issue 2: "API key not set"

**Problem:**
```
ValueError: Pinecone API key required. Set PINECONE_API_KEY or use --api-key
```

**Solutions:**
1. Check if set:
   ```bash
   echo $PINECONE_API_KEY
   ```
2. Set it:
   ```bash
   export PINECONE_API_KEY="your-key-here"
   ```
3. Or pass directly:
   ```bash
   python scripts/query_pinecone.py --q "test" --api-key "your-key"
   ```

---

### Issue 3: "Vectors not appearing"

**Problem:**
Index shows 0 vectors after upload

**Solutions:**
1. Wait 5-10 seconds (Pinecone processes asynchronously)
2. Check stats again:
   ```bash
   python -c "from pinecone import Pinecone; import os, time; pc = Pinecone(api_key=os.environ['PINECONE_API_KEY']); idx = pc.Index('dl-rag'); time.sleep(5); print(idx.describe_index_stats())"
   ```
3. Re-upload if still 0:
   ```bash
   python scripts/build_pinecone.py --input data/my_docs.jsonl --index-name dl-rag
   ```

---

### Issue 4: "Slow queries"

**Problem:**
Queries taking >2 seconds

**Solutions:**
1. **Reduce k value:**
   ```bash
   python scripts/query_pinecone.py --q "test" --k 3  # Instead of --k 10
   ```

2. **Use metadata filtering:**
   Filter before similarity search to reduce search space

3. **Check network:**
   Pinecone queries require internet - check your connection

4. **Upgrade index type:**
   Use pod-based index instead of serverless for dedicated resources

---

### Issue 5: "Rate limiting"

**Problem:**
```
429 Too Many Requests
```

**Solution:**
Free tier has rate limits. Either:
1. Add delays between queries:
   ```python
   import time
   time.sleep(1)  # Wait 1 second between queries
   ```
2. Upgrade to paid tier for higher limits
3. Batch queries instead of individual requests

---

## 🚀 Advanced Usage

### 1. Multiple Indexes for Different Data

Keep separate indexes for different document types:

```bash
# Create indexes
python scripts/build_pinecone.py --input data/research_papers.jsonl --index-name research
python scripts/build_pinecone.py --input data/documentation.jsonl --index-name docs
python scripts/build_pinecone.py --input data/tutorials.jsonl --index-name tutorials

# Query different indexes
python scripts/query_pinecone.py --q "neural networks" --index-name research
python scripts/query_pinecone.py --q "API usage" --index-name docs
python scripts/query_pinecone.py --q "how to train" --index-name tutorials
```

### 2. Update Existing Index

Add new documents to existing index:

```bash
# Convert new PDFs
python scripts/convert_pdf_to_jsonl.py --input-dir new_pdfs/ --output data/new_docs.jsonl

# Upload to existing index (will add, not replace)
python scripts/build_pinecone.py --input data/new_docs.jsonl --index-name dl-rag
```

### 3. Delete Vectors

Remove specific documents from index:

```python
# Delete by ID
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index('dl-rag')

# Delete single vector
index.delete(ids=['dl_chunk_0'])

# Delete multiple vectors
index.delete(ids=['dl_chunk_0', 'dl_chunk_1'])

# Delete all vectors (clear index)
index.delete(delete_all=True)
```

### 4. Manage Indexes

```python
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# List all indexes
indexes = pc.list_indexes()
print([idx.name for idx in indexes])

# Delete index
pc.delete_index('old-index-name')

# Check index stats
index = pc.Index('dl-rag')
stats = index.describe_index_stats()
print(stats)
```

### 5. Batch Operations

For large-scale uploads:

```python
# Modify build_pinecone.py for larger batch sizes
python scripts/build_pinecone.py \
  --input data/large_dataset.jsonl \
  --batch-size 500 \
  --index-name large-index
```

---

## ❓ FAQ

### Q1: How much does Pinecone cost?

**Answer:**
- **Free Tier:** 1 index, 100K vectors, limited queries
- **Starter:** ~$70/month for more capacity
- **Enterprise:** Custom pricing for large scale

Check current pricing: https://www.pinecone.io/pricing/

---

### Q2: Can I use Pinecone offline?

**Answer:**
No, Pinecone is a cloud service and requires internet. For offline use, stick with local FAISS.

---

### Q3: How do I migrate from FAISS to Pinecone?

**Answer:**
You already have everything! Your JSONL files work with both:

```bash
# Already built FAISS
python scripts/build_faiss.py --input data/docs.jsonl

# Now add Pinecone
python scripts/build_pinecone.py --input data/docs.jsonl --index-name my-index

# Use whichever you prefer
python scripts/query.py --q "test"  # Uses FAISS
python scripts/query_pinecone.py --q "test"  # Uses Pinecone
```

---

### Q4: What's the maximum document size?

**Answer:**
- Pinecone has metadata limits (~40KB per vector)
- We limit text to first 1000 chars in metadata
- Full text is retrieved via the ID from your original JSONL

---

### Q5: Can multiple people query the same index?

**Answer:**
Yes! That's a key advantage. Anyone with the API key can query:

```bash
# Person 1 (your machine)
python scripts/query_pinecone.py --q "question 1"

# Person 2 (different machine, same API key)
python scripts/query_pinecone.py --q "question 2"

# Both query the same cloud index!
```

---

### Q6: How do I secure my API key?

**Answer:**
1. **Never commit to git:**
   ```bash
   # Add to .gitignore
   echo "*.env" >> .gitignore
   echo "config.env" >> .gitignore
   ```

2. **Use environment variables:**
   ```bash
   export PINECONE_API_KEY="your-key"
   ```

3. **Use .env files (for development):**
   ```bash
   # Create .env file
   echo 'PINECONE_API_KEY=your-key' > .env
   
   # Load in Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

---

### Q7: What embedding model is used?

**Answer:**
`BAAI/bge-small-en-v1.5` (384 dimensions)

**Why this model?**
- ✅ Great quality (top of leaderboard)
- ✅ Small size (fast)
- ✅ Good for English text
- ✅ Works with sentence-transformers

**Can I use a different model?**
Yes, modify `--model-name` parameter, but you'll need to recreate the index with the new dimension size.

---

### Q8: How long do vectors stay in Pinecone?

**Answer:**
Forever (until you delete them), as long as:
- Your account is active
- You're within quota limits (free tier)
- You don't manually delete the index

---

### Q9: Can I download vectors from Pinecone?

**Answer:**
Yes, but not recommended (use local FAISS instead). You can fetch vectors:

```python
# Fetch by ID
vector_data = index.fetch(ids=['dl_chunk_0'])

# But rebuilding from JSONL is easier:
python scripts/build_faiss.py --input data/my_docs.jsonl
```

---

### Q10: What's the difference between this and ChatGPT?

**Answer:**
- **ChatGPT:** General AI, knows general information
- **This system:** Searches YOUR specific documents
- **Advantage:** Can find information ChatGPT doesn't have (your PDFs, proprietary docs, recent papers)

---

## 📊 Quick Reference Commands

### Setup Commands
```bash
# Activate environment
source .venv/bin/activate

# Install Pinecone
pip install pinecone

# Set API key
export PINECONE_API_KEY="your-key"
```

### Conversion Commands
```bash
# PDF to JSONL
python scripts/convert_pdf_to_jsonl.py --input-dir . --output data/docs.jsonl

# Text to JSONL
python scripts/convert_txt_to_jsonl.py --input-dir . --output data/docs.jsonl
```

### Upload Commands
```bash
# Upload to Pinecone
python scripts/build_pinecone.py --input data/docs.jsonl --index-name my-index

# Check upload
python -c "from pinecone import Pinecone; import os; pc = Pinecone(api_key=os.environ['PINECONE_API_KEY']); print(pc.Index('my-index').describe_index_stats())"
```

### Query Commands
```bash
# Basic query
python scripts/query_pinecone.py --q "your question" --k 5

# Different index
python scripts/query_pinecone.py --q "your question" --index-name other-index

# Save results
python scripts/query_pinecone.py --q "your question" > results.json
```

---

## 🎉 Success Checklist

After following this guide, you should be able to:

- [x] Create Pinecone account and get API key
- [x] Install correct Pinecone package (`pinecone`, not `pinecone-client`)
- [x] Convert PDFs to JSONL format
- [x] Upload documents to Pinecone cloud
- [x] Query documents from anywhere
- [x] Troubleshoot common errors
- [x] Manage multiple indexes
- [x] Understand when to use Pinecone vs FAISS

---

## 📞 Support & Resources

### Official Resources
- **Pinecone Docs:** https://docs.pinecone.io/
- **Pinecone Dashboard:** https://app.pinecone.io/
- **Python Client:** https://github.com/pinecone-io/pinecone-python-client

### Project Resources
- **Main README:** `README.md`
- **Execution Guide:** `EXECUTION_GUIDE.md`
- **Vector DB Guide:** `VECTOR_DATABASE_GUIDE.md`
- **Custom Docs Guide:** `CUSTOM_DOCUMENTS_GUIDE.md`

### Getting Help
1. Check this guide first
2. Review error messages carefully
3. Check Pinecone dashboard for index status
4. Verify API key is set correctly
5. Ensure internet connection is working

---

## 🔄 Changelog

### Version 2.0 (October 2, 2025)
- ✅ Updated for new Pinecone SDK (v3.0+)
- ✅ Removed environment parameter (no longer needed)
- ✅ Added ServerlessSpec for index creation
- ✅ Complete error documentation
- ✅ Step-by-step workflow
- ✅ Comprehensive troubleshooting

### Known Issues
- None currently! All errors documented and resolved.

---

**🎯 You're all set!** You now have complete knowledge of using Pinecone with this RAG system.

**Next steps:**
1. Upload your documents to Pinecone
2. Query from anywhere
3. Build amazing AI applications!

**Happy querying! 🚀**

---

*Last verified: October 2, 2025*  
*Tested with: Python 3.11, Pinecone SDK 7.3.0*  
*Status: ✅ Production Ready*

