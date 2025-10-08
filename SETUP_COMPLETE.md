# ✅ Setup Complete! 

## What's Working:
- ✅ All Python dependencies installed
- ✅ PyPDF2 installed - PDFs can be read
- ✅ Your PDFs processed: **16 chunks** from 2 files
- ✅ Dev set generated: **16 questions**
- ✅ Project structure created

## 🔧 Issues to Fix:

### Issue 1: API Keys Not Loading Properly

Your `.env` file exists but variables aren't being exported. 

**Quick Fix - Export Directly:**
```bash
# Open a new terminal, then run:
source venv/bin/activate

# Set your API keys:
export PINECONE_API_KEY="your-pinecone-key-here"
export GOOGLE_API_KEY="your-google-key-here"

# Test it works:
python3 test_gemini.py
```

### Issue 2: Java Version for BM25

Pyserini (BM25) requires **Java 21**, but you have **Java 17**.

**Option A: Upgrade Java (Recommended for full hybrid search)**
```bash
# Install Java 21 with Homebrew:
brew install openjdk@21

# Add to your PATH:
export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"

# Verify:
java -version
# Should show: openjdk version "21.x.x"
```

**Option B: Skip BM25 for now (Use Pinecone only)**

The system works fine with just Pinecone (dense search). You can add BM25 later.

---

## 🚀 Quick Test (Pinecone + Gemini Only)

Since you have your chunks ready, let's test with Gemini:

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Set API keys
export PINECONE_API_KEY="pc-xxxxx"
export GOOGLE_API_KEY="AIzaSyxxxxx"

# 3. Test Gemini works
python3 test_gemini.py

# 4. Build Pinecone index
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own \
  --namespace own_docs

# 5. Test with one question
python3 typed_rag/scripts/run_llm_only.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/test.jsonl \
  --model gemini-2.0-flash-exp
```

---

## 📋 What You Have Now:

```
typed_rag/
├── data/
│   ├── chunks.jsonl          ✅ 16 chunks from your PDFs
│   └── dev_set.jsonl         ✅ 16 test questions
├── runs/
│   └── llm_only.jsonl        ⚠️  Dummy responses (no API key)
└── retrieval/
    └── pipeline.py           ✅ Ready for retrieval
```

---

## 🎯 Next Steps (Choose One):

### Path A: Full System with Hybrid Search
1. Upgrade to Java 21 (see above)
2. Set API keys properly
3. Run full pipeline with BM25 + Pinecone

### Path B: Test Now with Pinecone Only
1. Set API keys (export commands above)
2. Run Pinecone-only pipeline (no BM25 needed)
3. Add BM25 later when you upgrade Java

---

##  Simple Commands Reference:

```bash
# After setting up API keys, you can run:

# Build Pinecone index
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own

# Generate answers with Gemini (no retrieval)
python3 typed_rag/scripts/run_llm_only.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/results.jsonl

# Or check individual chunks:
cat typed_rag/data/chunks.jsonl | head -3
cat typed_rag/data/dev_set.jsonl | head -3
```

---

## 🐛 Troubleshooting:

**"API key not set"**
```bash
# Make sure to export, not just set:
export GOOGLE_API_KEY="your-key"  # ✅ Correct
GOOGLE_API_KEY="your-key"          # ❌ Wrong (won't work)

# Verify it's set:
echo $GOOGLE_API_KEY
```

**"Java version error"**
- You need Java 21 for BM25
- OR skip BM25 and just use Pinecone

**"Pinecone connection error"**
```bash
# Test Pinecone key:
python3 -c "import os; from pinecone import Pinecone; pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')); print(pc.list_indexes())"
```

---

## 💡 Your Documents:

Your resume PDFs were successfully chunked! You now have:
- `indraneel-sarode-resume-toyota-data.pdf` → 8 chunks
- `indraneel-sarode-resume-toyota-enterprise.pdf` → 8 chunks

Total: 16 searchable chunks with 200-token segments and 60-token overlap.

---

You're almost there! Just need to properly set the API keys and optionally upgrade Java for full features. 🎉

