# 🚀 Quick Start with Gemini 2.5 Flash

Your Typed-RAG system is now configured to use **Gemini 2.5 Flash**!

## Step 1: Set Your API Key

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```

Get your Gemini key here: **https://aistudio.google.com/app/apikey**

## Step 2: Test Gemini Connection

```bash
python3 test_gemini.py
```

This will verify your API key works. You should see:
```
✓ GOOGLE_API_KEY is set
✓ google-generativeai package imported successfully
✓ API key configured
✓ Model initialized (gemini-2.0-flash-exp)
✓ Response received: '4'
✅ Gemini is working perfectly!
```

## Step 3: Prepare Your Documents

```bash
mkdir my_documents
# Copy your PDFs, DOCX, MD, TXT files into my_documents/
```

## Step 4: Run the Complete Pipeline

```bash
./quickstart.sh my_documents
```

This single command will:
1. ✅ Chunk your documents (200 tokens, 60 overlap)
2. ✅ Build BM25 index (Pyserini/Lucene)
3. ✅ Build Pinecone vector index (BGE embeddings)
4. ✅ Generate 100 test questions
5. ✅ Run LLM-only baseline with Gemini
6. ✅ Run RAG baseline with Gemini

Results will be in `typed_rag/runs/`

## Alternative: Step-by-Step

```bash
# 1. Chunk documents
python3 typed_rag/scripts/ingest_own_docs.py \
  --root my_documents \
  --out typed_rag/data/chunks.jsonl

# 2. Build BM25 index
python3 typed_rag/scripts/build_bm25.py \
  --in typed_rag/data/chunks.jsonl \
  --index typed_rag/indexes/lucene_own

# 3. Build Pinecone index
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own \
  --namespace own_docs

# 4. Generate questions
python3 typed_rag/scripts/make_own_dev.py \
  --root typed_rag/data/chunks.jsonl \
  --out typed_rag/data/dev_set.jsonl

# 5. Run LLM-only baseline
python3 typed_rag/scripts/run_llm_only.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/llm_only.jsonl \
  --model gemini-2.0-flash-exp

# 6. Run RAG baseline
python3 typed_rag/scripts/run_rag_baseline.py \
  --in typed_rag/data/dev_set.jsonl \
  --out typed_rag/runs/rag_baseline.jsonl \
  --model gemini-2.0-flash-exp
```

## Using Different Gemini Models

```bash
# Use Gemini 1.5 Flash (faster)
--model gemini-1.5-flash

# Use Gemini 1.5 Pro (better quality)
--model gemini-1.5-pro

# Use Gemini 2.0 Flash (default, recommended)
--model gemini-2.0-flash-exp
```

## Makefile Commands

```bash
make all                 # Run entire pipeline
make ingest              # Just chunk documents
make build-bm25          # Just build BM25 index
make build-pinecone      # Just build Pinecone index
make dev-set             # Just generate questions
make baseline-llm        # Just run LLM-only baseline
make baseline-rag        # Just run RAG baseline
```

## 💰 Gemini Pricing (as of 2024)

**Free Tier:**
- 15 requests/minute
- 1,500 requests/day
- 1M tokens/day

Perfect for development and testing!

## 🎯 What You Get

After running the pipeline, check `typed_rag/runs/`:

- `llm_only.jsonl` - Answers without retrieval
- `rag_baseline.jsonl` - Answers with hybrid retrieval (Pinecone + BM25)

Each result includes:
- Question
- Retrieved passages (with titles & URLs)
- Generated answer
- Latency stats

## 📚 Need Help?

- **Full docs**: `README.md`
- **Gemini setup**: `GEMINI_SETUP.md`
- **Test Gemini**: `python3 test_gemini.py`

## 🎉 That's It!

You now have a production-ready RAG system using:
- ⚡️ Gemini 2.5 Flash for generation
- 🔍 Pinecone for semantic search
- 📚 BM25 for lexical search
- 🤝 Hybrid fusion for best results

Enjoy!

