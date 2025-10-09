# Quick Start: Index Your Documents and Query

This guide shows you how to index your own documents in Pinecone and query them with an LLM.

## Prerequisites

Make sure you have your API keys set:

```bash
export PINECONE_API_KEY='your-pinecone-key'
export GOOGLE_API_KEY='your-google-key'
```

## Step-by-Step Workflow

### Option 1: Use the Automated Script (Recommended)

Run the complete pipeline with one command:

```bash
./setup_and_query.sh
```

This will:
1. Ingest all documents from `my-documents/` folder
2. Convert them to chunks (saved to `typed_rag/data/chunks.jsonl`)
3. Build Pinecone vector index with embeddings
4. Ready to query!

### Option 2: Run Steps Manually

#### Step 1: Ingest Your Documents

Put your documents (PDF, DOCX, TXT, MD, HTML) in the `my-documents/` folder, then run:

```bash
python3 typed_rag/scripts/ingest_own_docs.py \
  --root my-documents \
  --out typed_rag/data/chunks.jsonl \
  --chunk_tokens 200 \
  --stride_tokens 60
```

This converts your documents into overlapping chunks.

#### Step 2: Build Pinecone Index

```bash
python3 typed_rag/scripts/build_pinecone.py \
  --in typed_rag/data/chunks.jsonl \
  --index typedrag-own \
  --namespace own_docs
```

This:
- Generates embeddings using BGE model
- Uploads vectors to Pinecone
- Creates searchable index

## Query Your Documents

Now you can ask questions about your documents:

```bash
python3 ask.py "What does the document say about Amazon?"
```

Or:

```bash
python3 ask.py "What are the key points in the document?"
```

The script will:
1. 🔍 Retrieve relevant passages from Pinecone (top 5 by default)
2. 💡 Feed them to Gemini LLM
3. 🎯 Display the answer with citations

## Supported File Types

- **PDF** (.pdf) - requires PyPDF2
- **Word** (.docx) - requires python-docx  
- **Markdown** (.md)
- **Text** (.txt)
- **HTML** (.html, .htm) - requires beautifulsoup4

## Configuration

You can customize parameters by editing the variables in `setup_and_query.sh`:

- `DOCS_DIR` - Where your documents are located
- `PINECONE_INDEX` - Pinecone index name
- `PINECONE_NAMESPACE` - Namespace for your documents
- `--chunk_tokens` - Size of each chunk (default: 200)
- `--stride_tokens` - Overlap between chunks (default: 60)

## Troubleshooting

### Missing Dependencies

If you get import errors, install the required packages:

```bash
pip install PyPDF2 python-docx beautifulsoup4
```

### Re-indexing

To update your index with new documents:

1. Add documents to `my-documents/`
2. Run `./setup_and_query.sh` again
3. It will re-process all documents and update the index

### Query Different Index

To query a different Pinecone index, edit `ask.py` lines 29-30:

```python
pinecone_index = "your-index-name"
pinecone_namespace = "your-namespace"
```

## Example Workflow

```bash
# 1. Set up environment
export PINECONE_API_KEY='pc-xxx'
export GOOGLE_API_KEY='AIza-xxx'

# 2. Add your documents to my-documents/
cp ~/Documents/report.pdf my-documents/

# 3. Run the setup script
./setup_and_query.sh

# 4. Query your documents
python3 ask.py "What are the main findings in the report?"
python3 ask.py "Summarize the key recommendations"
python3 ask.py "What data sources were used?"
```

## What's Next?

- Check `typed_rag/data/chunks.jsonl` to see how your documents were chunked
- Modify chunk size and overlap for better results
- Adjust the number of retrieved passages (k parameter)
- Use different LLM models (edit model parameter in ask.py)

