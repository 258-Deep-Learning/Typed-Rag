# Makefile for Typed-RAG Weekend 1
# Makes it easy to run the pipeline

.PHONY: help setup ingest build-bm25 build-pinecone dev-set baseline-llm baseline-rag all clean

# Default paths
DOCS_DIR ?= my_documents
CHUNKS_FILE = typed_rag/data/chunks.jsonl
DEV_SET_FILE = typed_rag/data/dev_set.jsonl
BM25_INDEX = typed_rag/indexes/lucene_own
PINECONE_INDEX = typedrag-own
PINECONE_NAMESPACE = own_docs

help:
	@echo "Typed-RAG Weekend 1 Pipeline Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup              - Create necessary directories"
	@echo ""
	@echo "Pipeline Steps:"
	@echo "  make ingest             - Ingest and chunk documents from DOCS_DIR"
	@echo "  make build-bm25         - Build BM25 index"
	@echo "  make build-pinecone     - Build Pinecone vector index"
	@echo "  make dev-set            - Generate dev set questions"
	@echo "  make baseline-llm       - Run LLM-only baseline"
	@echo "  make baseline-rag       - Run RAG baseline"
	@echo ""
	@echo "Convenience:"
	@echo "  make all                - Run entire pipeline (ingest -> indexes -> baselines)"
	@echo "  make clean              - Clean generated files (keeps indexes)"
	@echo ""
	@echo "Variables:"
	@echo "  DOCS_DIR=path           - Path to your documents (default: my_documents)"

setup:
	@echo "Creating directories..."
	mkdir -p typed_rag/data
	mkdir -p typed_rag/indexes
	mkdir -p typed_rag/runs
	mkdir -p $(DOCS_DIR)
	@echo "✓ Directories created"

ingest: setup
	@echo "Ingesting documents from $(DOCS_DIR)..."
	python3 typed_rag/scripts/ingest_own_docs.py \
		--root $(DOCS_DIR) \
		--out $(CHUNKS_FILE) \
		--chunk_tokens 200 \
		--stride_tokens 60
	@echo "✓ Documents chunked"

build-bm25: setup
	@echo "Building BM25 index..."
	python3 typed_rag/scripts/build_bm25.py \
		--in $(CHUNKS_FILE) \
		--index $(BM25_INDEX)
	@echo "✓ BM25 index built"

build-pinecone: setup
	@echo "Building Pinecone vector index..."
	python3 typed_rag/scripts/build_pinecone.py \
		--in $(CHUNKS_FILE) \
		--index $(PINECONE_INDEX) \
		--namespace $(PINECONE_NAMESPACE)
	@echo "✓ Pinecone index built"

dev-set: setup
	@echo "Generating dev set..."
	python3 typed_rag/scripts/make_own_dev.py \
		--root $(CHUNKS_FILE) \
		--out $(DEV_SET_FILE) \
		--count 100
	@echo "✓ Dev set generated"

baseline-llm: setup
	@echo "Running LLM-only baseline..."
	python3 typed_rag/scripts/run_llm_only.py \
		--in $(DEV_SET_FILE) \
		--out typed_rag/runs/llm_only.jsonl \
		--model gemini-2.0-flash-exp
	@echo "✓ LLM-only baseline complete"

baseline-rag: setup
	@echo "Running RAG baseline..."
	python3 typed_rag/scripts/run_rag_baseline.py \
		--in $(DEV_SET_FILE) \
		--out typed_rag/runs/rag_baseline.jsonl \
		--pinecone_index $(PINECONE_INDEX) \
		--pinecone_namespace $(PINECONE_NAMESPACE) \
		--bm25_index $(BM25_INDEX) \
		--k 5 \
		--model gemini-2.0-flash-exp
	@echo "✓ RAG baseline complete"

all: ingest build-bm25 build-pinecone dev-set baseline-llm baseline-rag
	@echo ""
	@echo "🎉 All done! Check typed_rag/runs/ for results."

clean:
	@echo "Cleaning generated files..."
	rm -f $(CHUNKS_FILE)
	rm -f $(DEV_SET_FILE)
	rm -f typed_rag/runs/*.jsonl
	@echo "✓ Cleaned (indexes preserved)"

clean-all: clean
	@echo "Removing indexes..."
	rm -rf $(BM25_INDEX)
	@echo "✓ All cleaned (you'll need to rebuild Pinecone manually)"

