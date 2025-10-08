#!/usr/bin/env python3
"""
Build BM25 index using Pyserini from chunks.jsonl
"""

import argparse
import json
import tempfile
import shutil
from pathlib import Path
from typing import Iterator, Dict, Any
import structlog

logger = structlog.get_logger()


def load_chunks(input_path: Path) -> Iterator[Dict[str, Any]]:
    """Load chunks from JSONL."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_index(input_jsonl: Path, index_dir: Path) -> None:
    """Build Pyserini/Lucene BM25 index."""
    from pyserini.index.lucene import IndexReader, LuceneIndexer
    
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for the collection
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        collection_path = temp_path / "collection"
        collection_path.mkdir()
        
        # Copy the JSONL file to temp location
        temp_jsonl = collection_path / "chunks.jsonl"
        shutil.copy(input_jsonl, temp_jsonl)
        
        logger.info("Building BM25 index", input=str(input_jsonl), index=str(index_dir))
        
        # Build the index
        indexer = LuceneIndexer(str(index_dir), store_contents=True)
        
        # Read and index each document
        chunk_count = 0
        for chunk in load_chunks(temp_jsonl):
            # Create a document for indexing
            doc = {
                "id": chunk["id"],
                "contents": chunk["text"],  # This is what gets searched
                "raw": json.dumps(chunk, ensure_ascii=False),  # Store full metadata
            }
            indexer.add_doc_dict(doc)
            chunk_count += 1
            
            if chunk_count % 1000 == 0:
                logger.info("Indexed chunks", count=chunk_count)
        
        indexer.close()
        logger.info("BM25 index built", total_chunks=chunk_count, index_dir=str(index_dir))


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index with Pyserini")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input chunks.jsonl file")
    parser.add_argument("--index", type=str, required=True, help="Output index directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist", input=str(input_path))
        return
    
    index_dir = Path(args.index)
    build_index(input_path, index_dir)
    logger.info("Index build complete")


if __name__ == "__main__":
    main()

