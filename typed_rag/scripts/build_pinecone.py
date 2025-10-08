#!/usr/bin/env python3
"""
Build Pinecone vector index from chunks.jsonl
Embeds using BAAI/bge-small-en-v1.5 and uploads to Pinecone.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import BGEEmbedder, PineconeDenseStore


def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    """Load all chunks from JSONL."""
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def upsert_chunks_to_pinecone(
    store: PineconeDenseStore,
    embedder: BGEEmbedder,
    chunk_records: List[Dict[str, Any]],
    batch_size: int = 256,
) -> None:
    """Embed and upsert chunks to Pinecone."""
    if not chunk_records:
        logger.warning("No chunks to upsert")
        return
    
    logger.info("Starting embedding and upsert", total_chunks=len(chunk_records))
    
    ids = [r["id"] for r in chunk_records]
    texts = [r["text"] for r in chunk_records]
    
    # Prepare metadata (exclude 'text' field to save space in Pinecone)
    metadatas = []
    for r in chunk_records:
        meta = {k: v for k, v in r.items() if k != "text"}
        metadatas.append(meta)
    
    # Embed in batches
    logger.info("Encoding passages", count=len(texts))
    vecs = embedder.encode_passages(texts, batch_size=batch_size)
    logger.info("Encoding complete", embedding_dim=vecs.shape[1])
    
    # Upsert to Pinecone
    logger.info("Upserting to Pinecone", batch_size=batch_size)
    store.upsert(ids=ids, vectors=vecs, metadatas=metadatas, batch_size=batch_size)
    logger.info("Upsert complete")


def main():
    parser = argparse.ArgumentParser(description="Build Pinecone vector index")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input chunks.jsonl file")
    parser.add_argument("--index", type=str, default="typedrag-own", help="Pinecone index name")
    parser.add_argument("--namespace", type=str, default="own_docs", help="Pinecone namespace")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding and upsert")
    parser.add_argument("--device", type=str, default=None, help="Device for embeddings (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY environment variable not set")
        logger.info("Please set it: export PINECONE_API_KEY='your-key-here'")
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist", input=str(input_path))
        return
    
    # Load chunks
    logger.info("Loading chunks", input=str(input_path))
    chunks = load_chunks(input_path)
    logger.info("Loaded chunks", count=len(chunks))
    
    if not chunks:
        logger.warning("No chunks to process")
        return
    
    # Initialize embedder and store
    logger.info("Initializing embedder", model="BAAI/bge-small-en-v1.5", device=args.device)
    embedder = BGEEmbedder(device=args.device)
    
    logger.info("Initializing Pinecone store", index=args.index, namespace=args.namespace)
    store = PineconeDenseStore(
        index_name=args.index,
        namespace=args.namespace,
        dimension=384,
        metric="cosine",
        create_if_missing=True,
    )
    
    # Upsert
    upsert_chunks_to_pinecone(store, embedder, chunks, batch_size=args.batch_size)
    logger.info("Build complete", index=args.index, namespace=args.namespace, total_chunks=len(chunks))


if __name__ == "__main__":
    main()

