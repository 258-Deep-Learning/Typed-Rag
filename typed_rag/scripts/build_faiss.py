#!/usr/bin/env python3
"""
Build a FAISS vector index (LangChain) from chunks.jsonl using BAAI/bge-small-en-v1.5.

Artifacts written in out_dir via FAISS.save_local():
  - index.faiss
  - index.pkl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog

# Allow running as a script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import BGEEmbedder, LCBGEEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore


logger = structlog.get_logger()


def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_faiss_langchain(
    out_dir: Path,
    chunk_records: List[Dict[str, Any]],
    device: Optional[str] = None,
) -> None:
    if not chunk_records:
        logger.warning("No chunks to index")
        return

    # Prepare LC embeddings adapter
    lc_embeddings = LCBGEEmbeddings(BGEEmbedder(device=device))

    # Build LC FAISS vectorstore
    docs: List[Document] = []
    ids: List[str] = []
    for r in chunk_records:
        text = r.get("text", "")
        meta = {k: v for k, v in r.items() if k != "text"}
        # also copy id into metadata for easier retrieval later
        meta["id"] = r.get("id")
        docs.append(Document(page_content=text, metadata=meta))
        ids.append(r.get("id", ""))

    logger.info("Creating LangChain FAISS store", count=len(docs))
    # Initialize empty store (embedding function provided) and add docs with ids
    vs = FAISS.from_documents(docs, lc_embeddings)

    # Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    logger.info("Saved LangChain FAISS store", dir=str(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input chunks.jsonl file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write FAISS artifacts")
    parser.add_argument("--device", type=str, default=None, help="Device for embeddings (cuda/mps/cpu)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist", input=str(input_path))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(out_dir / "faiss.index")
    meta_path = str(out_dir / "faiss_meta.json")

    logger.info("Loading chunks", input=str(input_path))
    chunks = load_chunks(input_path)
    logger.info("Loaded chunks", count=len(chunks))
    if not chunks:
        logger.warning("No chunks to process")
        return

    logger.info("Building LangChain FAISS store", out_dir=str(out_dir))
    build_faiss_langchain(out_dir=out_dir, chunk_records=chunks, device=args.device)
    logger.info("Build complete", index=str(out_dir / "index.faiss"), meta=str(out_dir / "index.pkl"), total_chunks=len(chunks))


if __name__ == "__main__":
    main()


