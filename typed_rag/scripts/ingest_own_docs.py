#!/usr/bin/env python3
"""
Ingest documents (PDF, DOCX, MD, TXT, HTML) and chunk them into JSONL format.
Chunks are 200 tokens with 60-token stride.
"""

import argparse
import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    url: str
    section: str
    chunk_idx: int
    text: str
    token_len: int
    source: str


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def chunk_text(
    text: str,
    doc_id: str,
    title: str,
    url: str,
    chunk_tokens: int = 200,
    stride_tokens: int = 60,
) -> List[Chunk]:
    """
    Chunk text into overlapping segments.
    Uses character-based approximation for tokens.
    """
    chunk_chars = chunk_tokens * 4
    stride_chars = stride_tokens * 4
    
    chunks = []
    start = 0
    idx = 0
    
    while start < len(text):
        end = start + chunk_chars
        chunk_text = text[start:end].strip()
        
        if not chunk_text:
            break
        
        chunk_id = f"{doc_id}::chunk_{idx:04d}"
        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=doc_id,
                title=title,
                url=url,
                section="",  # Can be enhanced later
                chunk_idx=idx,
                text=chunk_text,
                token_len=estimate_tokens(chunk_text),
                source="internal",
            )
        )
        
        idx += 1
        start += chunk_chars - stride_chars
        
        # Stop if we've reached the end
        if end >= len(text):
            break
    
    return chunks


def extract_text_from_file(file_path: Path) -> Optional[str]:
    """Extract text from various file formats."""
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".txt" or suffix == ".md":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        
        elif suffix == ".pdf":
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = []
                    for page in reader.pages:
                        text.append(page.extract_text())
                    return "\n\n".join(text)
            except ImportError:
                logger.warning("PyPDF2 not installed, skipping PDF", file=str(file_path))
                return None
        
        elif suffix == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                logger.warning("python-docx not installed, skipping DOCX", file=str(file_path))
                return None
        
        elif suffix == ".html" or suffix == ".htm":
            try:
                from bs4 import BeautifulSoup
                html = file_path.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator="\n")
            except ImportError:
                logger.warning("beautifulsoup4 not installed, skipping HTML", file=str(file_path))
                return None
        
        else:
            logger.warning("Unsupported file type", file=str(file_path), suffix=suffix)
            return None
            
    except Exception as e:
        logger.error("Error extracting text", file=str(file_path), error=str(e))
        return None


def generate_doc_id(file_path: Path) -> str:
    """Generate a stable doc_id from file path."""
    return hashlib.md5(str(file_path).encode()).hexdigest()[:12]


def ingest_directory(
    root_dir: Path,
    chunk_tokens: int = 200,
    stride_tokens: int = 60,
) -> List[Chunk]:
    """Recursively ingest all documents from a directory."""
    all_chunks = []
    supported_exts = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
    
    files = [f for f in root_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts]
    logger.info("Found files", count=len(files), root=str(root_dir))
    
    for file_path in files:
        logger.info("Processing file", file=str(file_path))
        
        text = extract_text_from_file(file_path)
        if not text or len(text.strip()) < 100:
            logger.warning("Skipping file (too short or empty)", file=str(file_path))
            continue
        
        doc_id = generate_doc_id(file_path)
        title = file_path.stem.replace("_", " ").replace("-", " ").title()
        url = f"file://{file_path.absolute()}"
        
        chunks = chunk_text(text, doc_id, title, url, chunk_tokens, stride_tokens)
        all_chunks.extend(chunks)
        logger.info("Chunked file", file=str(file_path), num_chunks=len(chunks))
    
    return all_chunks


def save_chunks_jsonl(chunks: List[Chunk], output_path: Path) -> None:
    """Save chunks to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "url": chunk.url,
                "section": chunk.section,
                "chunk_idx": chunk.chunk_idx,
                "text": chunk.text,
                "token_len": chunk.token_len,
                "source": chunk.source,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info("Saved chunks", output=str(output_path), count=len(chunks))


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk documents")
    parser.add_argument("--root", type=str, required=True, help="Root directory with documents")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--chunk_tokens", type=int, default=200, help="Tokens per chunk")
    parser.add_argument("--stride_tokens", type=int, default=60, help="Overlap stride in tokens")
    
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    if not root_dir.exists():
        logger.error("Root directory does not exist", root=str(root_dir))
        return
    
    chunks = ingest_directory(root_dir, args.chunk_tokens, args.stride_tokens)
    
    if not chunks:
        logger.warning("No chunks generated!")
        return
    
    save_chunks_jsonl(chunks, Path(args.out))
    logger.info("Ingestion complete", total_chunks=len(chunks))


if __name__ == "__main__":
    main()

