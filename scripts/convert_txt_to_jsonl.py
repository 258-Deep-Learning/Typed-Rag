#!/usr/bin/env python3
"""
Convert text files to JSONL format for Typed-RAG system.

Usage:
    python scripts/convert_txt_to_jsonl.py --input-dir my_documents/ --output data/my_docs.jsonl
"""

import json
import os
from pathlib import Path
import typer
from typing import List

app = typer.Typer(help="Convert text files to JSONL format for RAG indexing")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks by words.
    
    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Skip empty chunks
            chunks.append(' '.join(chunk_words))
    
    return chunks


def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Intelligent chunking that respects sentence boundaries.
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    import re
    
    # Split by sentences (handles ., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        if current_length + sentence_length <= chunk_size:
            current_chunk.extend(sentence_words)
            current_length += sentence_length
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if len(current_chunk) > overlap:
                overlap_words = current_chunk[-overlap:]
            else:
                overlap_words = current_chunk
            
            current_chunk = overlap_words + sentence_words
            current_length = len(current_chunk)
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


@app.command()
def main(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Directory containing text files"),
    output: str = typer.Option("data/my_documents.jsonl", "--output", "-o", help="Output JSONL file"),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Words per chunk"),
    overlap: int = typer.Option(100, "--overlap", help="Overlapping words between chunks"),
    smart_chunking: bool = typer.Option(True, "--smart/--simple", help="Use smart sentence-aware chunking"),
    recursive: bool = typer.Option(True, "--recursive/--flat", help="Search subdirectories recursively"),
    extensions: List[str] = typer.Option([".txt", ".md"], "--ext", help="File extensions to process")
):
    """
    Convert all text files in a directory to JSONL format.
    """
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        typer.echo(f"❌ Error: Directory '{input_dir}' does not exist", err=True)
        raise typer.Exit(code=1)
    
    # Create output directory if needed
    output_path = Path(output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Find all text files
    text_files = []
    for ext in extensions:
        if recursive:
            text_files.extend(input_path.rglob(f"*{ext}"))
        else:
            text_files.extend(input_path.glob(f"*{ext}"))
    
    if not text_files:
        typer.echo(f"❌ No files found with extensions {extensions} in {input_dir}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Found {len(text_files)} text files")
    
    # Process files
    documents = []
    chunk_function = smart_chunk_text if smart_chunking else chunk_text
    
    for filepath in text_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                typer.echo(f"⚠️  Skipping empty file: {filepath}")
                continue
            
            # Create chunks
            chunks = chunk_function(content, chunk_size=chunk_size, overlap=overlap)
            
            # Create document records
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": f"{filepath.stem}_{i}",
                    "title": filepath.stem.replace("_", " ").replace("-", " ").title(),
                    "url": f"file://{filepath.absolute()}",
                    "chunk_text": chunk
                }
                documents.append(doc)
            
            typer.echo(f"✅ Processed: {filepath.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            typer.echo(f"❌ Error processing {filepath}: {e}", err=True)
    
    # Write JSONL output
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    typer.echo(f"\n✅ Conversion complete!")
    typer.echo(f"   Files processed: {len(text_files)}")
    typer.echo(f"   Total chunks: {len(documents)}")
    typer.echo(f"   Output: {output_path}")
    typer.echo(f"\nNext steps:")
    typer.echo(f"   1. Build BM25 index: python scripts/build_bm25.py --input {output}")
    typer.echo(f"   2. Build FAISS index: python scripts/build_faiss.py --input {output}")
    typer.echo(f"   3. Query: python scripts/query.py --q 'your question' --k 5 --mode hybrid")


if __name__ == "__main__":
    app()

