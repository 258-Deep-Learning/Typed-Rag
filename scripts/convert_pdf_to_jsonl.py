#!/usr/bin/env python3
"""
Convert PDF files to JSONL format for Typed-RAG system.

Usage:
    pip install PyPDF2  # Install dependency first
    python scripts/convert_pdf_to_jsonl.py --input-dir my_pdfs/ --output data/my_docs.jsonl
"""

import json
import os
from pathlib import Path
import typer
from typing import List

app = typer.Typer(help="Convert PDF files to JSONL format for RAG indexing")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
    
    return chunks


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import PyPDF2
    except ImportError:
        typer.echo("❌ PyPDF2 not installed. Install it with: pip install PyPDF2", err=True)
        raise typer.Exit(code=1)
    
    text = ""
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    typer.echo(f"⚠️  Warning: Could not extract text from page {page_num} of {pdf_path.name}: {e}")
        
        return text.strip()
        
    except Exception as e:
        typer.echo(f"❌ Error reading PDF {pdf_path.name}: {e}", err=True)
        return ""


@app.command()
def main(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Directory containing PDF files"),
    output: str = typer.Option("data/my_documents.jsonl", "--output", "-o", help="Output JSONL file"),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Words per chunk"),
    overlap: int = typer.Option(100, "--overlap", help="Overlapping words between chunks"),
    recursive: bool = typer.Option(True, "--recursive/--flat", help="Search subdirectories recursively"),
    min_chunk_words: int = typer.Option(50, "--min-words", help="Minimum words per chunk (filters out noise)")
):
    """
    Convert all PDF files in a directory to JSONL format.
    """
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        typer.echo(f"❌ Error: Directory '{input_dir}' does not exist", err=True)
        raise typer.Exit(code=1)
    
    # Create output directory if needed
    output_path = Path(output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Find all PDF files
    if recursive:
        pdf_files = list(input_path.rglob("*.pdf"))
    else:
        pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        typer.echo(f"❌ No PDF files found in {input_dir}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Found {len(pdf_files)} PDF files")
    
    # Process files
    documents = []
    
    for pdf_path in pdf_files:
        typer.echo(f"Processing: {pdf_path.name}...")
        
        # Extract text
        content = extract_text_from_pdf(pdf_path)
        
        # Skip if no content extracted
        if not content.strip():
            typer.echo(f"⚠️  Skipping (no text extracted): {pdf_path.name}")
            continue
        
        # Create chunks
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        
        # Filter out very short chunks (often noise)
        chunks = [c for c in chunks if len(c.split()) >= min_chunk_words]
        
        if not chunks:
            typer.echo(f"⚠️  Skipping (no valid chunks after filtering): {pdf_path.name}")
            continue
        
        # Create document records
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{pdf_path.stem}_chunk_{i}",
                "title": pdf_path.stem.replace("_", " ").replace("-", " ").title(),
                "url": f"file://{pdf_path.absolute()}#page={i}",
                "chunk_text": chunk
            }
            documents.append(doc)
        
        typer.echo(f"✅ Processed: {pdf_path.name} ({len(chunks)} chunks)")
    
    # Write JSONL output
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    typer.echo(f"\n✅ Conversion complete!")
    typer.echo(f"   PDFs processed: {len(pdf_files)}")
    typer.echo(f"   Total chunks: {len(documents)}")
    typer.echo(f"   Output: {output_path}")
    typer.echo(f"\nNext steps:")
    typer.echo(f"   1. Build BM25 index: python scripts/build_bm25.py --input {output}")
    typer.echo(f"   2. Build FAISS index: python scripts/build_faiss.py --input {output}")
    typer.echo(f"   3. Query: python scripts/query.py --q 'your question' --k 5 --mode hybrid")


if __name__ == "__main__":
    app()

