#!/usr/bin/env python3
"""
Multi-backend query interface for Typed-RAG system.

Supports multiple vector backends:
- faiss: Local FAISS index (default, fast)
- pinecone: Pinecone managed cloud service
- qdrant: Qdrant vector database
- weaviate: Weaviate vector database

Usage:
    # Local FAISS (default)
    python scripts/query_multi.py --q "your question" --k 5 --mode hybrid
    
    # Pinecone backend
    python scripts/query_multi.py --q "your question" --backend pinecone
    
    # Custom documents
    python scripts/query_multi.py --q "your question" --index-dir indexes/custom
    
    # Wikipedia + Pinecone
    python scripts/query_multi.py --q "your question" --backend pinecone --index-dir indexes
"""

import json
import os
import sys
from pathlib import Path
import typer
from typing import Literal, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.multi_backend import MultiBackendRetriever

app = typer.Typer(help="Multi-backend query interface with support for FAISS and vector databases")


@app.command()
def query(
    q: str = typer.Option(..., "--q", help="Query string"),
    k: int = typer.Option(5, "--k", help="Number of results to return"),
    mode: Literal["bm25", "vector", "hybrid"] = typer.Option("hybrid", "--mode", help="Search mode"),
    backend: Literal["faiss", "pinecone", "qdrant", "weaviate"] = typer.Option("faiss", "--backend", help="Vector backend to use"),
    index_dir: str = typer.Option("indexes", "--index-dir", help="Index directory (e.g., 'indexes' for Wikipedia, 'indexes/custom' for custom docs)"),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed information"),
    # Pinecone settings
    pinecone_api_key: Optional[str] = typer.Option(None, "--pinecone-api-key", help="Pinecone API key (or set PINECONE_API_KEY env var)"),
    pinecone_env: Optional[str] = typer.Option(None, "--pinecone-env", help="Pinecone environment (or set PINECONE_ENVIRONMENT env var)"),
    pinecone_index: str = typer.Option("typed-rag", "--pinecone-index", help="Pinecone index name"),
    # Qdrant settings
    qdrant_url: str = typer.Option("localhost", "--qdrant-url", help="Qdrant host"),
    qdrant_port: int = typer.Option(6333, "--qdrant-port", help="Qdrant port"),
    qdrant_collection: str = typer.Option("typed-rag", "--qdrant-collection", help="Qdrant collection name"),
    # Weaviate settings
    weaviate_url: str = typer.Option("http://localhost:8080", "--weaviate-url", help="Weaviate URL"),
    weaviate_class: str = typer.Option("Document", "--weaviate-class", help="Weaviate class name"),
):
    """
    Query the retrieval system with configurable backend.
    """
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if not os.path.isabs(index_dir):
        index_dir = str(project_root / index_dir)
    
    bm25_path = f"{index_dir}/bm25_rank.pkl"
    meta_path = f"{index_dir}/meta.jsonl"
    
    if backend == "faiss":
        faiss_dir = f"{index_dir}/faiss_bge_small"
        if not os.path.exists(faiss_dir):
            typer.echo(f"❌ FAISS directory not found: {faiss_dir}", err=True)
            typer.echo(f"   Build it with: python scripts/build_faiss.py --input data/passages.jsonl", err=True)
            raise typer.Exit(code=1)
    
    if verbose:
        typer.echo(f"Configuration:")
        typer.echo(f"  Query: {q}")
        typer.echo(f"  Mode: {mode}")
        typer.echo(f"  Backend: {backend}")
        typer.echo(f"  Index dir: {index_dir}")
        typer.echo(f"  Top-k: {k}")
        typer.echo()
    
    # Initialize retriever
    try:
        retriever = MultiBackendRetriever(
            bm25_path=bm25_path,
            meta_path=meta_path,
            vector_backend=backend,
            faiss_dir=f"{index_dir}/faiss_bge_small" if backend == "faiss" else "",
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_env,
            pinecone_index_name=pinecone_index,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            qdrant_collection_name=qdrant_collection,
            weaviate_url=weaviate_url,
            weaviate_class_name=weaviate_class,
        )
    except Exception as e:
        typer.echo(f"❌ Failed to initialize retriever: {e}", err=True)
        raise typer.Exit(code=1)
    
    if verbose:
        health = retriever.health_check()
        typer.echo("System Health:")
        for key, value in health.items():
            status = "✅" if value else "❌"
            typer.echo(f"  {status} {key}: {value}")
        typer.echo()
    
    # Perform search
    try:
        results = retriever.retrieve(query=q, k=k, mode=mode)
        
        if verbose:
            typer.echo(f"Found {len(results)} results\n")
        
        # Output JSON
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        typer.echo(f"❌ Search failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def health(
    index_dir: str = typer.Option("indexes", "--index-dir", help="Index directory"),
    backend: Literal["faiss", "pinecone", "qdrant", "weaviate"] = typer.Option("faiss", "--backend", help="Vector backend to check"),
):
    """
    Check system health for specified backend.
    """
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if not os.path.isabs(index_dir):
        index_dir = str(project_root / index_dir)
    
    typer.echo(f"🏥 Retrieval System Health Check")
    typer.echo(f"=" * 60)
    typer.echo(f"Backend: {backend}")
    typer.echo(f"Index directory: {index_dir}")
    typer.echo(f"=" * 60)
    
    try:
        retriever = MultiBackendRetriever(
            bm25_path=f"{index_dir}/bm25_rank.pkl",
            meta_path=f"{index_dir}/meta.jsonl",
            vector_backend=backend,
            faiss_dir=f"{index_dir}/faiss_bge_small" if backend == "faiss" else "",
        )
        
        health = retriever.health_check()
        
        all_ok = True
        for key, value in health.items():
            status = "✅" if value else "❌"
            typer.echo(f"{status} {key}: {value}")
            if not value and key != "vector_loaded":  # vector_loaded might be false if backend not set up yet
                all_ok = False
        
        typer.echo(f"=" * 60)
        if all_ok:
            typer.echo("🎉 All systems operational!")
        else:
            typer.echo("⚠️  Some components are not loaded. Check configuration.")
        
    except Exception as e:
        typer.echo(f"❌ Health check failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def demo(
    backend: Literal["faiss", "pinecone", "qdrant", "weaviate"] = typer.Option("faiss", "--backend", help="Vector backend to use"),
    index_dir: str = typer.Option("indexes", "--index-dir", help="Index directory"),
):
    """
    Run a demo query to test the system.
    """
    
    demo_queries = [
        "Who discovered penicillin?",
        "What is machine learning?",
        "Tell me about neural networks",
    ]
    
    typer.echo(f"🚀 Running demo with backend: {backend}\n")
    
    for q in demo_queries:
        typer.echo(f"Query: {q}")
        typer.echo("-" * 60)
        
        try:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            
            if not os.path.isabs(index_dir):
                index_dir = str(project_root / index_dir)
            
            retriever = MultiBackendRetriever(
                bm25_path=f"{index_dir}/bm25_rank.pkl",
                meta_path=f"{index_dir}/meta.jsonl",
                vector_backend=backend,
                faiss_dir=f"{index_dir}/faiss_bge_small" if backend == "faiss" else "",
            )
            
            results = retriever.retrieve(query=q, k=3, mode="hybrid")
            
            for i, result in enumerate(results, 1):
                typer.echo(f"\n{i}. {result.get('title', 'No title')}")
                typer.echo(f"   Score: {result.get('score', 0):.3f}")
                chunk = result.get('chunk_text', '')
                typer.echo(f"   Preview: {chunk[:150]}...")
            
            typer.echo("\n" + "=" * 60 + "\n")
            
        except Exception as e:
            typer.echo(f"❌ Demo failed: {e}\n", err=True)


if __name__ == "__main__":
    app()

