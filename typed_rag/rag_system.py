#!/usr/bin/env python3
"""
Core RAG System Module - Clean separation between building and querying.

This module provides:
- RAGConfig: Configuration management
- IndexBuilder: Build indices (teammate can focus here)
- QueryEngine: Query indices (teammate can focus here)
- RAGSystem: Main orchestrator
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for RAG system - single source of truth."""
    
    # Directories
    repo_root: Path
    docs_dir: Path
    chunks_jsonl: Path
    faiss_dir: Path
    
    # Pinecone settings
    pinecone_index: str = "typedrag-own"
    pinecone_namespace: str = "own_docs"
    
    # Query settings
    top_k: int = 5
    llm_model: str = "gemini-2.5-flash"
    
    @classmethod
    def default(cls, repo_root: Optional[Path] = None) -> "RAGConfig":
        """Create default configuration."""
        if repo_root is None:
            repo_root = Path(__file__).parent.parent.resolve()
        
        return cls(
            repo_root=repo_root,
            docs_dir=repo_root / "my-documents",
            chunks_jsonl=repo_root / "typed_rag" / "data" / "chunks.jsonl",
            faiss_dir=repo_root / "typed_rag" / "indexes" / "faiss",
        )
    
    def validate_env(self, backend: str) -> None:
        """Validate required environment variables."""
        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError("GOOGLE_API_KEY environment variable not set")
        
        if backend == "pinecone" and not os.getenv("PINECONE_API_KEY"):
            raise EnvironmentError("PINECONE_API_KEY environment variable not set")


class IndexBuilder:
    """
    Handles all index building operations.
    Teammate working on indexing can focus here.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def ingest_documents(self, force_rebuild: bool = False) -> None:
        """Ingest documents and create chunks.jsonl."""
        if self.config.chunks_jsonl.exists() and not force_rebuild:
            print(f"✓ Using existing chunks: {self.config.chunks_jsonl}")
            return
        
        if not self.config.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.config.docs_dir}")
        
        print(f"📥 Ingesting documents from: {self.config.docs_dir}")
        
        # Import here to avoid circular deps
        from typed_rag.scripts import ingest_own_docs as ingest
        
        chunks = ingest.ingest_directory(self.config.docs_dir)
        if not chunks:
            raise RuntimeError("No chunks generated (documents may be empty or unsupported)")
        
        ingest.save_chunks_jsonl(chunks, self.config.chunks_jsonl)
        print(f"✓ Saved {len(chunks)} chunks to: {self.config.chunks_jsonl}")
    
    def build_pinecone(self, force_rebuild: bool = False) -> None:
        """Build Pinecone index."""
        self.config.validate_env("pinecone")
        
        from typed_rag.scripts import build_pinecone as bp
        
        print("\n📦 Building Pinecone index...")
        
        records = bp.load_chunks(self.config.chunks_jsonl)
        if not records:
            raise RuntimeError("No records loaded from chunks.jsonl")
        
        embedder = bp.BGEEmbedder()
        store = bp.PineconeDenseStore(
            index_name=self.config.pinecone_index,
            namespace=self.config.pinecone_namespace,
            dimension=384,
            metric="cosine",
            create_if_missing=True,
        )
        bp.upsert_chunks_to_pinecone(store, embedder, records)
        print("✓ Pinecone index built successfully")
    
    def build_faiss(self, force_rebuild: bool = False) -> None:
        """Build FAISS index."""
        print("\n📦 Building FAISS index...")
        
        self.config.faiss_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        index_files = {"index.faiss", "index.pkl"}
        existing_files = {p.name for p in self.config.faiss_dir.glob("*")}
        
        if not force_rebuild and index_files.issubset(existing_files):
            print(f"✓ FAISS index already exists at {self.config.faiss_dir}")
            print("  (use --rebuild to force rebuild)")
            return
        
        # Build using the script
        cmd = [
            sys.executable,
            "-m",
            "typed_rag.scripts.build_faiss",
            "--in",
            str(self.config.chunks_jsonl),
            "--out_dir",
            str(self.config.faiss_dir),
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        print("✓ FAISS index built successfully")
    
    def build(self, backend: str, force_rebuild: bool = False) -> None:
        """
        Main entry point for building indices.
        
        Args:
            backend: 'pinecone' or 'faiss'
            force_rebuild: Force rebuild even if index exists
        """
        # Step 1: Ingest documents
        self.ingest_documents(force_rebuild=force_rebuild)
        
        # Step 2: Build index
        if backend == "pinecone":
            self.build_pinecone(force_rebuild=force_rebuild)
        elif backend == "faiss":
            self.build_faiss(force_rebuild=force_rebuild)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'pinecone' or 'faiss'")


class QueryEngine:
    """
    Handles all query operations.
    Teammate working on querying can focus here.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def query(self, question: str, backend: str) -> None:
        """
        Execute a query using ask.py.
        
        Args:
            question: The question to ask
            backend: 'pinecone' or 'faiss'
        """
        self.config.validate_env(backend)
        
        ask_script = self.config.repo_root / "ask.py"
        if not ask_script.exists():
            raise FileNotFoundError(f"ask.py not found at {ask_script}")
        
        # Validate backend-specific requirements
        if backend == "faiss":
            if not self.config.faiss_dir.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {self.config.faiss_dir}\n"
                    f"Build it first: python rag_cli.py build --backend faiss"
                )
        
        # Set up environment
        env = os.environ.copy()
        env["VECTOR_STORE"] = backend
        
        if backend == "pinecone":
            env["PINECONE_INDEX"] = self.config.pinecone_index
            env["PINECONE_NAMESPACE"] = self.config.pinecone_namespace
        elif backend == "faiss":
            env["FAISS_DIR"] = str(self.config.faiss_dir)
        
        # Execute query
        print("💬 Querying...\n")
        result = subprocess.run(
            [sys.executable, str(ask_script), question],
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError("Query failed")


class RAGSystem:
    """
    Main orchestrator - combines IndexBuilder and QueryEngine.
    Simple interface for common operations.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        if config is None:
            config = RAGConfig.default()
        
        self.config = config
        self.builder = IndexBuilder(config)
        self.query_engine = QueryEngine(config)
    
    def build_index(self, backend: str = "faiss", force_rebuild: bool = False) -> None:
        """Build an index."""
        self.builder.build(backend, force_rebuild=force_rebuild)
    
    def ask(self, question: str, backend: str = "faiss") -> None:
        """Ask a question."""
        self.query_engine.query(question, backend)
    
    def detect_default_backend(self) -> str:
        """
        Auto-detect which backend to use based on what's available.
        
        Priority:
        1. VECTOR_STORE environment variable
        2. FAISS index exists
        3. Pinecone API key exists
        4. Default to FAISS
        """
        env_choice = os.getenv("VECTOR_STORE", "").lower()
        if env_choice in {"faiss", "pinecone"}:
            return env_choice
        
        # Check if FAISS index exists
        index_files = {"index.faiss", "index.pkl"}
        if self.config.faiss_dir.exists():
            existing = {p.name for p in self.config.faiss_dir.glob("*")}
            if index_files.issubset(existing):
                return "faiss"
        
        # Check for Pinecone key
        if os.getenv("PINECONE_API_KEY"):
            return "pinecone"
        
        return "faiss"

