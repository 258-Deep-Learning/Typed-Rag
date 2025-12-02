#!/usr/bin/env python3
"""
Core RAG System Module - Clean separation between building and querying.

This module provides:
- DataType: Specifies backend type and data source
- RAGConfig: Configuration management
- IndexBuilder: Build indices (teammate can focus here)
- QueryEngine: Query indices (teammate can focus here)
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from typed_rag.classifier import classify_question
from typed_rag.core.keys import get_fastest_model
from typed_rag.decompose import decompose_question
from typed_rag.generation import TypedAnswerAggregator, TypedAnswerGenerator
from typed_rag.retrieval.orchestrator import RetrievalOrchestrator
from typed_rag.retrieval.pipeline import (
    BGEEmbedder,
    PineconeDenseStore,
    load_faiss_adapter,
)
from typed_rag.scripts import build_pinecone as bp
from typed_rag.scripts import ingest_own_docs as ingest

import ask

# ============================================================================
# Helper utilities
# ============================================================================

def detect_default_backend(source: str = "own_docs") -> str:
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
    config = RAGConfig.default()
    paths = config.get_paths_for_source(source)
    faiss_dir = paths["faiss_dir"]
    
    index_files = {"index.faiss", "index.pkl"}
    if faiss_dir.exists():
        existing = {p.name for p in faiss_dir.glob("*")}
        if index_files.issubset(existing):
            return "faiss"
    
    # Check for Pinecone key
    if os.getenv("PINECONE_API_KEY"):
        return "pinecone"
    
    return "faiss"


class DataType:
    """
    Specifies the RAG configuration.
    
    Attributes:
        type: Backend type - 'pinecone' or 'faiss'
        source: Data source - 'own_docs' or 'wikipedia'
    
    Example:
        data_type = DataType('pinecone', 'own_docs')
        data_type = DataType('faiss', 'wikipedia')
    """
    def __init__(self, type: str, source: str):
        if type not in {'pinecone', 'faiss'}:
            raise ValueError(f"Invalid type '{type}'. Must be 'pinecone' or 'faiss'")
        if source not in {'own_docs', 'wikipedia'}:
            raise ValueError(f"Invalid source '{source}'. Must be 'own_docs' or 'wikipedia'")
        
        self.type = type
        self.source = source
    
    def __repr__(self):
        return f"DataType(type='{self.type}', source='{self.source}')"


@dataclass
class RAGConfig:
    """Configuration for RAG system - single source of truth."""
    
    # Directories
    repo_root: Path
    # Query settings
    top_k: int = 5
    llm_model = get_fastest_model()
    
    @classmethod
    def default(cls, repo_root: Optional[Path] = None) -> "RAGConfig":
        """Create default configuration."""
        if repo_root is None:
            repo_root = Path(__file__).parent.parent.resolve()
        
        return cls(repo_root=repo_root)
    
    def get_paths_for_source(self, source: str) -> dict:
        """Get paths based on source type."""
        if source == "own_docs":
            return {
                "docs_dir": self.repo_root / "my-documents",
                "chunks_jsonl": self.repo_root / "typed_rag" / "data" / "chunks.jsonl",
                "faiss_dir": self.repo_root / "typed_rag" / "indexes" / "faiss",
                "pinecone_index": "typedrag-own",
                "pinecone_namespace": "own_docs",
            }
        elif source == "wikipedia":
            return {
                "docs_dir": None,  # Wikipedia doesn't have docs dir
                "chunks_jsonl": self.repo_root / "data" / "passages.jsonl",
                "faiss_dir": self.repo_root / "indexes" / "wikipedia" / "faiss",
                "pinecone_index": "typedrag-wiki",
                "pinecone_namespace": "wikipedia",
            }
        else:
            raise ValueError(f"Unknown source: {source}")
    
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
    
    def ingest_documents(self, data_type: DataType, force_rebuild: bool = False) -> None:
        """Ingest documents and create chunks.jsonl."""
        paths = self.config.get_paths_for_source(data_type.source)
        chunks_jsonl = paths["chunks_jsonl"]
        
        if chunks_jsonl.exists() and not force_rebuild:
            print(f"✓ Using existing chunks: {chunks_jsonl}")
            return
        
        # For own_docs, ingest from directory
        if data_type.source == "own_docs":
            docs_dir = paths["docs_dir"]
            if not docs_dir.exists():
                raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
            
            print(f"📥 Ingesting documents from: {docs_dir}")
            
            
            chunks = ingest.ingest_directory(docs_dir)
            if not chunks:
                raise RuntimeError("No chunks generated (documents may be empty or unsupported)")
            
            ingest.save_chunks_jsonl(chunks, chunks_jsonl)
            print(f"✓ Saved {len(chunks)} chunks to: {chunks_jsonl}")
        
        elif data_type.source == "wikipedia":
            # Wikipedia uses pre-existing passages.jsonl
            if not chunks_jsonl.exists():
                raise FileNotFoundError(
                    f"Wikipedia passages file not found: {chunks_jsonl}\n"
                    "This file should already exist in your data directory."
                )
            print(f"✓ Using Wikipedia passages: {chunks_jsonl}")
    
    def build_pinecone(self, data_type: DataType, force_rebuild: bool = False) -> None:
        """Build Pinecone index."""
        self.config.validate_env("pinecone")
        
        
        paths = self.config.get_paths_for_source(data_type.source)
        
        print(f"\n📦 Building Pinecone index for {data_type.source}...")
        
        records = bp.load_chunks(paths["chunks_jsonl"])
        if not records:
            raise RuntimeError("No records loaded from chunks file")
        
        embedder = bp.BGEEmbedder()
        store = bp.PineconeDenseStore(
            index_name=paths["pinecone_index"],
            namespace=paths["pinecone_namespace"],
            dimension=384,
            metric="cosine",
            create_if_missing=True,
        )
        bp.upsert_chunks_to_pinecone(store, embedder, records)
        print("✓ Pinecone index built successfully")
    
    def build_faiss(self, data_type: DataType, force_rebuild: bool = False) -> None:
        """Build FAISS index."""
        paths = self.config.get_paths_for_source(data_type.source)
        faiss_dir = paths["faiss_dir"]
        
        print(f"\n📦 Building FAISS index for {data_type.source}...")
        
        faiss_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        index_files = {"index.faiss", "index.pkl"}
        existing_files = {p.name for p in faiss_dir.glob("*")}
        
        if not force_rebuild and index_files.issubset(existing_files):
            print(f"✓ FAISS index already exists at {faiss_dir}")
            print("  (use --rebuild to force rebuild)")
            return
        
        # Build using the script
        cmd = [
            sys.executable,
            "-m",
            "typed_rag.scripts.build_faiss",
            "--in",
            str(paths["chunks_jsonl"]),
            "--out_dir",
            str(faiss_dir),
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print("✓ FAISS index built successfully")
    
    def build(self, data_type: DataType, force_rebuild: bool = True) -> None:
        """
        Main entry point for building indices.
        
        Args:
            data_type: DataType specifying backend and source
            force_rebuild: Force rebuild even if index exists
        """
        # Step 1: Ingest documents
        self.ingest_documents(data_type, force_rebuild=force_rebuild)
        
        # Step 2: Build index
        if data_type.type == "pinecone":
            self.build_pinecone(data_type, force_rebuild=force_rebuild)
        elif data_type.type == "faiss":
            self.build_faiss(data_type, force_rebuild=force_rebuild)
        else:
            raise ValueError(f"Unsupported backend: {data_type.type}")


class QueryEngine:
    """
    Handles all query operations.
    Teammate working on querying can focus here.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers (readability only; behavior unchanged)
    # ------------------------------------------------------------------
    def _get_ask_script(self) -> Path:
        """Resolve and verify the path to ask.py in the repo root."""
        ask_script = self.config.repo_root / "ask.py"
        if not ask_script.exists():
            raise FileNotFoundError(f"ask.py not found at {ask_script}")
        return ask_script

    def _validate_backend_requirements(self, data_type: DataType, paths: dict) -> None:
        """Validate backend-specific prerequisites (e.g., FAISS index exists)."""
        if data_type.type == "faiss":
            faiss_dir = paths["faiss_dir"]
            if not faiss_dir.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {faiss_dir}\n"
                    f"Build it first: python rag_cli.py build --backend faiss --source {data_type.source}"
                )

    def _load_vector_store(
        self,
        data_type: DataType,
        paths: dict,
        embedder: BGEEmbedder,
    ):
        """Instantiate the vector store corresponding to the backend."""
        if data_type.type == "faiss":
            faiss_dir = paths["faiss_dir"]
            return load_faiss_adapter(str(faiss_dir), embedder)

        return PineconeDenseStore(
            index_name=paths["pinecone_index"],
            namespace=paths["pinecone_namespace"],
            create_if_missing=False,
        )

    def _build_env(self, data_type: DataType, paths: dict) -> dict:
        """Prepare environment variables for the ask.py subprocess."""
        env = os.environ.copy()
        env["VECTOR_STORE"] = data_type.type
        if data_type.type == "pinecone":
            env["PINECONE_INDEX"] = paths["pinecone_index"]
            env["PINECONE_NAMESPACE"] = paths["pinecone_namespace"]
        elif data_type.type == "faiss":
            env["FAISS_DIR"] = str(paths["faiss_dir"])
        return env

    def query(self, question: str, data_type: DataType) -> None:
        """
        Execute a query by calling ask.main() directly.

        Args:
            question: The question to ask
            data_type: DataType specifying backend and source
        """
        self.config.validate_env(data_type.type)
        paths = self.config.get_paths_for_source(data_type.source)
        self._validate_backend_requirements(data_type, paths)

        # Set environment variables for ask.main()
        env = self._build_env(data_type, paths)
        # Call ask.main() directly instead of subprocess
        ask.main(question)

    def typed_query(
        self,
        question: str,
        data_type: DataType,
        *,
        model_name: Optional[str] = None,
        rerank: bool = False,
        use_llm: bool = True,
        save_artifacts: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """
        Execute the full Typed-RAG pipeline and return the aggregated answer.

        Args:
            question: User question.
            data_type: Backend/source selection.
            model_name: Model to use (e.g., gemini-2.0-flash-lite or meta-llama/Llama-3.2-3B-Instruct).
            rerank: Enable cross-encoder reranking.
            use_llm: Enable Gemini calls (falls back to heuristic otherwise).
            save_artifacts: Persist plan/evidence/final answer to disk.
            output_dir: Directory for persisted artifacts (defaults to ./output).
        """
        self.config.validate_env(data_type.type)
        paths = self.config.get_paths_for_source(data_type.source)
        self._validate_backend_requirements(data_type, paths)

        embedder = BGEEmbedder()
        vector_store = self._load_vector_store(data_type, paths, embedder)

        # Typed pipeline steps
        question_type = classify_question(question, use_llm=use_llm)
        plan = decompose_question(
            question,
            question_type,
            cache_dir=self.config.repo_root / "cache" / "decomposition",
        )

        orchestrator = RetrievalOrchestrator(
            embedder=embedder,
            vector_store=vector_store,
            vector_store_type=data_type.type,
            cache_dir=self.config.repo_root / "cache" / "evidence",
            rerank=rerank,
        )

        bundle = orchestrator.retrieve_evidence(
            plan,
            use_cache=True,
            final_top_k=self.config.top_k,
        )

        generator = TypedAnswerGenerator(
            model_name=model_name,
            cache_dir=self.config.repo_root / "cache" / "answers",
            use_llm=use_llm,
        )
        aspect_answers = generator.generate(plan, bundle)

        aggregator = TypedAnswerAggregator(
            model_name=model_name,
            cache_dir=self.config.repo_root / "cache" / "final_answers",
            use_llm=use_llm,
        )
        final_answer = aggregator.aggregate(plan, aspect_answers)

        if save_artifacts:
            target_dir = Path(output_dir) if output_dir else self.config.repo_root / "output"
            target_dir.mkdir(parents=True, exist_ok=True)
            orchestrator.save_plan_and_evidence(plan, bundle, target_dir)
            with open(target_dir / f"{plan.question_id}_final_answer.json", "w", encoding="utf-8") as f:
                json.dump(final_answer.to_dict(), f, indent=2)

        return final_answer


# ============================================================================
# Main API - Simple functions that match user's exact requirements
# ============================================================================

def create_index(data_type: DataType, rebuild: bool = False) -> None:
    """
    Build an index for the specified data type.

    Args:
        data_type: DataType instance specifying backend and source
        rebuild: Force rebuild even if index exists

    Example:
        from typed_rag.rag_system import DataType, create_index

        data_type = DataType('pinecone', 'own_docs')
        create_index(data_type)

        data_type = DataType('faiss', 'wikipedia')
        create_index(data_type, rebuild=True)
    """
    config = RAGConfig.default()
    builder = IndexBuilder(config)
    builder.build(data_type, force_rebuild=rebuild)


def ask_question(query: str, data_type: DataType) -> None:
    """
    Ask a question using the specified data type.

    Args:
        query: The question to ask
        data_type: DataType instance specifying backend and source

    Example:
        from typed_rag.rag_system import DataType, ask_question

        data_type = DataType('pinecone', 'own_docs')
        ask_question("What is Amazon's revenue?", data_type)

        data_type = DataType('faiss', 'wikipedia')
        ask_question("Explain quantum computing", data_type)
    """
    config = RAGConfig.default()
    query_engine = QueryEngine(config)
    query_engine.query(query, data_type)


def ask_typed_question(
    query: str,
    data_type: DataType,
    *,
    model_name: Optional[str] = None,
    rerank: bool = False,
    use_llm: bool = True,
    save_artifacts: bool = True,
    output_dir: Optional[Path] = None,
):
    """
    Run the full Typed-RAG pipeline and return the aggregated answer.

    Args:
        query: The question to ask.
        data_type: Backend/source configuration.
        model_name: Model to use (e.g., gemini-2.0-flash-lite or meta-llama/Llama-3.2-3B-Instruct).
        rerank: Enable cross-encoder reranking.
        use_llm: Use Gemini for generation/aggregation (fallback otherwise).
        save_artifacts: Persist plan/evidence/final JSON to disk.
        output_dir: Directory for saved artifacts (defaults to ./output).
    """
    config = RAGConfig.default()
    query_engine = QueryEngine(config)
    return query_engine.typed_query(
        query,
        data_type,
        model_name=model_name,
        rerank=rerank,
        use_llm=use_llm,
        save_artifacts=save_artifacts,
        output_dir=output_dir,
    )
