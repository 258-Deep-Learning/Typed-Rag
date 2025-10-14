#!/usr/bin/env python3
"""
Typed-RAG split CLI: build-only vs ask-only.

Usage examples:
  # Build indices (no queries)
  python rag_cli.py build --backend pinecone --source own_docs
  python rag_cli.py build --backend faiss --source own_docs --rebuild

  # Ask (no building)
  python rag_cli.py ask --backend pinecone --source own_docs "What is Amazon's revenue?"
  python rag_cli.py ask --backend faiss --source own_docs "Summarize the design."
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Use your exact modules
from typed_rag.scripts import ingest_own_docs as ingest
from typed_rag.scripts import build_pinecone as bp
from typed_rag.scripts import build_faiss as bf

# -------- defaults (match your current code/structure) --------
MY_DOCS_DIR = REPO_ROOT / "my-documents"
CHUNKS_JSONL = REPO_ROOT / "typed_rag" / "data" / "chunks.jsonl"
FAISS_DIR = REPO_ROOT / "typed_rag" / "indexes" / "faiss"

PINECONE_INDEX = "typedrag-own"
PINECONE_NAMESPACE = "own_docs"


# -------- helpers --------
def require_env(name: str) -> None:
    if not os.getenv(name):
        raise EnvironmentError(f"Environment variable {name} is not set")


def ensure_chunks_jsonl(docs_dir: Path, out_path: Path, rebuild: bool) -> None:
    """
    Create chunks.jsonl if missing, or when --rebuild is true.
    Otherwise, reuse existing chunks.jsonl to avoid re-ingest.
    """
    if out_path.exists() and not rebuild:
        print(f"✓ Using existing chunks file: {out_path}")
        return

    if not docs_dir.exists():
        raise FileNotFoundError(f"❌ Directory not found: {docs_dir}")

    print(f"📥 Ingesting documents from: {docs_dir}")
    chunks = ingest.ingest_directory(docs_dir)
    if not chunks:
        raise RuntimeError("❌ No chunks generated (documents may be empty or unsupported)")
    ingest.save_chunks_jsonl(chunks, out_path)
    print(f"✓ Saved chunks: {out_path} ({len(chunks)} records)")


def detect_default_backend() -> str:
    """Pick a sensible default backend for asking.
    Priority:
      1) VECTOR_STORE env if set to 'faiss' or 'pinecone'
      2) FAISS artifacts exist → 'faiss'
      3) PINECONE_API_KEY present → 'pinecone'
      4) fallback → 'faiss'
    """
    env_choice = os.getenv("VECTOR_STORE", "").lower()
    if env_choice in {"faiss", "pinecone"}:
        return env_choice
    index_files = {"index.faiss", "index.pkl"}
    if FAISS_DIR.exists() and index_files.issubset({p.name for p in FAISS_DIR.glob("*")}):
        return "faiss"
    if os.getenv("PINECONE_API_KEY"):
        return "pinecone"
    return "faiss"


# -------- class + API you requested --------
class DataType:
    """
    type: 'pinecone' | 'faiss'
    source: 'own_docs'
    (BM25 not wired here since that script wasn't provided in this drop.)
    """
    def __init__(self, type: str, source: str):
        self.type = type
        self.source = source


def create_index(data_type: DataType, docs_dir: Path = MY_DOCS_DIR, rebuild: bool = False) -> None:
    """
    Build the requested index. Never asks questions.
    - pinecone: uses bp.load_chunks + bp.upsert_chunks_to_pinecone
    - faiss: uses typed_rag.scripts.build_faiss main routine
    """
    if data_type.source != "own_docs":
        raise ValueError("Only source='own_docs' supported in this file.")

    # 1) Ensure chunks.jsonl
    ensure_chunks_jsonl(docs_dir, CHUNKS_JSONL, rebuild=rebuild)

    # 2) Build index
    if data_type.type == "pinecone":
        require_env("PINECONE_API_KEY")
        print("\n📦 Building Pinecone index...")
        recs = bp.load_chunks(CHUNKS_JSONL)
        if not recs:
            raise RuntimeError("❌ No records loaded from chunks.jsonl")

        embedder = bp.BGEEmbedder()
        store = bp.PineconeDenseStore(
            index_name=PINECONE_INDEX,
            namespace=PINECONE_NAMESPACE,
            dimension=384,
            metric="cosine",
            create_if_missing=True,
        )
        bp.upsert_chunks_to_pinecone(store, embedder, recs)
        print("✓ Pinecone build complete")

    elif data_type.type == "faiss":
        print("\n📦 Building FAISS index...")
        FAISS_DIR.mkdir(parents=True, exist_ok=True)

        # If already present and no --rebuild, reuse
        index_files = {"index.faiss", "index.pkl"}
        if not rebuild and index_files.issubset({p.name for p in FAISS_DIR.glob("*")}):
            print(f"✓ FAISS index already exists at {FAISS_DIR} (use --rebuild to force).")
            return

        # Call your script's main routine via args (ensures same behavior)
        import subprocess
        cmd = [
            sys.executable, "-m", "typed_rag.scripts.build_faiss",
            "--in", str(CHUNKS_JSONL),
            "--out_dir", str(FAISS_DIR),
        ]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
        print("✓ FAISS build complete")

    else:
        raise ValueError("Unsupported backend. Use 'pinecone' or 'faiss'.")


def ask_question(query: str, data_type: DataType) -> None:
    """
    Ask a question against an already-built index. Never builds.
    Uses your ask.py exactly as-is.
    """
    ask_py = REPO_ROOT / "ask.py"
    if not ask_py.exists():
        raise FileNotFoundError(f"❌ ask.py not found at {ask_py}")

    # Sanity checks based on backend
    env = os.environ.copy()

    # LLM key
    require_env("GOOGLE_API_KEY")

    if data_type.type == "pinecone":
        require_env("PINECONE_API_KEY")
        env["VECTOR_STORE"] = "pinecone"
        env["PINECONE_INDEX"] = PINECONE_INDEX
        env["PINECONE_NAMESPACE"] = PINECONE_NAMESPACE

    elif data_type.type == "faiss":
        # Make sure FAISS artifacts exist
        if not FAISS_DIR.exists():
            raise FileNotFoundError(
                f"❌ FAISS directory not found at {FAISS_DIR}. "
                "Run: python rag_cli.py build --backend faiss --source own_docs"
            )
        env["VECTOR_STORE"] = "faiss"
        env["FAISS_DIR"] = str(FAISS_DIR)

    else:
        raise ValueError("Unsupported backend. Use 'pinecone' or 'faiss'.")

    # Delegate to your ask.py so output stays consistent
    print("💬 Querying via ask.py ...\n")
    import subprocess
    res = subprocess.run([sys.executable, str(ask_py), query], env=env)
    if res.returncode != 0:
        raise RuntimeError("❌ Query failed")


# -------- small CLI --------
def main():
    p = argparse.ArgumentParser(prog="rag-cli", description="Build-only and ask-only flows for Typed-RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="Build an index (no querying)")
    b.add_argument("--backend", "-b", choices=["pinecone", "faiss"], required=False, default="faiss",
                   help="Backend to build (default: faiss)")
    b.add_argument("--source", "-s", choices=["own_docs"], required=False, default="own_docs",
                   help="Source corpus (default: own_docs)")
    b.add_argument("--docs_dir", default=str(MY_DOCS_DIR))
    b.add_argument("--rebuild", action="store_true", help="Force re-ingest/re-index")

    # ask
    a = sub.add_parser("ask", help="Ask a question (no building)")
    a.add_argument("--backend", "-b", choices=["pinecone", "faiss"], required=False,
                   help="Backend to query (default: auto-detect; faiss if built else pinecone)")
    a.add_argument("--source", "-s", choices=["own_docs"], required=False, default="own_docs",
                   help="Source corpus (default: own_docs)")
    a.add_argument("question")

    args = p.parse_args()
    # Choose backend defaults if not provided
    backend = getattr(args, "backend", None)
    if backend is None:
        backend = detect_default_backend()
    dt = DataType(backend, args.source)

    if args.cmd == "build":
        create_index(dt, docs_dir=Path(args.docs_dir), rebuild=args.rebuild)
    else:
        ask_question(args.question, dt)


if __name__ == "__main__":
    main()
