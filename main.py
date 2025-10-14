

"""

CLI to run two flows:

first (Own Docs → Pinecone):
    - Ingest documents from my-documents/ → create typed_rag/data/chunks.jsonl
    - Build Pinecone index from chunks.jsonl
    - Query by calling ask.py (which prints retrieved chunks and the final answer)

second (Passages → BM25):
    - Use typed_rag/data/passages.jsonl
    - Build BM25 index
    - For each query, print top_3 chunks (score + snippet)
    - Then call Gemini directly here using those BM25 results as context

Common printing for queries:
    - rank, score, title/doc, chunk_idx, url
    - snippet/highlight
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Make package imports work when running from repo root
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Import script helpers
from typed_rag.scripts import ingest_own_docs as ingest
from typed_rag.scripts import build_pinecone as bp
from typed_rag.scripts import build_faiss as bf
from typed_rag.scripts import build_bm25 as bb


# ---------------------------- Config Defaults ----------------------------
MY_DOCS_DIR = REPO_ROOT / "my-documents"
CHUNKS_JSONL = REPO_ROOT / "typed_rag" / "data" / "chunks.jsonl"
PASSAGES_JSONL = REPO_ROOT / "typed_rag" / "data" / "passages.jsonl"
BM25_PKL = REPO_ROOT / "typed_rag" / "indexes" / "bm25_chunks.pkl"
FAISS_DIR = REPO_ROOT / "typed_rag" / "indexes" / "faiss"

PINECONE_INDEX = "typedrag-own"
PINECONE_NAMESPACE = "own_docs"

# User requested top_k=3 for BM25 → LLM
DEFAULT_TOP_K = 3


# ---------------------------- Helpers ----------------------------
def require_env(var_name: str) -> bool:
    if not os.getenv(var_name):
        print(f"❌ Environment variable {var_name} is not set")
        return False
    return True


def print_bm25_hits(hits: List[Dict[str, Any]]) -> None:
    if not hits:
        print("No results.")
        return
    print("📄 Retrieved Chunks:")
    print("-" * 80)
    for i, h in enumerate(hits, 1):
        title = h.get("title") or h.get("raw", {}).get("title") or "Untitled"
        doc_id = h.get("doc_id") or h.get("raw", {}).get("doc_id")
        chunk_idx = h.get("chunk_idx") or h.get("raw", {}).get("chunk_idx")
        url = h.get("url") or h.get("raw", {}).get("url") or ""
        score = float(h.get("score", 0.0))
        meta_line = f"[{i}] score={score:.4f} | title={title} | doc={doc_id} | chunk_idx={chunk_idx}"
        if url:
            meta_line += f" | url={url}"
        print(meta_line)
        snippet = h.get("highlight") or h.get("raw", {}).get("text") or ""
        snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
        print(f"    {snippet}")
    print("-" * 80)


def gemini_answer_from_passages(question: str, passages: List[str], model_name: str = "gemini-2.5-flash") -> Optional[str]:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        print(f"❌ google-generativeai not available: {e}")
        return None

    if not require_env("GOOGLE_API_KEY"):
        return None

    context = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    prompt = (
        "Answer the question based on the following passages.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (be concise and cite passage numbers):"
    )

    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None)
    except Exception as e:
        print(f"❌ Gemini call failed: {e}")
        return None


# ---------------------------- Flows ----------------------------
def flow_first_ingest_pinecone() -> None:
    print("\n=== First Flow: Own Docs → Pinecone ===\n")

    # 1) Ingest and chunk
    root = MY_DOCS_DIR
    out_path = CHUNKS_JSONL
    print(f"📥 Ingesting from: {root}")
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        return
    chunks = ingest.ingest_directory(root)
    if not chunks:
        print("❌ No chunks generated (documents may be empty or unsupported)")
        return
    ingest.save_chunks_jsonl(chunks, out_path)
    print(f"✓ Saved chunks: {out_path} ({len(chunks)} records)")

    # 2) Build Pinecone
    if not require_env("PINECONE_API_KEY"):
        return
    print("\n📦 Building Pinecone index (this may take a while)...")
    recs = bp.load_chunks(out_path)
    if not recs:
        print("❌ No records loaded from chunks.jsonl")
        return
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

    # 3) Query loop via ask.py (which handles retrieval + LLM)
    print("\nYou can now query. Type an empty line to return to menu.")
    ask_py = REPO_ROOT / "ask.py"
    if not ask_py.exists():
        print(f"❌ ask.py not found at {ask_py}")
        return
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        # Delegate to ask.py so output format stays consistent for Pinecone flow
        print()
        os.system(f"{sys.executable} {ask_py} \"{q.replace('\\', r'\\').replace('"', r'\\"')}\"")
        print()


def flow_faiss_from_own_docs() -> None:
    print("\n=== FAISS Flow: Own Docs → FAISS (ingest + index + ask) ===\n")

    # 1) Ingest and chunk
    root = MY_DOCS_DIR
    out_path = CHUNKS_JSONL
    print(f"📥 Ingesting from: {root}")
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        return
    chunks = ingest.ingest_directory(root)
    if not chunks:
        print("❌ No chunks generated (documents may be empty or unsupported)")
        return
    ingest.save_chunks_jsonl(chunks, out_path)
    print(f"✓ Saved chunks: {out_path} ({len(chunks)} records)")

    # 2) Build FAISS
    print("\n📦 Building FAISS index...")
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    bf.load_chunks  # type: ignore  # ensure import not optimized away
    cmd = [
        sys.executable,
        "-m",
        "typed_rag.scripts.build_faiss",
        "--in",
        str(out_path),
        "--out_dir",
        str(FAISS_DIR),
    ]
    try:
        import subprocess
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
        print("✓ FAISS build complete")
    except Exception as e:
        print(f"❌ FAISS build failed: {e}")
        return

    # 3) Query loop via ask.py with VECTOR_STORE=faiss
    print("\nYou can now query. Type an empty line to return to menu.")
    ask_py = REPO_ROOT / "ask.py"
    if not ask_py.exists():
        print(f"❌ ask.py not found at {ask_py}")
        return
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print()
        env = os.environ.copy()
        env["VECTOR_STORE"] = "faiss"
        env["FAISS_DIR"] = str(FAISS_DIR)
        import subprocess
        res = subprocess.run([sys.executable, str(ask_py), q], env=env)
        if res.returncode != 0:
            print("❌ Query failed")
        print()


def flow_second_passages_bm25(top_k: int = DEFAULT_TOP_K) -> None:
    print("\n=== Second Flow: Passages → BM25 ===\n")

    # 1) Build BM25 from passages.jsonl
    in_path = PASSAGES_JSONL
    out_pkl = BM25_PKL
    print(f"📚 Loading passages: {in_path}")
    if not in_path.exists():
        print(f"❌ passages.jsonl not found at {in_path}")
        return
    chunks = bb.load_chunks(in_path)
    if not chunks:
        print("❌ No chunks in passages.jsonl")
        return
    bm25, chunk_ids, chunk_texts, chunk_tokens, chunk_meta = bb.build_bm25_from_chunks(
        chunks, bb.DEFAULT_FIELD_WEIGHTS
    )
    bb.save_index(out_pkl, bm25, chunk_ids, chunk_texts, chunk_tokens, chunk_meta)
    print(f"✓ BM25 artifacts saved: {out_pkl}")

    # 2) Query loop (BM25 → print top_k → LLM with BM25 context)
    if not require_env("GOOGLE_API_KEY"):
        return
    print("\nYou can now query. Type an empty line to return to menu.")
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        hits = bb.search(
            bm25=bm25,
            chunk_ids=chunk_ids,
            chunk_meta=chunk_meta,
            query=q,
            k=top_k,
        )
        print()
        print_bm25_hits(hits)

        # Prepare passages for LLM
        passages: List[str] = []
        for h in hits:
            raw = h.get("raw", {})
            text = raw.get("text") or raw.get("chunk_text") or ""
            if text:
                passages.append(text)
        if not passages:
            print("No textual passages available to send to LLM.")
            continue

        print("💡 Generating answer with Gemini...")
        answer = gemini_answer_from_passages(q, passages)
        if answer:
            print("=" * 80)
            print(answer)
            print("=" * 80)
        else:
            print("❌ Failed to get answer from Gemini.")
        print()


# ---------------------------- CLI ----------------------------
def main() -> None:
    while True:
        print("\n==== Typed-RAG CLI ====")
        print("1) Own docs → Pinecone (ingest + index + ask)")
        print("2) Passages → BM25 (build + query + Gemini)")
        print("3) Own docs → FAISS (ingest + index + ask)")
        print("4) Exit")
        try:
            choice = input("Select an option [1-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice == "1":
            flow_first_ingest_pinecone()
        elif choice == "2":
            flow_second_passages_bm25(top_k=DEFAULT_TOP_K)
        elif choice == "3":
            flow_faiss_from_own_docs()
        elif choice == "4":
            print("Goodbye!")
            return
        else:
            print("Please choose 1, 2 or 3.")


if __name__ == "__main__":
    main()

