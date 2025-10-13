#!/usr/bin/env python3
"""
Build a light-weight BM25 index over chunk-level JSONL.

Each JSONL line is expected to be a chunk record with fields like:
  id, doc_id, title, text, section, chunk_idx, url, source, ...

We treat each chunk as a retrievable unit, lightly weight fields (title > section > text),
and persist a pickled structure for fast reload. Also supports an optional query mode.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from rank_bm25 import BM25Okapi


logger = structlog.get_logger()

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---- Defaults (tweakable via CLI) ----
DEFAULT_FIELD_WEIGHTS = {"title": 3, "section": 2, "text": 1}
DEFAULT_TOP_K = 10

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "with",
    "by",
    "as",
    "at",
    "be",
    "from",
    "that",
    "this",
    "it",
    "its",
    "into",
    "over",
    "under",
}


def normalize(text: str) -> str:
    text = text.replace("•", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, stopwords: set[str]) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if t not in stopwords]


def weighted_concat(obj: Dict[str, Any], field_weights: Dict[str, int]) -> str:
    parts: List[str] = []
    for field_name, weight in field_weights.items():
        value = obj.get(field_name)
        if value:
            s = normalize(str(value))
            parts.append((s + " ") * int(weight))
    return (" ".join(parts)).strip()


def load_chunks(jsonl_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                logger.warning("Skipping invalid JSON line", line_number=i, error=str(e))
                continue
            chunks.append(obj)
    return chunks


def build_bm25_from_chunks(
    chunk_records: List[Dict[str, Any]],
    field_weights: Dict[str, int],
    stopwords: Optional[set[str]] = None,
):
    if stopwords is None:
        stopwords = STOPWORDS

    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
    chunk_tokens: List[List[str]] = []
    chunk_meta: List[Dict[str, Any]] = []

    for i, obj in enumerate(chunk_records):
        # Normalize common schema variations: treat `chunk_text` as `text` if present
        if "text" not in obj and obj.get("chunk_text"):
            obj = {**obj, "text": obj["chunk_text"]}
        cid = str(obj.get("id", f"row_{i}"))
        text = weighted_concat(obj, field_weights)
        if not text:
            continue
        toks = tokenize(text, stopwords)
        if not toks:
            continue
        chunk_ids.append(cid)
        chunk_texts.append(text)
        chunk_tokens.append(toks)
        chunk_meta.append(obj)

    if not chunk_tokens:
        raise RuntimeError("No valid chunks to index after preprocessing")

    bm25 = BM25Okapi(chunk_tokens, k1=1.5, b=0.75)
    return bm25, chunk_ids, chunk_texts, chunk_tokens, chunk_meta


def save_index(
    out_path: Path,
    bm25,
    chunk_ids: List[str],
    chunk_texts: List[str],
    chunk_tokens: List[List[str]],
    chunk_meta: List[Dict[str, Any]],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "chunk_ids": chunk_ids,
                "chunk_texts": chunk_texts,
                "chunk_tokens": chunk_tokens,
                "chunk_meta": chunk_meta,
            },
            f,
        )
    logger.info("Saved BM25 artifacts", path=str(out_path), chunks=len(chunk_ids))


def load_index(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    bm25 = BM25Okapi(data["chunk_tokens"], k1=1.5, b=0.75)
    return bm25, data


def _highlight(snippet: str, query_terms: List[str], max_len: int = 220) -> str:
    s = normalize(snippet or "")
    qset = set(query_terms)

    def repl(m):
        w = m.group(0)
        return f"**{w}**" if w.lower() in qset else w

    marked = re.sub(r"[A-Za-z0-9]+", repl, s)
    return marked[:max_len] + ("…" if len(marked) > max_len else "")


def search(
    bm25,
    chunk_ids: List[str],
    chunk_meta: List[Dict[str, Any]],
    query: str,
    k: int = DEFAULT_TOP_K,
    filter_doc_id: Optional[str] = None,
    filter_source: Optional[str] = None,
    stopwords: Optional[set[str]] = None,
):
    if stopwords is None:
        stopwords = STOPWORDS
    q_tokens = tokenize(query, stopwords)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)

    mask = np.ones(len(scores), dtype=bool)
    if filter_doc_id is not None:
        mask &= np.array([m.get("doc_id") == filter_doc_id for m in chunk_meta], dtype=bool)
    if filter_source is not None:
        mask &= np.array([m.get("source") == filter_source for m in chunk_meta], dtype=bool)

    if not mask.all():
        filtered = np.where(mask)[0]
        if len(filtered) == 0:
            return []
        eligible = scores[filtered]
        k = min(k, len(filtered))
        top_rel = np.argpartition(eligible, -k)[-k:]
        top_rel = top_rel[np.argsort(eligible[top_rel])[::-1]]
        top_idx = filtered[top_rel]
    else:
        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    results = []
    for i in top_idx:
        meta = chunk_meta[i]
        preview_src = meta.get("text") or meta.get("chunk_text") or meta.get("title") or ""
        results.append(
            {
                "chunk_id": chunk_ids[i],
                "score": float(scores[i]),
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "section": meta.get("section"),
                "chunk_idx": meta.get("chunk_idx"),
                "url": meta.get("url"),
                "highlight": _highlight(preview_src, q_tokens),
                "raw": meta,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Build and optionally query a BM25 chunk index")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input chunks.jsonl path")
    parser.add_argument(
        "--out",
        dest="output",
        type=str,
        default=str(Path(__file__).parent.parent / "indexes" / "bm25_chunks.pkl"),
        help="Output pickle path for BM25 artifacts",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top K for sample query")
    parser.add_argument("--no_stopwords", action="store_true", help="Disable stopword removal")
    parser.add_argument("--query", type=str, default=None, help="Optional: run a test query after build")
    parser.add_argument("--filter_doc_id", type=str, default=None)
    parser.add_argument("--filter_source", type=str, default=None)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        logger.error("Input file does not exist", input=str(input_path))
        sys.exit(1)

    field_weights = DEFAULT_FIELD_WEIGHTS
    stopwords = set() if args.no_stopwords else STOPWORDS

    logger.info("Loading chunks", input=str(input_path))
    chunks = load_chunks(input_path)
    logger.info("Loaded chunks", count=len(chunks))
    if not chunks:
        logger.error("No chunks found in input")
        sys.exit(1)

    logger.info("Building BM25 index")
    bm25, chunk_ids, chunk_texts, chunk_tokens, chunk_meta = build_bm25_from_chunks(
        chunks, field_weights, stopwords
    )
    logger.info("BM25 built", chunks=len(chunk_ids))

    logger.info("Saving artifacts", path=str(output_path))
    save_index(output_path, bm25, chunk_ids, chunk_texts, chunk_tokens, chunk_meta)

    if args.query:
        logger.info("Running sample query", query=args.query, top_k=args.top_k)
        hits = search(
            bm25,
            chunk_ids,
            chunk_meta,
            query=args.query,
            k=args.top_k,
            filter_doc_id=args.filter_doc_id,
            filter_source=args.filter_source,
            stopwords=stopwords,
        )
        for h in hits:
            print(f"{h['chunk_id']}\t{h['score']:.3f}\t{h.get('title','')}\t{h['highlight']}")


if __name__ == "__main__":
    main()


