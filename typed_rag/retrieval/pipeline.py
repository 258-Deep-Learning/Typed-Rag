# retrieval/pipeline.py
from __future__ import annotations

import os
import time
import math
import json
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------- Embedding (BGE small v1.5) ----------
# Prefer sentence-transformers if available, else fall back to HF Transformers.
class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._use_st = False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(model_name, device=device)
            self._use_st = True
        except Exception:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.torch = torch
            if device:
                self.model.to(device)

    @staticmethod
    def _l2_normalize(vecs):
        import numpy as np
        vecs = vecs.astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def encode_passages(self, texts: List[str], batch_size: int = 64) -> "np.ndarray":
        # For BGE, no special prefix is needed for passages in v1.5
        return self._encode(texts, batch_size)

    def encode_queries(self, texts: List[str], batch_size: int = 64) -> "np.ndarray":
        # For BGE, query benefits from the "query: " prefix (lightly improves alignment)
        texts = [f"query: {t}" for t in texts]
        return self._encode(texts, batch_size)

    def _encode(self, texts: List[str], batch_size: int) -> "np.ndarray":
        import numpy as np
        if self._use_st:
            # sentence-transformers handles batching internally
            vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        # transformers fallback (mean pooling)
        tok = self.tok
        model = self.model
        torch = self.torch
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                enc = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                if self.device:
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                out = model(**enc)
                # mean pool
                last_hidden = out.last_hidden_state  # [B, T, H]
                attn_mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                summed = (last_hidden * attn_mask).sum(dim=1)
                counts = attn_mask.sum(dim=1).clamp(min=1)
                emb = (summed / counts).cpu().numpy()
                all_vecs.append(emb)
        vecs = np.vstack(all_vecs)
        return self._l2_normalize(vecs)


# ---------- Dense (Pinecone) ----------
class PineconeDenseStore:
    def __init__(
        self,
        index_name: str,
        namespace: str = "own_docs",
        dimension: int = 384,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        create_if_missing: bool = True,
    ):
        from pinecone import Pinecone, ServerlessSpec  # type: ignore

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric

        # Create index if needed (serverless)
        if create_if_missing and index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(index_name)

    def upsert(
        self,
        ids: List[str],
        vectors: "np.ndarray",
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 200,
    ) -> None:
        assert len(ids) == len(vectors), "ids and vectors length mismatch"
        if metadatas is not None:
            assert len(metadatas) == len(ids), "metadatas length mismatch"
        import numpy as np
        # ensure float32 + (optionally) normalized
        vectors = vectors.astype("float32", copy=False)
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vecs = vectors[i : i + batch_size]
            if metadatas is None:
                upsert_batch = list(zip(batch_ids, batch_vecs))
            else:
                upsert_batch = [
                    {"id": _id, "values": _vec.tolist(), "metadata": metadatas[j]}
                    for j, (_id, _vec) in enumerate(zip(batch_ids, batch_vecs))
                ]
            self.index.upsert(vectors=upsert_batch, namespace=self.namespace)

    def query(
        self,
        query_vec: "np.ndarray",
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        query_vec = query_vec.astype("float32", copy=False).tolist()
        res = self.index.query(
            namespace=self.namespace,
            vector=query_vec,
            top_k=top_k,
            filter=metadata_filter,
            include_values=include_values,
            include_metadata=True,
        )
        out = []
        for m in res.matches or []:
            out.append(
                {
                    "id": m.id,
                    "score": float(m.score),
                    "metadata": dict(m.metadata or {}),
                }
            )
        return out


# ---------- Lexical (BM25 / Pyserini) ----------
class BM25Lexical:
    """Wrap Pyserini SimpleSearcher. Assumes you indexed JSON docs with fields:
       id, text, and stored raw JSON to recover metadata (title,url,doc_id,section,chunk_idx...)."""

    def __init__(self, index_dir: str, language: str = "en"):
        from pyserini.search import SimpleSearcher  # type: ignore
        self.searcher = SimpleSearcher(index_dir)
        # standard BM25 defaults work well; tweak if you like:
        self.searcher.set_bm25(k1=0.9, b=0.4)
        self.language = language

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        hits = self.searcher.search(query, k=top_k)
        results = []
        for h in hits:
            # raw is the original json doc stored at indexing time
            raw = h.raw
            try:
                meta = json.loads(raw)
            except Exception:
                meta = {"text": raw}
            results.append(
                {
                    "id": meta.get("id", h.docid),
                    "score": float(h.score),
                    "metadata": {k: v for k, v in meta.items() if k != "text"},
                    "text": meta.get("text", None),
                }
            )
        return results


# ---------- Fusion ----------
def _zscore(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mu = sum(scores) / len(scores)
    var = sum((s - mu) ** 2 for s in scores) / max(len(scores) - 1, 1)
    std = math.sqrt(var) or 1.0
    return [(s - mu) / std for s in scores]


def hybrid_fuse(
    dense: List[Dict[str, Any]],
    lex: List[Dict[str, Any]],
    final_k: int = 20,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[Dict[str, Any]]:
    """Z-score fusion with stable tie-break by id. Returns unique ids."""
    # Normalize score scales
    d_scores = _zscore([d["score"] for d in dense]) if dense else []
    l_scores = _zscore([l["score"] for l in lex]) if lex else []

    # Map id -> fused score + carry metadata/text (prefer richer record if available)
    fused: Dict[str, Dict[str, Any]] = {}

    for (d, z) in zip(dense, d_scores):
        rec = fused.get(d["id"], {"id": d["id"], "metadata": d.get("metadata", {}), "text": d.get("text")})
        rec["dense_z"] = max(z, rec.get("dense_z", float("-inf")))
        fused[d["id"]] = rec

    for (l, z) in zip(lex, l_scores):
        rec = fused.get(l["id"], {"id": l["id"], "metadata": l.get("metadata", {}), "text": l.get("text")})
        rec["lex_z"] = max(z, rec.get("lex_z", float("-inf")))
        # Prefer text/metadata if lex carries the raw chunk text
        if l.get("text") and not rec.get("text"):
            rec["text"] = l["text"]
        # merge metadata shallowly
        if l.get("metadata"):
            rec["metadata"] = {**rec.get("metadata", {}), **l["metadata"]}
        fused[l["id"]] = rec

    # Compute fused score
    for rec in fused.values():
        dz = rec.get("dense_z", float("-inf"))
        lz = rec.get("lex_z", float("-inf"))
        dz = dz if math.isfinite(dz) else -10.0
        lz = lz if math.isfinite(lz) else -10.0
        rec["score"] = alpha * dz + beta * lz

    # Stable sort: by fused score desc, then id asc
    ordered = sorted(fused.values(), key=lambda r: (-r["score"], r["id"]))
    return ordered[:final_k]


def rrf_fuse(
    dense: List[Dict[str, Any]],
    lex: List[Dict[str, Any]],
    final_k: int = 20,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion as a fallback."""
    ranks: Dict[str, float] = {}
    def add(listing: List[Dict[str, Any]]):
        for rank, rec in enumerate(listing, start=1):
            ranks[rec["id"]] = ranks.get(rec["id"], 0.0) + 1.0 / (k + rank)
    add(dense)
    add(lex)

    # Carry metadata/text from the best available source
    merged: Dict[str, Dict[str, Any]] = {}
    for lst in (dense, lex):
        for rec in lst:
            keep = merged.get(rec["id"], {"id": rec["id"], "metadata": {}, "text": None})
            keep["metadata"] = {**keep["metadata"], **rec.get("metadata", {})}
            keep["text"] = keep["text"] or rec.get("text")
            merged[rec["id"]] = keep

    out = []
    for _id, score in ranks.items():
        rec = merged[_id]
        rec["score"] = score
        out.append(rec)
    out.sort(key=lambda r: (-r["score"], r["id"]))
    return out[:final_k]


# ---------- Public types ----------
@dataclass
class Doc:
    id: str
    title: Optional[str]
    url: Optional[str]
    doc_id: Optional[str]
    section: Optional[str]
    chunk_idx: Optional[int]
    text: Optional[str]
    score: float


# ---------- High-level Retriever ----------
class Retriever:
    def __init__(
        self,
        pinecone_index: str,
        pinecone_namespace: str,
        bm25_index_dir: str,
        device: Optional[str] = None,
    ):
        self.embedder = BGEEmbedder(device=device)
        self.dense = PineconeDenseStore(index_name=pinecone_index, namespace=pinecone_namespace)
        self.lex = BM25Lexical(index_dir=bm25_index_dir)

    # --- Dense (Pinecone) ---
    def dense_search(self, q: str, top_k: int = 50, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        qv = self.embedder.encode_queries([q])
        return self.dense.query(qv, top_k=top_k, metadata_filter=metadata_filter)

    # --- Lexical (BM25)
    def lex_search(self, q: str, top_k: int = 50) -> List[Dict[str, Any]]:
        return self.lex.search(q, top_k=top_k)

    # --- Unified retrieve
    def retrieve(
        self,
        query: str,
        k: int = 20,
        mode: str = "hybrid",  # "dense" | "lex" | "hybrid"
        metadata_filter: Optional[Dict[str, Any]] = None,
        fuse: str = "zscore",  # "zscore" | "rrf"
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> List[Doc]:
        t0 = time.time()
        dense_res: List[Dict[str, Any]] = []
        lex_res: List[Dict[str, Any]] = []

        if mode in ("dense", "hybrid"):
            dense_res = self.dense_search(query, top_k=max(k * 3, 50), metadata_filter=metadata_filter)
        if mode in ("lex", "hybrid"):
            lex_res = self.lex_search(query, top_k=max(k * 3, 50))

        if mode == "dense":
            fused = dense_res[:k]
        elif mode == "lex":
            fused = lex_res[:k]
        else:
            if fuse == "rrf":
                fused = rrf_fuse(dense_res, lex_res, final_k=k)
            else:
                fused = hybrid_fuse(dense_res, lex_res, final_k=k, alpha=alpha, beta=beta)

        # Build Doc objects; ensure deterministic secondary ordering by id
        docs: List[Doc] = []
        for r in fused:
            md = r.get("metadata", {}) or {}
            docs.append(
                Doc(
                    id=r["id"],
                    title=md.get("title"),
                    url=md.get("url"),
                    doc_id=md.get("doc_id"),
                    section=md.get("section"),
                    chunk_idx=md.get("chunk_idx"),
                    text=r.get("text"),  # may be None if coming from dense-only
                    score=float(r["score"]),
                )
            )
        docs.sort(key=lambda d: (-d.score, d.id))
        # Optional: latency metric if you want to log it here
        _latency_ms = int((time.time() - t0) * 1000)
        return docs[:k]

