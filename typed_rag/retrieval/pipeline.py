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


# ---------- Dense (FAISS) ----------
class FAISSDenseStore:
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        dimension: int = 384,
        metric: str = "cosine",
        create_if_missing: bool = True,
    ):
        """A lightweight FAISS-based dense store.

        Persists two artifacts:
          - index_path: FAISS index (IndexFlatIP wrapped in IndexIDMap2)
          - meta_path: JSON mapping of int_id -> {"id": str_id, "metadata": {...}}
        """
        import os as _os
        import json as _json
        import faiss  # type: ignore

        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = dimension
        self.metric = metric

        self._faiss = faiss
        self._index = None
        self._meta: Dict[int, Dict[str, Any]] = {}

        idx_exists = _os.path.exists(index_path)
        meta_exists = _os.path.exists(meta_path)

        if idx_exists and meta_exists:
            self._index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = _json.load(f)
            # keys from JSON are strings; normalize to int keys
            self._meta = {int(k): v for k, v in self._meta.items()}
        elif create_if_missing:
            if metric == "cosine" or metric == "ip":
                base = faiss.IndexFlatIP(dimension)
            elif metric == "l2":
                base = faiss.IndexFlatL2(dimension)
            else:
                raise ValueError(f"Unsupported metric for FAISS: {metric}")
            self._index = faiss.IndexIDMap2(base)
            self._meta = {}
        else:
            raise FileNotFoundError(
                f"FAISS index or meta not found: index={index_path}, meta={meta_path}"
            )

    @staticmethod
    def _strings_to_int64_ids(strings: List[str]) -> List[int]:
        import hashlib
        ids: List[int] = []
        for s in strings:
            h = hashlib.sha1(s.encode("utf-8")).digest()  # 20 bytes
            # take first 8 bytes for a stable 64-bit id
            i = int.from_bytes(h[:8], byteorder="big", signed=False)
            # FAISS expects signed int64 ids; constrain to [0, 2^63-1]
            i &= (1 << 63) - 1
            ids.append(i)
        return ids

    @staticmethod
    def _ensure_float32(vecs: "np.ndarray") -> "np.ndarray":
        import numpy as np
        if vecs.dtype != np.float32:
            return vecs.astype("float32", copy=False)
        return vecs

    @staticmethod
    def _maybe_normalize_for_ip(vecs: "np.ndarray", metric: str) -> "np.ndarray":
        import numpy as np
        if metric in ("cosine", "ip"):
            # L2 normalize so IP == cosine
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            return (vecs / norms).astype("float32", copy=False)
        return vecs

    def upsert(
        self,
        ids: List[str],
        vectors: "np.ndarray",
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 10000,
        persist: bool = True,
    ) -> None:
        assert self._index is not None, "FAISS index is not initialized"
        faiss = self._faiss
        import numpy as np

        int_ids = self._strings_to_int64_ids(ids)
        vectors = self._maybe_normalize_for_ip(self._ensure_float32(vectors), self.metric)

        for i in range(0, len(int_ids), batch_size):
            batch_int = np.asarray(int_ids[i : i + batch_size], dtype=np.int64)
            batch_vecs = vectors[i : i + batch_size]
            self._index.add_with_ids(batch_vecs, batch_int)
            if metadatas is not None:
                for j, k in enumerate(batch_int):
                    self._meta[int(k)] = {
                        "id": ids[i + j],
                        "metadata": metadatas[i + j] if metadatas else {},
                    }

        if persist:
            faiss.write_index(self._index, self.index_path)
            import json as _json
            with open(self.meta_path, "w", encoding="utf-8") as f:
                _json.dump(self._meta, f)

    def query(
        self,
        query_vec: "np.ndarray",
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        assert self._index is not None, "FAISS index is not initialized"
        import numpy as np

        if query_vec.ndim == 2:
            q = query_vec[0]
        else:
            q = query_vec
        q = self._maybe_normalize_for_ip(self._ensure_float32(q.reshape(1, -1)), self.metric)
        scores, ids = self._index.search(q, top_k)

        out: List[Dict[str, Any]] = []
        for score, int_id in zip(scores[0].tolist(), ids[0].tolist()):
            if int_id == -1:
                continue
            meta_rec = self._meta.get(int_id, {})
            str_id = meta_rec.get("id", str(int_id))
            md = meta_rec.get("metadata", {})
            rec = {"id": str_id, "score": float(score), "metadata": md}
            out.append(rec)

        if metadata_filter:
            # Simple shallow filter: all keys in filter must match exactly in metadata
            def _ok(m: Dict[str, Any]) -> bool:
                for k, v in metadata_filter.items():
                    if m.get(k) != v:
                        return False
                return True
            out = [r for r in out if _ok(r.get("metadata", {}))]

        return out


# ---------- LangChain Embeddings adapter for BGE ----------
from langchain_core.embeddings import Embeddings  # type: ignore


class LCBGEEmbeddings(Embeddings):
    """Adapter to use our BGEEmbedder with LangChain's FAISS.

    Implements the minimal interface of langchain_core.embeddings.Embeddings.
    """

    def __init__(self, embedder: Optional[BGEEmbedder] = None, device: Optional[str] = None):
        if embedder is None:
            embedder = BGEEmbedder(device=device)
        self._embedder = embedder

    # LangChain interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        vecs = self._embedder.encode_passages(texts)
        return np.asarray(vecs, dtype="float32").tolist()

    def embed_query(self, text: str) -> List[float]:
        import numpy as np
        vec = self._embedder.encode_queries([text])
        return np.asarray(vec[0], dtype="float32").tolist()

    # Do not implement __call__; ensure LC treats this as an Embeddings object

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
                    text=md.get("text") or r.get("text"),  # ensure downstream access
                    score=float(r["score"]),
                )
            )
        docs.sort(key=lambda d: (-d.score, d.id))
        # Optional: latency metric if you want to log it here
        _latency_ms = int((time.time() - t0) * 1000)
        return docs[:k]


# ---------- LangChain FAISS adapter ----------
class LangChainFAISSAdapter:
    """Wraps a LangChain FAISS vector store to match the query interface."""

    def __init__(self, lc_store):
        self.store = lc_store

    def query(
        self,
        query_vec: "np.ndarray",
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        if query_vec.ndim == 2:
            vec = query_vec[0]
        else:
            vec = query_vec
        vec = np.asarray(vec, dtype="float32")
        vec_list = vec.tolist()
        if hasattr(self.store, "similarity_search_by_vector_with_relevance_scores"):
            results = self.store.similarity_search_by_vector_with_relevance_scores(vec_list, k=top_k)
        elif hasattr(self.store, "similarity_search_with_score_by_vector"):
            results = self.store.similarity_search_with_score_by_vector(vec_list, k=top_k)
        elif hasattr(self.store, "similarity_search_by_vector"):
            docs = self.store.similarity_search_by_vector(vec_list, k=top_k)
            results = [(doc, 1.0) for doc in docs]
        else:
            raise AttributeError("FAISS store does not expose a compatible similarity search method")
        out: List[Dict[str, Any]] = []
        for doc, score in results:
            md = dict(doc.metadata or {})
            text = doc.page_content or ""
            if "text" not in md:
                md["text"] = text
            record = {
                "id": md.get("id") or md.get("doc_id") or "",
                "score": float(score),
                "metadata": md,
                "text": text,
            }
            out.append(record)

        if metadata_filter:
            def _matches(meta: Dict[str, Any]) -> bool:
                for k, v in metadata_filter.items():
                    if meta.get(k) != v:
                        return False
                return True

            out = [rec for rec in out if _matches(rec.get("metadata", {}))]

        return out[:top_k]


def load_faiss_adapter(
    faiss_dir: str,
    embedder: Optional[BGEEmbedder] = None,
) -> LangChainFAISSAdapter:
    """Load a FAISS vector store saved via LangChain and wrap it for retrieval."""
    from langchain_community.vectorstores import FAISS as LCFAISS  # type: ignore

    embeddings = LCBGEEmbeddings(embedder=embedder)
    lc_store = LCFAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return LangChainFAISSAdapter(lc_store)
