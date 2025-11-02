Awesome—here’s a concrete, coder-agent-ready **Weekend 1 (on your own documents) plan** that swaps FAISS for **Pinecone** while keeping parity with the TYPED-RAG pipeline and the acceptance bars from your proposal.

---



# Weekend 1 — Own Docs + Pinecone + Baselines

## A) Repo layout & configs (keep as in the proposal)

* Use the same folders so Phase 2–3 plug in with zero churn:

  ```
  typed_rag/
    data/            # raw + processed
    retrieval/       # bm25, pinecone client, hybrid
    scripts/         # ingestion, indexing, baselines
    configs/         # yaml/tyro
    runs/            # outputs
  ```

  This mirrors your proposal’s structure so later steps (decompose/rerank/eval) don’t change. 

**Why:** We keep the retrieval-first contract that the paper/pipeline assumes. 

---

## B) Dependencies (Python 3.11)

* `pip install pyserini pinecone-client transformers sentencepiece datasets faiss-cpu tyro typer structlog`
* Model: **BAAI/bge-small-en-v1.5** (384-dim; normalize embeddings).

**Why:** Your proposal already standardizes chunking + BM25 + embeddings + baselines; we just swap FAISS→Pinecone. 

---

## C) Ingest your documents → chunks.jsonl

**Goal:** parse PDFs/DOCX/MD/HTML, make 200-token chunks with 60-token stride, preserve titles/URLs/paths.
**Script:** `scripts/ingest_own_docs.py`

1. Crawl your source folder(s) → extract clean text per file.
2. Chunk with: `chunk_tokens=200`, `stride_tokens=60`.
3. Emit **JSONL** with one record per chunk:

```json
{
  "id": "doc123::chunk_0007",
  "doc_id": "doc123",
  "title": "Kubernetes Runbook",
  "url": "file:///docs/runbook.pdf#page=3",
  "section": "Rolling updates",
  "chunk_idx": 7,
  "text": "...",
  "token_len": 198,
  "source": "internal"
}
```

**Why:** Same chunk size/stride and “keep titles + URLs” guidance from the plan—this preserves later reranking and citations.  

---

## D) Build BM25 (lexical) index over chunks

**Script:** `scripts/build_bm25.py --in data/own_docs/chunks.jsonl --index indexes/lucene_own`

* Use **Pyserini** (Lucene).
* Index fields: `text` (analyzed), store `id,title,url,doc_id,section,chunk_idx` as metadata.
  **Why:** Hybrid > dense-only for early precision; and your proposal already expects a BM25 component. 

---

## E) Pinecone vector store (dense)

**Script:** `scripts/build_pinecone.py --in data/own_docs/chunks.jsonl --index typedrag-own --namespace own_docs`

1. **Create index**

   * name: `typedrag-own`
   * dimension: **384**
   * metric: **cosine**
   * (serverless/project/region per your account)
2. **Embed** each chunk with `bge-small-en-v1.5` → **L2-normalize**.
3. **Upsert** in batches (e.g., 200–500 vectors):

   ```
   (id=chunk["id"], vector=[...384...],
    metadata={
      "title":..., "url":..., "doc_id":..., "section":...,
      "chunk_idx":..., "token_len":..., "source":"internal"
    })
   ```
4. **Determinism:** store a `hash(text)` and the embedding checksum in metadata for auditability.

**Why:** We’re replacing FAISS from the original Weekend-1 with Pinecone while keeping the same retrieval contract. 

---

## F) Retrieval API (drop-in) — `retrieval/pipeline.py`

We keep your original interface so later phases don’t change:

```python
def retrieve(query: str, k: int = 20, mode: str = "hybrid") -> list[Doc]:
    # Doc = {id, title, url, chunk_text, score}
```

* `dense_search(q, top_k, filters=None)` → Pinecone `query` (namespace=own_docs, filter=metadata).
* `lex_search(q, top_k, filters=None)` → Pyserini BM25.
* `hybrid_fuse(dense, lex, k)` → **z-score** fusion (or RRF).
* **Return** fused top-k with **deterministic ordering** (tie-break by `id`).

**Why:** This interface is exactly what your proposal prescribes; keeping it stable protects all downstream code and acceptance checks.  

**Rationale from the paper:** Retrieval quality is the bottleneck; typed decomposition later reduces noise, but we first need a clean, reproducible retrieval layer. 

---

## G) Dev set for your own docs (100 items)

**Script:** `scripts/make_own_dev.py --root data/own_docs/ --out data/own_docs.dev100.jsonl`

* Auto-derive **non-factoid** questions from headings/sections (Instruction/Comparison/Reason/etc.).
* Predict a **type label** (light prompt + simple patterns) and save it; that matches what weekend-2 expects. 
* (Optional) For retrieval sanity, record **weak pseudo-references** (top titles/sections) for spot checks.

**Why:** Your original plan uses a **100-item stratified dev slice**—we recreate that idea for *your* corpus. 

---

## H) Baselines & logging

**LLM-only:** `scripts/run_llm_only.py --in data/own_docs.dev100.jsonl --out runs/llm_only.jsonl`
**Vanilla RAG:** `scripts/run_rag_baseline.py --in data/own_docs.dev100.jsonl --retr indexes:pinecone --out runs/rag.jsonl`

Both scripts must:

* Log per-question JSONL: `{question_id, prompt, passages[], answer, latency_ms, seed}`
* Include passage **titles/URLs** in the logs.
* Enforce seed & deterministic fusion ordering.

**Why:** Same baselines and reproducibility bars as your proposal (top-1 latency, determinism, no manual fixes).  

---

## I) Acceptance for Weekend-1 (own docs)

* `retrieve()` returns stable top-20 under a fixed **seed**. 
* Baselines complete on the **100-item** dev set end-to-end. 
* **Median top-1 latency ≤ 2s** per query on a laptop. 
* Every kept passage has **title + URL/path**. 

**Pitfalls to guard:** dedupe boilerplate, keep titles, avoid huge chunks—NFQA quality drops with long/noisy passages (this is *exactly* why typed decomposition helps later).  

---

## J) Pinecone specifics for the agent

* **Index:** `typedrag-own` (cosine, dim=384), **namespace:** `own_docs`.
* **Upsert batching:** 200–500 vectors; retry with exponential backoff.
* **Filters:** map typed retrieval needs later (e.g., allow `{"section": {"$eq": "Design"}}` or `{"doc_id": {"$in": [...]}}`).
* **Determinism:** persist the **exact Pinecone hits** (ids + scores) for each query into `runs/…jsonl` so you can replay experiments.
* **Near-dupe control:** keep `doc_id+chunk_idx` uniqueness in the final top-k to reduce repeated snippets.

---

## K) CLI targets (Typer/Makefile)

```
# 1) Ingest + chunk
python scripts/ingest_own_docs.py --root /path/to/corpus --out data/own_docs/chunks.jsonl --chunk_tokens 200 --stride_tokens 60

# 2) BM25
python scripts/build_bm25.py --in data/own_docs/chunks.jsonl --index indexes/lucene_own

# 3) Pinecone
python scripts/build_pinecone.py --in data/own_docs/chunks.jsonl --index typedrag-own --namespace own_docs

# 4) Dev set
python scripts/make_own_dev.py --root data/own_docs --out data/own_docs.dev100.jsonl

# 5) Baselines
python scripts/run_llm_only.py --in data/own_docs.dev100.jsonl --out runs/llm_only.jsonl
python scripts/run_rag_baseline.py --in data/own_docs.dev100.jsonl --retr indexes:pinecone --out runs/rag.jsonl
```

These mirror the style of your “concrete commands & configs,” just pointed at your corpus and Pinecone instead of Wikipedia/FAISS. 

---

## L) Hand-off notes to protect Weekends 2–3

* Keep the `retrieve()` contract stable and log everything; Weekend-2 will swap in **typed sub-queries + reranker** against the *same* backends (BM25 + Pinecone hybrid). 
* Preserve metadata (`title,url,doc_id,section,chunk_idx`) so typed aggregation can cite properly in Weekend-3. 
* Remember: typed RAG wins by **reducing retrieval noise** via per-type decomposition; you’re laying that foundation now. 

---

## M) What the coder agent should implement (checklist)

* [ ] `scripts/ingest_own_docs.py` (parse → chunk → `chunks.jsonl`) — 200/60, keep titles/URLs. 
* [ ] `scripts/build_bm25.py` (Pyserini index). 
* [ ] `scripts/build_pinecone.py` (embed → normalize → upsert).
* [ ] `retrieval/pipeline.py` (`dense_search`, `lex_search`, `hybrid_fuse`, `retrieve`). 
* [ ] `scripts/make_own_dev.py` (100 questions + type guess). 
* [ ] `scripts/run_llm_only.py` and `scripts/run_rag_baseline.py` (logging, latency stats). 
* [ ] **README** with exact commands and acceptance checks. 

---

If you want, I can also draft the tiny Pinecone client wrapper (`PineconeDenseStore`) and a reference `hybrid_fuse` (z-score + stable tie-break) so your agent can copy-paste into `retrieval/pipeline.py`.
Awesome — here’s a copy-pasteable reference you can drop into `retrieval/pipeline.py`. It gives you:

* `PineconeDenseStore` (create/query/upsert to Pinecone)
* A minimal BM25 searcher with Pyserini
* `hybrid_fuse` (z-score with stable tie-break) and `rrf_fuse`
* A `retrieve()` function that returns a unified, deterministic top-K

> **Assumptions**
>
> * Python 3.11
> * Installed: `pinecone-client`, `pyserini`, `transformers`, `torch`, `sentencepiece`, (optional) `sentence-transformers`
> * Env var: `PINECONE_API_KEY` is set
> * Pinecone: serverless index (cosine, dim 384) for `BAAI/bge-small-en-v1.5`
> * Your chunks were ingested with metadata: `title,url,doc_id,section,chunk_idx,token_len,source`

---

### `retrieval/pipeline.py` (drop this file in)

```python
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
```

---

## How your coder agent should wire this up

1. **Init once** (e.g., in your baseline runner):

```python
retr = Retriever(
    pinecone_index="typedrag-own",
    pinecone_namespace="own_docs",
    bm25_index_dir="indexes/lucene_own",
    device=None,  # or "cuda"
)
```

2. **Call from your baseline** (Vanilla RAG):

```python
topk = retr.retrieve(
    query=user_question,
    k=5,
    mode="hybrid",          # keep hybrid for stronger early precision
    metadata_filter=None,   # add filters later in Typed-RAG
    fuse="zscore",          # or "rrf"
)
# Feed topk[i].text (or fetch text via your store if None) into the generator
```

3. **Determinism tips**

* Keep the fusion secondary sort by `id` (already done).
* Log returned `(id, score)` for each query so Weekend-2/3 can replay.
* For reproducible embeddings: pin `BAAI/bge-small-en-v1.5` version in your `requirements.txt`.

4. **Pinecone index setup (once)**

* Create `typedrag-own` (cosine, 384-dim) in your account (or let the wrapper create it serverlessly).
* During ingestion, upsert with metadata:

  ```python
  # example shape
  {
    "id": "doc123::chunk_0007",
    "values": vector,  # length 384, float32
    "metadata": {
      "title": "...",
      "url": "file:///path/or/http",
      "doc_id": "doc123",
      "section": "Rolling updates",
      "chunk_idx": 7,
      "token_len": 198,
      "source": "internal",
      "text_hash": "sha1:...",   # optional audit
    }
  }
  ```

5. **BM25 expectations**

* Your Pyserini index should store each chunk as a JSON doc with at least: `id`, `text`, and the same metadata keys; the searcher restores them via `h.raw` → `json.loads(...)`.

---

## Optional: tiny upsert helper (put in your Pinecone ingestion script)

```python
def upsert_chunks_to_pinecone(
    store: PineconeDenseStore,
    embedder: BGEEmbedder,
    chunk_records: List[dict],
    batch_size: int = 256,
):
    ids = [r["id"] for r in chunk_records]
    texts = [r["text"] for r in chunk_records]
    metas = [{k: v for k, v in r.items() if k != "text"} for r in chunk_records]

    vecs = embedder.encode_passages(texts, batch_size=batch_size)
    store.upsert(ids=ids, vectors=vecs, metadatas=metas, batch_size=batch_size)
```

---

### Sanity checklist for Weekend-1 (own docs + Pinecone)

* `retrieve()` returns deterministic, fused top-K given a fixed corpus.
* **Hybrid** mode uses Pinecone + BM25 with z-score fusion and stable tie-break.
* Baselines log `{question_id, prompt, passages[], answer, latency_ms, seed}` with passage titles/URLs.
* Median top-1 latency comfortably ≤ 2s on laptop-scale corpora.
