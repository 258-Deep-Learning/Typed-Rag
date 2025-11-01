#!/usr/bin/env python3
"""Retrieval orchestration for Typed-RAG with reranking and evidence bundling."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from typed_rag.decompose import SubQuery, DecompositionPlan


@dataclass
class RetrievedDocument:
    """Retrieved document with metadata."""
    id: str
    title: str
    text: str
    score: float
    url: Optional[str] = None
    char_offset_start: Optional[int] = None
    char_offset_end: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievedDocument":
        return cls(**data)


@dataclass
class AspectEvidence:
    """Evidence for a specific aspect."""
    aspect: str
    sub_query: str
    strategy: str
    documents: List[RetrievedDocument]

    def to_dict(self) -> dict:
        return {
            "aspect": self.aspect,
            "sub_query": self.sub_query,
            "strategy": self.strategy,
            "documents": [doc.to_dict() for doc in self.documents]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AspectEvidence":
        return cls(
            aspect=data["aspect"],
            sub_query=data["sub_query"],
            strategy=data["strategy"],
            documents=[RetrievedDocument.from_dict(d) for d in data["documents"]]
        )


@dataclass
class EvidenceBundle:
    """Complete evidence bundle for a question."""
    question_id: str
    original_question: str
    question_type: str
    evidence: List[AspectEvidence]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "original_question": self.original_question,
            "type": self.question_type,
            "evidence": [ev.to_dict() for ev in self.evidence]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceBundle":
        return cls(
            question_id=data["question_id"],
            original_question=data["original_question"],
            question_type=data["type"],
            evidence=[AspectEvidence.from_dict(ev) for ev in data["evidence"]]
        )


class RetrievalOrchestrator:
    """Orchestrates retrieval: retrieve top-k, optionally rerank, bundle by aspect."""

    def __init__(self, embedder, vector_store, vector_store_type: str = "faiss",
                 cache_dir: Optional[Path] = None, rerank: bool = False):
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_store_type = vector_store_type
        self.rerank = rerank
        self.cache_dir = Path(cache_dir or "./cache/evidence")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load reranker on first use."""
        if self._reranker is None and self.rerank:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✓ Loaded reranker")
            except Exception as e:
                print(f"Warning: Reranker failed: {e}")
                self.rerank = False
        return self._reranker

    def _cache_path(self, question_id: str) -> Path:
        return self.cache_dir / f"{question_id}_evidence.json"

    def _load_cache(self, question_id: str) -> Optional[EvidenceBundle]:
        """Load cached evidence bundle."""
        path = self._cache_path(question_id)
        if path.exists():
            try:
                with open(path) as f:
                    return EvidenceBundle.from_dict(json.load(f))
            except Exception as e:
                print(f"Warning: Cache load failed: {e}")
        return None

    def _save_cache(self, bundle: EvidenceBundle) -> None:
        """Save evidence bundle to cache."""
        try:
            with open(self._cache_path(bundle.question_id), 'w') as f:
                json.dump(bundle.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Cache save failed: {e}")

    def _retrieve_subquery(self, sub_query: SubQuery, top_k: int = 50) -> List[RetrievedDocument]:
        """Retrieve documents for a sub-query using vector search."""
        query_vec = self.embedder.encode_queries([sub_query.query])
        results = self.vector_store.query(query_vec, top_k=top_k)

        return [RetrievedDocument(
            id=r.get("id", ""),
            title=r.get("metadata", {}).get("title", "Untitled"),
            text=r.get("metadata", {}).get("text", ""),
            score=r.get("score", 0.0),
            url=r.get("metadata", {}).get("url")
        ) for r in results]

    def _rerank(self, query: str, docs: List[RetrievedDocument], top_k: int = 5) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder."""
        if not self.rerank or not docs:
            return docs[:top_k]

        reranker = self._get_reranker()
        if not reranker:
            return docs[:top_k]

        scores = reranker.predict([[query, d.text[:512]] for d in docs])
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:top_k]

        for score, doc in scored:
            doc.score = float(score)

        return [doc for _, doc in scored]

    def retrieve_evidence(self, plan: DecompositionPlan, use_cache: bool = True,
                         initial_top_k: int = 50, final_top_k: int = 5) -> EvidenceBundle:
        """
        Retrieve and bundle evidence for all sub-queries.

        Args:
            plan: Decomposition plan with sub-queries
            use_cache: Whether to use cached results
            initial_top_k: Number of docs to retrieve initially
            final_top_k: Number of docs after reranking

        Returns:
            EvidenceBundle with retrieved documents per aspect
        """
        if use_cache:
            cached = self._load_cache(plan.question_id)
            if cached:
                print("✓ Loaded from cache")
                return cached

        print(f"🔍 Retrieving evidence for {len(plan.sub_queries)} sub-queries...")
        evidence_list = []

        for i, sq in enumerate(plan.sub_queries, 1):
            print(f"  [{i}/{len(plan.sub_queries)}] {sq.aspect}: {sq.query[:50]}...")

            docs = self._retrieve_subquery(sq, initial_top_k)

            if self.rerank and len(docs) > final_top_k:
                docs = self._rerank(sq.query, docs, final_top_k)
                print(f"      ✓ {initial_top_k} → {len(docs)} (reranked)")
            else:
                docs = docs[:final_top_k]
                print(f"      ✓ {len(docs)} docs")

            evidence_list.append(AspectEvidence(sq.aspect, sq.query, sq.strategy, docs))

        bundle = EvidenceBundle(plan.question_id, plan.original_question,
                               plan.question_type, evidence_list)

        if use_cache:
            self._save_cache(bundle)

        return bundle

    def save_plan_and_evidence(self, plan: DecompositionPlan, bundle: EvidenceBundle, output_dir: Path):
        """Save decomposition plan and evidence bundle to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"{plan.question_id}_typed_plan.json", 'w') as f:
            json.dump(plan.to_dict(), f, indent=2)

        with open(output_dir / f"{plan.question_id}_evidence_bundle.json", 'w') as f:
            json.dump(bundle.to_dict(), f, indent=2)

        print(f"✓ Saved plan and evidence to {output_dir}")


def retrieve_and_save(question: str, question_type: str, embedder, vector_store,
                     output_dir: Path, vector_store_type: str = "faiss",
                     use_cache: bool = True, rerank: bool = False):
    """Complete pipeline: decompose → retrieve → save."""
    from typed_rag.decompose import decompose_question

    plan = decompose_question(question, question_type)
    orchestrator = RetrievalOrchestrator(embedder, vector_store, vector_store_type, rerank=rerank)
    bundle = orchestrator.retrieve_evidence(plan, use_cache=use_cache)
    orchestrator.save_plan_and_evidence(plan, bundle, output_dir)

    return plan, bundle
