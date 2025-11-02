"""LINKAGE-style evaluation utilities for Typed-RAG."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

from typed_rag.core.keys import get_fastest_model
from typed_rag.retrieval.pipeline import BGEEmbedder

LINKAGE_PROMPT = """Please impartially rank the given candidate answer to a non-factoid question accurately within the reference answer list, which are ranked in descending order of quality. The top answers are of the highest quality, while those at the bottom may be poor or unrelated.
Determine the ranking of the given candidate answer within the provided reference answer list. For instance, if it outperforms all references, output [[1]]. If it’s deemed inferior to all references, output [[{last}]].
Your response must strictly follow this format: "[[2]]" if candidate answer could rank 2nd.

Question: {question}
Reference answer list:
{references}

Candidate answer:
{candidate}
"""


@dataclass
class LinkageInstance:
    """Single evaluation example."""

    question: str
    references: Sequence[str]
    candidate: str


@dataclass
class LinkageResult:
    """Result returned by the scorer."""

    rank: int
    raw_response: Optional[str] = None
    used_llm: bool = False


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    """Compute MRR for a list of 1-indexed ranks."""
    if not ranks:
        return 0.0
    return float(sum(1.0 / rank for rank in ranks) / len(ranks))


def mean_percentile_rank(ranks: Sequence[int], reference_sizes: Sequence[int]) -> float:
    """Compute MPR (%) given ranks and corresponding reference list sizes."""
    if not ranks:
        return 0.0
    percentiles = []
    for rank, ref_size in zip(ranks, reference_sizes):
        percentiles.append(1.0 - (rank - 1) / ref_size)
    return float(sum(percentiles) / len(percentiles) * 100.0)


class LinkageScorer:
    """LLM-backed LINKAGE scorer with deterministic fallback."""

    def __init__(
        self,
        model_name =None,
        temperature: float = 0.0,
        use_llm: bool = True,
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        self.model_name = get_fastest_model() or model_name
        self.temperature = temperature
        self.use_llm = use_llm
        self.embedder = embedder or BGEEmbedder()
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set")
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=google_api_key,
                temperature=self.temperature,
            )
        return self._llm

    def _score_with_llm(self, instance: LinkageInstance) -> LinkageResult:
        prompt = LINKAGE_PROMPT.format(
            question=instance.question,
            references="\n".join(
                f"{idx+1}. {text}" for idx, text in enumerate(instance.references)
            ),
            candidate=instance.candidate,
            last=len(instance.references) + 1,
        )
        response = self._get_llm().invoke(prompt)
        content = str(getattr(response, "content", response)).strip()
        for part in content.split():
            if part.startswith("[[") and part.endswith("]]"):
                try:
                    rank = int(part.strip("[]"))
                    return LinkageResult(rank=rank, raw_response=content, used_llm=True)
                except ValueError:
                    continue
        raise RuntimeError(f"LLM response did not contain a valid rank: {content}")

    def _score_with_embeddings(self, instance: LinkageInstance) -> LinkageResult:
        """Deterministic fallback based on cosine similarity to references."""
        refs = list(instance.references)
        # Embed candidate + references
        texts = [instance.candidate, *refs]
        vecs = self.embedder.encode_passages(texts)
        candidate_vec = vecs[0]
        ref_vecs = vecs[1:]
        sims = ref_vecs @ candidate_vec  # cosine since embeddings are normalized
        # Determine insertion rank based on similarity preserving original order
        ranked_refs = list(enumerate(sims, start=1))
        ranked_refs.sort(key=lambda item: item[1], reverse=True)
        # Determine how many references candidate surpasses
        better = sum(sim > sims.mean() for _, sim in ranked_refs)
        rank = max(1, min(len(refs) + 1, len(refs) - better + 1))
        return LinkageResult(rank=rank, raw_response=None, used_llm=False)

    def score(self, instance: LinkageInstance) -> LinkageResult:
        """Score a candidate answer using LINKAGE."""
        if not instance.references:
            return LinkageResult(rank=1, raw_response=None, used_llm=False)

        if self.use_llm:
            try:
                return self._score_with_llm(instance)
            except Exception as exc:
                print(f"Warning: LINKAGE LLM scorer failed, falling back ({exc})")

        return self._score_with_embeddings(instance)


def evaluate_linkage(
    instances: Iterable[LinkageInstance],
    scorer: LinkageScorer,
) -> Tuple[float, float, List[LinkageResult]]:
    """Evaluate a collection of instances and return (MRR, MPR, detailed results)."""
    ranks: List[int] = []
    ref_sizes: List[int] = []
    results: List[LinkageResult] = []

    for instance in instances:
        result = scorer.score(instance)
        ranks.append(result.rank)
        ref_sizes.append(len(instance.references) + 1)
        results.append(result)

    mrr = mean_reciprocal_rank(ranks)
    mpr = mean_percentile_rank(ranks, ref_sizes)
    return mrr, mpr, results

