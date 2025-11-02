#!/usr/bin/env python3
"""
Utility for tracking LINKAGE-style ranking metrics (MRR, MPR) inside examples.

Implements the scoring formulas described in Typed-RAG (Park et al., 2025)
using either a lightweight lexical fallback or an optional LLM-based judge.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None  # type: ignore


PROMPT_TEMPLATE = """Please impartially rank the given candidate answer to a non-factoid question accurately within the reference answer list, which are ranked in descending order of quality. The top answers are of the highest quality, while those at the bottom may be poor or unrelated.

Determine the ranking of the given candidate answer within the provided reference answer list. For instance, if it outperforms all references, output [[1]]. If it’s deemed inferior to all references, output [[{n}]].

Your response must strictly follow this format: "[[2]]" if the candidate answer would rank second.

Question: {question}
Reference answer list:
{reference_answers}
Candidate answer:
{candidate_answer}
"""


@dataclass
class RankingRecord:
    """Captured ranking data for a single question."""

    question: str
    rank: int
    reference_count: int


class LinkageMetricsTracker:
    """
    Collect LINKAGE-style rankings and compute MRR / MPR metrics.

    Parameters
    ----------
    reference_answers:
        Mapping from exact question text to an ordered list of reference answers
        (best first, worst last).
    use_llm:
        When True, uses Gemini via langchain to obtain rankings. Falls back to a
        lexical similarity heuristic otherwise.
    model_name:
        Gemini model identifier. Only used when `use_llm` is True.
    coverage_thresholds:
        Optional sequence of float thresholds (0-1) used by the lexical fallback
        to map overlap scores to ranks. If not provided, thresholds are derived
        automatically per reference list length.
    """

    def __init__(
        self,
        reference_answers: Dict[str, Sequence[str]],
        *,
        use_llm: bool = False,
        model_name: str = "gemini-2.5-flash-lite",
        coverage_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        self._refs = reference_answers
        self._records: List[RankingRecord] = []
        self._use_llm = use_llm and ChatGoogleGenerativeAI is not None
        self._coverage_thresholds = list(coverage_thresholds) if coverage_thresholds else None

        self._llm: Optional[ChatGoogleGenerativeAI] = None
        if self._use_llm:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY is required for LLM-based ranking.")
            self._llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def record(self, question: str, candidate_answer: str) -> Optional[int]:
        """
        Record the ranking for a candidate answer.

        Returns the inferred rank (1 == best). If no references are provided for
        the question, returns None and the candidate is ignored for metrics.
        """
        references = self._refs.get(question)
        if not references:
            print(f"⚠️  No reference answers registered for question: {question!r}. Skipping metrics.")
            return None

        rank = self._rank_answer(question, references, candidate_answer)
        record = RankingRecord(question=question, rank=rank, reference_count=len(references))
        self._records.append(record)
        return rank

    def metrics(self) -> Optional[Dict[str, float]]:
        """Compute MRR and MPR over all recorded questions."""
        if not self._records:
            return None

        total = len(self._records)
        mrr = sum(1.0 / rec.rank for rec in self._records) / total
        mpr = sum(self._percentile(rec) for rec in self._records) / total
        return {"MRR": mrr, "MPR": mpr}

    def pretty_report(self) -> str:
        """Format the aggregated metrics for terminal output."""
        metrics = self.metrics()
        if not metrics:
            return "No metrics collected."
        return " | ".join(f"{name}: {value:.3f}" for name, value in metrics.items())

    # ------------------------------------------------------------------ #
    # Ranking helpers
    # ------------------------------------------------------------------ #
    def _rank_answer(self, question: str, references: Sequence[str], candidate: str) -> int:
        if self._use_llm and self._llm is not None:
            return self._rank_with_llm(question, references, candidate)
        return self._rank_with_lexical_similarity(references, candidate)

    def _rank_with_llm(self, question: str, references: Sequence[str], candidate: str) -> int:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            reference_answers="\n".join(f"{idx}. {text}" for idx, text in enumerate(references, start=1)),
            candidate_answer=candidate,
            n=len(references),
        )
        response = self._llm.invoke(prompt)
        parsed = self._parse_rank(str(getattr(response, "content", response)))
        if parsed is None:
            raise ValueError(f"Could not parse LINKAGE rank from LLM response: {response!r}")
        return max(1, min(parsed, len(references)))

    def _rank_with_lexical_similarity(self, references: Sequence[str], candidate: str) -> int:
        candidate_tokens = self._tokenize(candidate)
        if not candidate_tokens:
            return len(references)

        thresholds = self._thresholds_for(len(references))
        ranks = []
        for idx, reference in enumerate(references, start=1):
            ref_tokens = self._tokenize(reference)
            overlap = len(candidate_tokens & ref_tokens)
            coverage = overlap / max(len(ref_tokens), 1)
            ranks.append((coverage, idx))
            threshold = thresholds[idx - 1]
            if coverage >= threshold:
                return idx

        # If no threshold met, degrade gracefully by selecting the position
        # following the highest overlap (closer to LOWER quality references).
        _, fallback_idx = max(ranks, key=lambda item: item[0])
        return min(max(fallback_idx + 1, 1), len(references))

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _parse_rank(self, text: str) -> Optional[int]:
        match = re.search(r"\[\[(\d+)\]\]", text)
        return int(match.group(1)) if match else None

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9$%,]+", text.lower())
        return {
            token.strip(",")
            for token in tokens
            if token and (token.isdigit() or token.startswith("$") or len(token) > 2)
        }

    def _thresholds_for(self, n: int) -> List[float]:
        if self._coverage_thresholds:
            return list(self._coverage_thresholds)[:n] + [self._coverage_thresholds[-1]] * max(0, n - len(self._coverage_thresholds))
        # Default: start high, relax toward lower-quality references.
        start = 0.65
        step = 0.12 if n > 1 else 0.0
        return [max(0.2, start - step * i) for i in range(n)]

    def _percentile(self, record: RankingRecord) -> float:
        percentile = 1.0 - (record.rank - 1) / max(record.reference_count, 1)
        return percentile * 100.0

