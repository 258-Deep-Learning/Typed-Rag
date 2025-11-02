"""Utilities to synthesise reference answers for LINKAGE evaluation."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class ReferenceSet:
    """Reference answers sorted from highest to lowest quality."""

    question: str
    references: Sequence[str]

    def to_dict(self) -> dict:
        return {"question": self.question, "references": list(self.references)}


def _simplify(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def generate_reference_answers(base_answer: str, num_variations: int = 4) -> List[str]:
    """
    Generate deterministic quality-ordered references from a seed answer.

    The first element mirrors the base answer (best quality) followed by
    progressively degraded variants.
    """
    base_answer = base_answer.strip()
    if not base_answer:
        return [
            "No reliable answer was provided.",
            "The assistant could not determine an answer.",
            "Insufficient information is available.",
            "Answer not found.",
        ][:num_variations]

    variations: List[str] = [base_answer]

    # High-quality but concise
    variations.append(textwrap.shorten(base_answer, width=420, placeholder="…"))

    # Partial answer (first sentences)
    sentences = base_answer.split(". ")
    partial = ". ".join(sentences[:2]).strip()
    if partial and partial != variations[-1]:
        variations.append(partial if partial.endswith(".") else f"{partial}.")
    else:
        variations.append(_simplify(base_answer, 35))

    # Generic hedge answer
    variations.append(
        "The available sources are inconclusive; more evidence is needed to answer confidently."
    )

    return variations[:num_variations]


def save_reference_set(reference_set: ReferenceSet, path: Path) -> None:
    """Persist references to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reference_set.to_dict(), f, indent=2)

