"""Type-aware answer generation for Typed-RAG."""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient

from typed_rag.decompose.query_decompose import DecompositionPlan
from typed_rag.retrieval.orchestrator import AspectEvidence, EvidenceBundle

from .prompts import GENERATION_PROMPTS
from ..core.keys import get_fastest_model


@dataclass
class AspectAnswer:
    """Structured answer for a single aspect/sub-query."""

    aspect: str
    answer: str
    strategy: str
    citations: List[str]
    sources: List[dict]

    def to_dict(self) -> dict:
        data = asdict(self)
        # Ensure sources are JSON-serializable (they already are dicts)
        return data


@dataclass
class GeneratedAnswer:
    """Container for all aspect answers."""

    question_id: str
    question: str
    question_type: str
    aspects: List[AspectAnswer]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "type": self.question_type,
            "aspects": [aspect.to_dict() for aspect in self.aspects],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedAnswer":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            question_type=data["type"],
            aspects=[
                AspectAnswer(
                    aspect=a["aspect"],
                    answer=a["answer"],
                    strategy=a["strategy"],
                    citations=list(a.get("citations", [])),
                    sources=list(a.get("sources", [])),
                )
                for a in data.get("aspects", [])
            ],
        )


class TypedAnswerGenerator:
    """Generates aspect-level answers given retrieved evidence."""

    def __init__(
        self,
        model_name=None,
        cache_dir: Optional[Path] = None,
        max_snippets: int = 3,
        temperature: float = 0.2,
        use_llm: bool = True,
    ) -> None:
        self.model_name = model_name or get_fastest_model()
        self.is_hf = "/" in self.model_name
        self.cache_dir = Path(cache_dir or "./cache/answers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_snippets = max_snippets
        self.temperature = temperature
        self.use_llm = use_llm
        self._llm = None

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #
    def _get_llm(self):
        if self._llm is None:
            if self.is_hf:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise EnvironmentError("HF_TOKEN not set")
                self._llm = InferenceClient(
                    model=self.model_name,
                    token=hf_token
                )
            else:
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    raise EnvironmentError("GOOGLE_API_KEY not set")
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=google_api_key,
                    temperature=self.temperature,
                )
        return self._llm

    def _invoke_llm(self, prompt: str) -> str:
        if not self.use_llm:
            raise RuntimeError("LLM usage disabled")

        llm = self._get_llm()
        if self.is_hf:
            response = llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        else:
            response = llm.invoke(prompt)
            return str(getattr(response, "content", response)).strip()

    # ------------------------------------------------------------------ #
    # Caching utilities
    # ------------------------------------------------------------------ #
    def _cache_path(self, question_id: str) -> Path:
        return self.cache_dir / f"{question_id}_answers.json"

    def _load_cache(self, plan: DecompositionPlan) -> Optional[GeneratedAnswer]:
        path = self._cache_path(plan.question_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return GeneratedAnswer.from_dict(json.load(f))
            except Exception as exc:
                print(f"Warning: failed to load cached answers: {exc}")
        return None

    def _save_cache(self, answers: GeneratedAnswer) -> None:
        try:
            with open(self._cache_path(answers.question_id), "w", encoding="utf-8") as f:
                json.dump(answers.to_dict(), f, indent=2)
        except Exception as exc:
            print(f"Warning: failed to save answers cache: {exc}")

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #
    def _format_snippets(self, evidence: AspectEvidence) -> str:
        snippets = []
        for idx, doc in enumerate(evidence.documents[: self.max_snippets], start=1):
            title = doc.title or "Untitled"
            text = doc.text or ""
            text = textwrap.shorten(text, width=600, placeholder="…")
            snippets.append(f"[{idx}] {title}\n{text}")
        if not snippets:
            snippets.append("[1] No supporting evidence retrieved.")
        return "\n\n".join(snippets)

    def _fallback_answer(self, evidence: AspectEvidence) -> AspectAnswer:
        """Generate a deterministic fallback answer using top documents."""
        top_docs = evidence.documents[: self.max_snippets]
        if not top_docs:
            answer = "No supporting evidence was retrieved, so no answer can be produced."
            return AspectAnswer(
                aspect=evidence.aspect,
                answer=answer,
                strategy=evidence.strategy,
                citations=[],
                sources=[],
            )

        parts = []
        citations = []
        sources = []
        for idx, doc in enumerate(top_docs, start=1):
            snippet = doc.text or ""
            snippet = textwrap.shorten(snippet, width=220, placeholder="…")
            title = doc.title or "Untitled source"
            parts.append(f"{title}: {snippet} [{idx}]")
            citations.append(f"[{idx}]")
            sources.append(
                {
                    "id": doc.id,
                    "title": title,
                    "url": doc.url,
                    "score": doc.score,
                }
            )
        answer = " ".join(parts)
        return AspectAnswer(
            aspect=evidence.aspect,
            answer=answer,
            strategy=evidence.strategy,
            citations=citations,
            sources=sources,
        )

    def _build_prompt(
        self,
        question: str,
        question_type: str,
        evidence: AspectEvidence,
    ) -> str:
        template = GENERATION_PROMPTS.get(
            question_type, GENERATION_PROMPTS["Evidence-based"]
        )
        snippets = self._format_snippets(evidence)
        axis = evidence.aspect
        return template.format(
            question=question,
            aspect=evidence.aspect,
            axis=axis,
            snippets=snippets,
        )

    def _build_sources(self, evidence: AspectEvidence) -> List[dict]:
        sources = []
        for doc in evidence.documents[: self.max_snippets]:
            sources.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "url": doc.url,
                    "score": doc.score,
                }
            )
        return sources

    def _extract_citations(self, answer: str) -> List[str]:
        citations = []
        for token in answer.split():
            if token.startswith("[") and token.endswith("]") and len(token) <= 6:
                citations.append(token)
        return list(dict.fromkeys(citations))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        plan: DecompositionPlan,
        evidence_bundle: EvidenceBundle,
        force_regen: bool = False,
    ) -> GeneratedAnswer:
        """
        Generate aspect answers for the provided plan and evidence bundle.

        Args:
            plan: Decomposition plan.
            evidence_bundle: Retrieved evidence grouped by aspect.
            force_regen: Ignore cache and force regeneration.
        """
        if not force_regen:
            cached = self._load_cache(plan)
            if cached:
                return cached

        aspects: List[AspectAnswer] = []
        for aspect_ev in evidence_bundle.evidence:
            if self.use_llm:
                # Always use LLM - no fallback for fair comparison with RAG baseline
                prompt = self._build_prompt(
                    plan.original_question,
                    plan.question_type,
                    aspect_ev,
                )
                try:
                    raw_answer = self._invoke_llm(prompt)
                    if not raw_answer:
                        raise RuntimeError("LLM returned empty answer")
                    sources = self._build_sources(aspect_ev)
                    citations = self._extract_citations(raw_answer)
                    aspects.append(
                        AspectAnswer(
                            aspect=aspect_ev.aspect,
                            answer=raw_answer,
                            strategy=aspect_ev.strategy,
                            citations=citations,
                            sources=sources,
                        )
                    )
                except Exception as exc:
                    # Re-raise exception instead of using fallback for transparent evaluation
                    print(f"Error: aspect generation failed ({aspect_ev.aspect}): {exc}")
                    raise
            else:
                # Only use fallback when LLM is explicitly disabled
                fallback = self._fallback_answer(aspect_ev)
                aspects.append(fallback)

        generated = GeneratedAnswer(
            question_id=plan.question_id,
            question=plan.original_question,
            question_type=plan.question_type,
            aspects=aspects,
        )

        self._save_cache(generated)
        return generated
