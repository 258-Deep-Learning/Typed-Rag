"""Aggregation of aspect answers into a final response."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient

from typed_rag.decompose.query_decompose import DecompositionPlan

from .generator import GeneratedAnswer
from .prompts import AGGREGATION_PROMPTS
from ..core.keys import get_fastest_model


@dataclass
class AggregatedAnswer:
    """Final aggregated answer for a question."""

    question_id: str
    question: str
    question_type: str
    answer: str
    aspects: List[dict]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "type": self.question_type,
            "answer": self.answer,
            "aspects": self.aspects,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AggregatedAnswer":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            question_type=data["type"],
            answer=data["answer"],
            aspects=list(data.get("aspects", [])),
        )


class TypedAnswerAggregator:
    """Aggregates aspect answers into type-aware final responses."""

    def __init__(
        self,
        model_name = None,
        cache_dir: Optional[Path] = None,
        temperature: float = 0.2,
        use_llm: bool = True,
    ) -> None:
        self.model_name = get_fastest_model()  or model_name
        self.is_hf = "/" in self.model_name
        self.cache_dir = Path(cache_dir or "./cache/final_answers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.use_llm = use_llm
        self._llm = None

    # ------------------------------------------------------------------ #
    # LLM utilities
    # ------------------------------------------------------------------ #
    def _get_llm(self):
        if self._llm is None:
            if self.is_hf:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise EnvironmentError("HF_TOKEN not set")
                self._llm = InferenceClient(token=hf_token)
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
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _cache_path(self, question_id: str) -> Path:
        return self.cache_dir / f"{question_id}_final.json"

    def _load_cache(self, plan: DecompositionPlan) -> Optional[AggregatedAnswer]:
        path = self._cache_path(plan.question_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return AggregatedAnswer.from_dict(json.load(f))
            except Exception as exc:
                print(f"Warning: failed to load final answer cache: {exc}")
        return None

    def _save_cache(self, answer: AggregatedAnswer) -> None:
        try:
            with open(self._cache_path(answer.question_id), "w", encoding="utf-8") as f:
                json.dump(answer.to_dict(), f, indent=2)
        except Exception as exc:
            print(f"Warning: failed to save final answer cache: {exc}")

    # ------------------------------------------------------------------ #
    # Fallback aggregation
    # ------------------------------------------------------------------ #
    def _first_sentence(self, text: str) -> str:
        match = re.search(r"(.+?[.!?])\s", text + " ")
        return match.group(1).strip() if match else text.strip()

    def _strip_bullets(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^[\-\*\d\.]+\s*", "", stripped)
            lines.append(stripped)
        return lines

    def _fallback_evidence(self, aspects_text: List[tuple[str, str]]) -> str:
        if not aspects_text:
            return "No answer could be generated because no aspect answers were available."
        return self._first_sentence(aspects_text[0][1])

    def _fallback_comparison(self, aspects_text: List[tuple[str, str]]) -> str:
        if not aspects_text:
            return "No comparative findings were generated."
        overview = self._first_sentence(aspects_text[0][1])
        bullets = "\n".join(f"- {aspect}: {text}" for aspect, text in aspects_text)
        return f"{overview}\n\n{bullets}"

    def _fallback_experience(self, aspects_text: List[tuple[str, str]]) -> str:
        if not aspects_text:
            return "No experiential guidance is available."
        overview = self._first_sentence(aspects_text[0][1])
        bullets = "\n".join(f"- {text}" for _, text in aspects_text)
        return f"{overview}\n\nRecommendations:\n{bullets}"

    def _fallback_reason(self, aspects_text: List[tuple[str, str]]) -> str:
        if not aspects_text:
            return "No explanatory factors were identified."
        overview = "Key factors include:"
        bullets = "\n".join(f"- {text}" for _, text in aspects_text)
        return f"{overview}\n{bullets}"

    def _fallback_instruction(self, aspects_text: List[tuple[str, str]]) -> str:
        if not aspects_text:
            return "No procedural guidance is available."
        steps: List[str] = []
        for _, text in aspects_text:
            steps.extend(self._strip_bullets(text))
        deduped: List[str] = []
        seen = set()
        for step in steps:
            key = step.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(step)
        if not deduped:
            deduped = [text for _, text in aspects_text]
        numbered = "\n".join(f"{idx}. {step}" for idx, step in enumerate(deduped, start=1))
        return numbered

    def _fallback_debate(self, aspects_text: List[tuple[str, str]]) -> str:
        if not aspects_text:
            return "No debate perspectives were retrieved."
        pro_lines = []
        con_lines = []
        neutral_lines = []
        for aspect, text in aspects_text:
            bucket = neutral_lines
            if "pro" in aspect.lower() or "support" in aspect.lower():
                bucket = pro_lines
            elif "con" in aspect.lower() or "against" in aspect.lower():
                bucket = con_lines
            bucket.append(text)
        summary = ["Debate summary:"]
        if pro_lines:
            summary.append("Pro perspective:")
            summary.extend(f"- {t}" for t in pro_lines)
        if con_lines:
            summary.append("Con perspective:")
            summary.extend(f"- {t}" for t in con_lines)
        if neutral_lines:
            summary.append("Neutral synthesis:")
            summary.extend(f"- {t}" for t in neutral_lines)
        return "\n".join(summary)

    def _fallback_aggregate(
        self,
        question_type: str,
        aspects_text: List[tuple[str, str]],
    ) -> str:
        if question_type == "Evidence-based":
            return self._fallback_evidence(aspects_text)
        if question_type == "Comparison":
            return self._fallback_comparison(aspects_text)
        if question_type == "Experience":
            return self._fallback_experience(aspects_text)
        if question_type == "Reason":
            return self._fallback_reason(aspects_text)
        if question_type == "Instruction":
            return self._fallback_instruction(aspects_text)
        if question_type == "Debate":
            return self._fallback_debate(aspects_text)
        return self._fallback_evidence(aspects_text)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def aggregate(
        self,
        plan: DecompositionPlan,
        generated_answers: GeneratedAnswer,
        force_regen: bool = False,
    ) -> AggregatedAnswer:
        if not force_regen:
            cached = self._load_cache(plan)
            if cached:
                return cached

        aspect_text_pairs = [
            (aspect.aspect, aspect.answer) for aspect in generated_answers.aspects
        ]

        final_answer: Optional[str] = None
        if self.use_llm and aspect_text_pairs:
            template = AGGREGATION_PROMPTS.get(
                plan.question_type, AGGREGATION_PROMPTS["Evidence-based"]
            )
            formatted_aspects = "\n".join(
                f"- {aspect}: {text}" for aspect, text in aspect_text_pairs
            )
            prompt = template.format(
                question=plan.original_question,
                aspect_answers=formatted_aspects,
            )
            try:
                final_answer = self._invoke_llm(prompt)
                if not final_answer:
                    raise RuntimeError("LLM returned empty answer")
            except Exception as exc:
                print(f"Warning: aggregation LLM failed: {exc}")
                final_answer = None

        if final_answer is None:
            final_answer = self._fallback_aggregate(
                plan.question_type,
                aspect_text_pairs,
            )

        aggregated = AggregatedAnswer(
            question_id=plan.question_id,
            question=plan.original_question,
            question_type=plan.question_type,
            answer=final_answer,
            aspects=[a.to_dict() for a in generated_answers.aspects],
        )
        self._save_cache(aggregated)
        return aggregated

