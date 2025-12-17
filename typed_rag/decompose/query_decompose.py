#!/usr/bin/env python3
"""
Query Decomposition Module for Typed-RAG

Decomposes non-factoid questions into type-aware sub-queries based on question type.
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from typed_rag.core.keys import get_fastest_model


@dataclass
class SubQuery:
    """Represents a decomposed sub-query for targeted retrieval."""
    aspect: str
    query: str
    strategy: str


@dataclass
class DecompositionPlan:
    """Complete decomposition plan for a question."""
    question_id: str
    original_question: str
    question_type: str
    sub_queries: List[SubQuery]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "original_question": self.original_question,
            "type": self.question_type,
            "sub_queries": [asdict(sq) for sq in self.sub_queries]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecompositionPlan":
        return cls(
            question_id=data["question_id"],
            original_question=data["original_question"],
            question_type=data["type"],
            sub_queries=[SubQuery(**sq) for sq in data["sub_queries"]]
        )


class QueryDecomposer:
    """Decomposes questions into type-aware sub-queries using LLM."""

    # Type-specific decomposition prompts and strategies
    DECOMPOSITION_CONFIG = {
        "Evidence-based": {
            "strategy": "evidence",
            "prompt": None,  # Uses passthrough
        },
        "Comparison": {
            "strategy": "compare",
            "prompt": """Decompose this comparison question into 3-5 sub-queries covering different comparison axes.
Question: "{question}"
Extract the subjects being compared and create sub-queries for axes like: performance, safety, ecosystem, learning curve, tooling.
Return ONLY a JSON array: [{{"aspect": "...", "query": "..."}}, ...]""",
        },
        "Experience": {
            "strategy": "opinion",
            "prompt": """Decompose this experience/recommendation question into 2-4 sub-queries covering different perspectives.
Question: "{question}"
Cover angles like: reliability/quality, cost/value (TCO), maintainability/longevity, support/community.
Return ONLY a JSON array: [{{"aspect": "...", "query": "..."}}, ...]""",
        },
        "Reason": {
            "strategy": "evidence",
            "prompt": """Decompose this "why" question into 2-4 causal sub-queries.
Question: "{question}"
Cover: direct causes, mechanisms/how it works, historical/contextual factors, constraints/limitations.
Return ONLY a JSON array: [{{"aspect": "...", "query": "..."}}, ...]""",
        },
        "Instruction": {
            "strategy": "steps",
            "prompt": """Decompose this how-to question into 3-5 step-based sub-queries.
Question: "{question}"
Cover: prerequisites/requirements, setup/preparation, core steps/implementation, validation/testing, troubleshooting.
Return ONLY a JSON array: [{{"aspect": "...", "query": "..."}}, ...]""",
        },
        "Debate": {
            "strategy": "stance",
            "prompt": """Decompose this debate question into sub-queries covering different perspectives.
Question: "{question}"
Generate 2-3 sub-queries for different stances/positions and 1 neutral/synthesis sub-query.
Return ONLY a JSON array: [{{"aspect": "...", "query": "..."}}, ...]""",
        },
    }

    def __init__(self, model_name=None, cache_dir: Optional[Path] = None):
        self.model_name = model_name or get_fastest_model()
        self.is_groq = self.model_name.startswith("groq/")
        self.is_hf = "/" in self.model_name and not self.is_groq
        
        # Make cache model-specific
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            model_slug = self.model_name.replace("/", "-").replace(":", "-")
            self.cache_dir = Path(f"./cache/decomposition/{model_slug}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._llm = None

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
                    temperature=0.3
                )
        return self._llm

    def _generate_question_id(self, question: str) -> str:
        return hashlib.md5(question.encode()).hexdigest()[:12]

    def _get_cache_path(self, question: str) -> Path:
        return self.cache_dir / f"{self._generate_question_id(question)}.json"

    def _load_from_cache(self, question: str) -> Optional[DecompositionPlan]:
        cache_path = self._get_cache_path(question)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return DecompositionPlan.from_dict(json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
        return None

    def _save_to_cache(self, plan: DecompositionPlan) -> None:
        try:
            with open(self._get_cache_path(plan.original_question), 'w') as f:
                json.dump(plan.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        content = content.strip()
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()
        elif "```javascript" in content:
            return content.split("```javascript")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content

    def _decompose_with_llm(self, question: str, question_type: str) -> List[SubQuery]:
        """Decompose using LLM based on question type."""
        config = self.DECOMPOSITION_CONFIG.get(question_type)
        if not config:
            config = self.DECOMPOSITION_CONFIG["Evidence-based"]

        # Evidence-based uses passthrough
        if question_type == "Evidence-based":
            return [SubQuery(aspect="evidence", query=question, strategy="evidence")]

        try:
            llm = self._get_llm()
            prompt_text = config["prompt"].format(question=question)
            
            if self.is_hf:
                response = llm.chat_completion(
                    messages=[{"role": "user", "content": prompt_text}],
                    model=self.model_name,
                    max_tokens=500
                )
                content = response.choices[0].message.content
            else:
                response = llm.invoke(prompt_text)
                content = str(response.content)
            
            sub_queries_data = json.loads(self._extract_json(content))
            return [
                SubQuery(aspect=sq["aspect"], query=sq["query"], strategy=config["strategy"])
                for sq in sub_queries_data
            ]
        except Exception as e:
            print(f"Warning: LLM decomposition failed: {e}, using fallback")
            return self._fallback_decomposition(question, question_type, config["strategy"])

    def _fallback_decomposition(self, question: str, question_type: str, strategy: str) -> List[SubQuery]:
        """Simple fallback when LLM fails."""
        fallbacks = {
            "Comparison": [
                SubQuery("overview", question, strategy),
                SubQuery("performance", f"{question} - performance comparison", strategy),
                SubQuery("features", f"{question} - features comparison", strategy),
            ],
            "Experience": [
                SubQuery("reliability", f"{question} - reliability and quality", strategy),
                SubQuery("value", f"{question} - cost and value", strategy),
            ],
            "Reason": [
                SubQuery("causes", f"What causes {question.replace('Why', '').strip()}", strategy),
                SubQuery("mechanism", f"How does {question.replace('Why', '').strip()} work", strategy),
            ],
            "Instruction": [
                SubQuery("prerequisites", f"{question} - requirements", strategy),
                SubQuery("steps", question, strategy),
                SubQuery("troubleshooting", f"{question} - common issues", strategy),
            ],
            "Debate": [
                SubQuery("pro_stance", f"Arguments in favor of {question}", strategy),
                SubQuery("con_stance", f"Arguments against {question}", strategy),
                SubQuery("synthesis", f"Balanced analysis of {question}", strategy),
            ],
        }
        return fallbacks.get(question_type, [SubQuery("overview", question, strategy)])

    def decompose(self, question: str, question_type: str, use_cache: bool = True) -> DecompositionPlan:
        """
        Decompose a question into type-aware sub-queries.

        Args:
            question: The question to decompose
            question_type: Question type (from QuestionClassifier)
            use_cache: Whether to use cached decompositions

        Returns:
            DecompositionPlan with sub-queries
        """
        if use_cache:
            cached = self._load_from_cache(question)
            if cached and cached.question_type == question_type:
                print(f"✓ Loaded decomposition from cache")
                return cached

        print(f"🔄 Decomposing {question_type} question...")
        sub_queries = self._decompose_with_llm(question, question_type)

        plan = DecompositionPlan(
            question_id=self._generate_question_id(question),
            original_question=question,
            question_type=question_type,
            sub_queries=sub_queries
        )

        if use_cache:
            self._save_to_cache(plan)

        return plan


def decompose_question(
    question: str, 
    question_type: str, 
    cache_dir: Optional[Path] = None,
    model_name: Optional[str] = None
) -> DecompositionPlan:
    """Convenience function to decompose a single question."""
    return QueryDecomposer(model_name=model_name, cache_dir=cache_dir).decompose(question, question_type)


if __name__ == "__main__":
    from typed_rag.classifier import classify_question

    test_questions = [
        "How to install Python on Mac?",
        "Why is the sky blue?",
        "Python vs Java for web development?",
        "Should I invest in stocks or bonds?",
        "What are the pros and cons of remote work?",
    ]

    print("Testing Query Decomposer\n" + "=" * 80)
    decomposer = QueryDecomposer()

    for question in test_questions:
        print(f"\n📝 Question: {question}")
        qtype = classify_question(question, use_llm=False)
        print(f"   Type: {qtype}")

        plan = decomposer.decompose(question, qtype, use_cache=False)
        print(f"\n   Sub-queries ({len(plan.sub_queries)}):")
        for i, sq in enumerate(plan.sub_queries, 1):
            print(f"   [{i}] {sq.aspect}: {sq.query} (Strategy: {sq.strategy})")
        print("-" * 80)

    print("\n✓ Decomposition complete!")
