#!/usr/bin/env python3
"""Unit tests for Typed-RAG generation and aggregation fallbacks."""

from pathlib import Path

from typed_rag.decompose.query_decompose import DecompositionPlan, SubQuery
from typed_rag.generation.generator import TypedAnswerGenerator
from typed_rag.generation.aggregator import TypedAnswerAggregator
from typed_rag.retrieval.orchestrator import (
    AspectEvidence,
    EvidenceBundle,
    RetrievedDocument,
)


def _make_plan(question_type: str) -> DecompositionPlan:
    return DecompositionPlan(
        question_id="unit-test",
        original_question="Explain the benefits of remote work?",
        question_type=question_type,
        sub_queries=[
            SubQuery(aspect="flexibility", query="remote work flexibility benefits", strategy="stance"),
            SubQuery(aspect="productivity", query="remote work productivity evidence", strategy="stance"),
        ],
    )


def _make_document(idx: int, text: str) -> RetrievedDocument:
    return RetrievedDocument(
        id=f"doc-{idx}",
        title=f"Source {idx}",
        text=text,
        score=0.8 - idx * 0.1,
        url=None,
    )


def _make_bundle(question_type: str) -> EvidenceBundle:
    evidence = [
        AspectEvidence(
            aspect="flexibility",
            sub_query="remote work flexibility benefits",
            strategy="stance",
            documents=[
                _make_document(1, "Remote work lets employees balance life and work schedules effectively."),
                _make_document(2, "Flexible hours can improve morale and retention."),
            ],
        ),
        AspectEvidence(
            aspect="productivity",
            sub_query="remote work productivity evidence",
            strategy="stance",
            documents=[
                _make_document(3, "Studies show productivity gains when distractions are reduced."),
            ],
        ),
    ]
    return EvidenceBundle(
        question_id="unit-test",
        original_question="Explain the benefits of remote work?",
        question_type=question_type,
        evidence=evidence,
    )


def test_generator_fallback_produces_citations(tmp_path: Path) -> None:
    plan = _make_plan("Debate")
    bundle = _make_bundle("Debate")

    generator = TypedAnswerGenerator(cache_dir=tmp_path / "answers", use_llm=False)
    generated = generator.generate(plan, bundle, force_regen=True)

    assert len(generated.aspects) == 2
    first = generated.aspects[0]
    assert first.citations, "Fallback generator should include pseudo citations"
    assert "Source 1" in first.answer


def test_aggregator_debate_structure(tmp_path: Path) -> None:
    plan = _make_plan("Debate")
    bundle = _make_bundle("Debate")

    generator = TypedAnswerGenerator(cache_dir=tmp_path / "answers", use_llm=False)
    generated = generator.generate(plan, bundle, force_regen=True)

    aggregator = TypedAnswerAggregator(cache_dir=tmp_path / "final", use_llm=False)
    final_answer = aggregator.aggregate(plan, generated, force_regen=True)

    assert "Debate summary" in final_answer.answer
    assert "Pro perspective" in final_answer.answer or "Neutral synthesis" in final_answer.answer


def test_instruction_aggregator_creates_numbered_steps(tmp_path: Path) -> None:
    plan = DecompositionPlan(
        question_id="unit-test-instruction",
        original_question="How to deploy a web app?",
        question_type="Instruction",
        sub_queries=[
            SubQuery(aspect="prepare", query="deployment prerequisites", strategy="steps"),
            SubQuery(aspect="deploy", query="deployment steps", strategy="steps"),
        ],
    )

    evidence = [
        AspectEvidence(
            aspect="prepare",
            sub_query="deployment prerequisites",
            strategy="steps",
            documents=[_make_document(1, "Install dependencies and configure environment variables.")],
        ),
        AspectEvidence(
            aspect="deploy",
            sub_query="deployment steps",
            strategy="steps",
            documents=[_make_document(2, "Build the project, run tests, and push to production.")],
        ),
    ]
    bundle = EvidenceBundle(
        question_id="unit-test-instruction",
        original_question="How to deploy a web app?",
        question_type="Instruction",
        evidence=evidence,
    )

    generator = TypedAnswerGenerator(cache_dir=tmp_path / "answers", use_llm=False)
    generated = generator.generate(plan, bundle, force_regen=True)

    aggregator = TypedAnswerAggregator(cache_dir=tmp_path / "final", use_llm=False)
    final_answer = aggregator.aggregate(plan, generated, force_regen=True)

    lines = [line for line in final_answer.answer.splitlines() if line.strip()]
    assert lines[0].startswith("1.")
    assert any(line.startswith("2.") for line in lines), "Expect numbered steps in instruction aggregation"
