#!/usr/bin/env python3
"""
Happy-path demo for the Typed-RAG pipeline.

Runs one question per non-factoid type through:
  classify → decompose → retrieve → generate → aggregate

Usage:
  python examples/happy_flow_demo.py
  python examples/happy_flow_demo.py --backend pinecone --source wikipedia --rerank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.rag_system import DataType, ask_typed_question
from examples.metrics_tracker import LinkageMetricsTracker


QUESTIONS: Dict[str, str] = {
    "Evidence-based": (
        "What is the Out-of-Network Deductible for this plan?"
    ),
    "Comparison": (
        "How do cost-sharing obligations for outpatient physiotherapy differ between Select Plus providers "
        "and out-of-network clinics under the CSU student policy?"
    ),
    "Experience": (
        "For students who used the home health care benefit capped at 120 visits per policy year, "
        "what coordination steps and approvals were required?"
    ),
    "Reason": (
        "Why does the plan require written consent before an out-of-network physician at a contracting facility "
        "can bill beyond the preferred cost share, and how does that protect me from surprise bills?"
    ),
    "Instruction": (
        "How do I request a prescription drug exception—such as for a contraceptive not on the Preferred Drug List—"
        "and what timelines does the plan give for 24- or 72-hour responses?"
    ),
    "Debate": (
        "Should the CSU student plan continue to set the same $5,000 out-of-pocket maximum for preferred and "
        "out-of-network services despite the higher out-of-network coinsurance?"
    ),
}

REFERENCE_ANSWERS: Dict[str, List[str]] = {
    QUESTIONS["Evidence-based"]: [
        "Deductible Out-of-Network for this plan is $200",
        # "The CSU plan sets $100 in-network and $200 out-of-network deductibles with 100 percent and 80 percent coinsurance respectively, "
        # "and both tiers accumulate toward a unified $5,000 out-of-pocket maximum.",
        # "Preferred providers waive coinsurance after the $100 deductible while out-of-network stays at 80 percent after a $200 deductible, "
        # "but the plan caps annual spending for both at $5,000.",
        # "Students pay $100 in-network or $200 outside the network before coinsurance applies, and out-of-pocket costs stop once $5,000 is reached.",
    ],
    QUESTIONS["Comparison"]: [
        "Select Plus physiotherapy costs a $10 copay per visit and is exempt from the deductible, whereas out-of-network sessions "
        "require satisfying the deductible and then incur 20 percent coinsurance.",
        "In-network physical therapy is treated as preventive with a $10 copay, while out-of-network bills draw down the deductible "
        "and only reimburse 80 percent afterward.",
        "Preferred providers collect a modest copay and skip the deductible, but non-network visits trigger deductible use plus 80 percent reimbursement.",
        "All physiotherapy visits eventually reimburse 80 percent outside the network compared to a $10 copay in-network.",
    ],
    QUESTIONS["Experience"]: [
        "Students needed a physician-ordered care plan, insurer authorisation, and diligent scheduling to use the 120-visit home health care allowance without denials.",
        "Successful users coordinated with the provider’s case manager to log visit counts, renew orders, and keep insurer approvals current.",
        "Reaching the visit ceiling typically involved confirming eligibility and keeping records of every home health appointment.",
        "Some students simply trusted the agency to track visits toward the 120-day policy maximum.",
    ],
    QUESTIONS["Reason"]: [
        "Written consent locks in preferred cost-sharing, preventing surprise bills when an out-of-network physician treats you at a contracted facility by capping charges at the in-network level.",
        "Because the plan requires written disclosure, students can refuse higher priced clinicians and avoid being balance billed beyond preferred coinsurance.",
        "Consent paperwork ensures the provider acknowledges the preferred cost-sharing cap on patient responsibility.",
        "Without consent, physicians could charge more than the preferred copay amounts.",
    ],
    QUESTIONS["Instruction"]: [
        "Call UnitedHealthcare or submit a written request to start an exception; the plan answers normal cases within 72 hours and urgent contraceptive or drug needs within 24 hours.",
        "Members contact 1-800-767-0700 to ask for a formulary exception, supplying clinical justification, and the insurer responds in three days or faster for urgent needs.",
        "Requesting an exception means notifying the company, documenting medical necessity, and waiting up to 72 hours unless the case is urgent.",
        "Students simply appeal and wait for the reviewer to respond.",
    ],
    QUESTIONS["Debate"]: [
        "Keeping both out-of-pocket maximums at $5,000 helps plan budgeting and signals fairness, but it blunts the penalty for non-network care even with 80 percent coinsurance.",
        "Equal caps simplify plan messaging, yet some argue out-of-network liabilities should be higher to reinforce network use.",
        "The shared limit is convenient but arguably should be higher out of network.",
        "Matching caps do not matter because coinsurance already differs.",
    ],
}


def run_demo(
    backend: str,
    source: str,
    rerank: bool,
    use_llm: bool,
    save_artifacts: bool,
    output_dir: Path | None,
    model_name: str | None = None,
) -> None:
    """Execute the typed pipeline on sample questions covering all types."""
    data_type = DataType(backend, source)
    metrics = LinkageMetricsTracker(reference_answers=REFERENCE_ANSWERS)

    print(
        "\n=== Typed-RAG Happy Flow ===\n"
        f"Backend : {backend}\n"
        f"Source  : {source}\n"
        f"Rerank  : {'on' if rerank else 'off'}\n"
        f"LLM     : {'on' if use_llm else 'fallback-only'}\n"
        f"Model   : {model_name or 'default'}\n"
    )

    for qtype, question in QUESTIONS.items():
        print("-" * 80)
        print(f"[{qtype}] {question}")
        try:
            result = ask_typed_question(
                question,
                data_type,
                model_name=model_name,
                rerank=rerank,
                use_llm=use_llm,
                save_artifacts=save_artifacts,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"!! Failed to answer: {exc}")
            continue

        print(f"\nFinal Answer:\n{result.answer}\n")

        rank = metrics.record(question, result.answer)
        if rank is not None:
            total_refs = len(REFERENCE_ANSWERS.get(question, []))
            print(f"LINKAGE rank estimate: {rank}/{total_refs}")

        if result.aspects:
            print("Aspect Answers:")
            for aspect in result.aspects:
                label = aspect.get("aspect") or "aspect"
                text = aspect.get("answer") or ""
                print(f"- {label}: {text}\n")

    print("-" * 80)
    if save_artifacts:
        dest = output_dir or Path("./output")
        print(f"Artifacts saved under: {dest.resolve()}")
    print(f"\nMetrics → {metrics.pretty_report()}")
    print("\n✓ Demo complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Typed-RAG demo questions for all NFQ types.")
    parser.add_argument(
        "--backend",
        "-b",
        choices=["faiss", "pinecone"],
        default="faiss",
        help="Vector store backend to use (default: faiss)",
    )
    parser.add_argument(
        "--source",
        "-s",
        choices=["own_docs", "wikipedia"],
        default="own_docs",
        help="Document source to query (default: own_docs)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking when retrieving evidence",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable Gemini calls and rely on deterministic fallbacks",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving plan/evidence/final answer artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to persist artifacts (default: ./output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., meta-llama/Llama-3.2-3B-Instruct or gemini-2.0-flash-lite)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    run_demo(
        backend=args.backend,
        source=args.source,
        rerank=args.rerank,
        use_llm=not args.no_llm,
        save_artifacts=not args.no_save,
        output_dir=args.output_dir,
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
