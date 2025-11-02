#!/usr/bin/env python3
"""CLI to run LINKAGE evaluation on generated answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from typed_rag.core.keys import get_fastest_model
from .linkage import LinkageInstance, LinkageScorer, evaluate_linkage


def load_instances(path: Path) -> List[LinkageInstance]:
    """Load evaluation instances from a JSONL file."""
    instances: List[LinkageInstance] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data["question"]
            candidate = data["candidate"]
            references = data.get("references")
            if not references:
                raise ValueError("Each record must contain a non-empty 'references' list.")
            instances.append(LinkageInstance(question=question, references=references, candidate=candidate))
    return instances


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LINKAGE evaluation runner for Typed-RAG")
    parser.add_argument("--input", required=True, help="Path to JSONL file with fields question/references/candidate")
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini scorer (requires GOOGLE_API_KEY)")
    parser.add_argument("--model", default=get_fastest_model() , help="Gemini model for scoring (default: gemini-2.5-flash-lite)")

    args = parser.parse_args(argv)

    instances = load_instances(Path(args.input))
    scorer = LinkageScorer(model_name=args.model, use_llm=args.use_llm)
    mrr, mpr, results = evaluate_linkage(instances, scorer)

    print(f"Total examples: {len(instances)}")
    print(f"MRR: {mrr:.4f}")
    print(f"MPR: {mpr:.2f}%")
    print()

    for idx, (instance, result) in enumerate(zip(instances, results), start=1):
        mode = "LLM" if result.used_llm else "embedder"
        print(f"[{idx}] Rank={result.rank} ({mode}) | Question: {instance.question}")
        if result.raw_response:
            print(f"    LLM response: {result.raw_response}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

