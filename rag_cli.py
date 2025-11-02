#!/usr/bin/env python3
"""
Typed-RAG CLI - Simple interface for building indices and asking questions.

Usage examples:
  # Build indices
  python rag_cli.py build
  python rag_cli.py build --backend pinecone --source wikipedia
  python rag_cli.py build --backend faiss --source own_docs --rebuild

  # Ask questions
  python rag_cli.py ask "What is Amazon's revenue?"
  python rag_cli.py ask --backend pinecone --source wikipedia "Explain quantum computing"
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import the new core module
from typed_rag.rag_system import (
    DataType,
    ask_question,
    ask_typed_question,
    create_index,
    detect_default_backend,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="Typed-RAG: Build indices and ask questions with clean separation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_parser = subparsers.add_parser("build", help="Build an index (no querying)")
    build_parser.add_argument(
        "--backend", "-b",
        choices=["pinecone", "faiss"],
        default="faiss",
        help="Backend to build (default: faiss)"
    )
    build_parser.add_argument(
        "--source", "-s",
        choices=["own_docs", "wikipedia"],
        default="own_docs",
        help="Data source (default: own_docs)"
    )
    build_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-ingest and re-index"
    )

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question (no building)")
    ask_parser.add_argument(
        "--backend", "-b",
        choices=["pinecone", "faiss"],
        help="Backend to query (default: auto-detect)"
    )
    ask_parser.add_argument(
        "--source", "-s",
        choices=["own_docs", "wikipedia"],
        default="own_docs",
        help="Data source (default: own_docs)"
    )
    ask_parser.add_argument(
        "question",
        help="The question to ask"
    )

    # Typed ask command
    typed_parser = subparsers.add_parser(
        "typed-ask",
        help="Ask a question with type-aware decomposition and aggregation",
    )
    typed_parser.add_argument(
        "--backend", "-b",
        choices=["pinecone", "faiss"],
        help="Backend to query (default: auto-detect)"
    )
    typed_parser.add_argument(
        "--source", "-s",
        choices=["own_docs", "wikipedia"],
        default="own_docs",
        help="Data source (default: own_docs)"
    )
    typed_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking for retrieved documents"
    )
    typed_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM calls for generation/aggregation (use heuristic fallback)"
    )
    typed_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing plan/evidence/answer artifacts to disk"
    )
    typed_parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to store pipeline artifacts (default: ./output)"
    )
    typed_parser.add_argument(
        "question",
        help="The question to ask"
    )

    args = parser.parse_args()

    # Execute command
    try:
        if args.command == "build":
            # Create DataType
            data_type = DataType(args.backend, args.source)
            
            print(f"\n🔨 Building {args.backend.upper()} index for {args.source}...")
            create_index(data_type, rebuild=args.rebuild)
            
            print(f"\n✅ Successfully built {args.backend.upper()} index for {args.source}!")
            print(f"\nNow you can ask questions:")
            print(f'  python rag_cli.py ask "Your question here"')
            print(f'  python rag_cli.py ask --source {args.source} "Your question here"')
        
        elif args.command == "ask":
            # Auto-detect backend if not specified
            backend = args.backend or detect_default_backend(args.source)
            
            # Create DataType
            data_type = DataType(backend, args.source)
            
            print(f"\n🔍 Using {backend.upper()} backend for {args.source}...")
            ask_question(args.question, data_type)
        
        elif args.command == "typed-ask":
            backend = args.backend or detect_default_backend(args.source)
            data_type = DataType(backend, args.source)
            print(f"\n🧭 Typed-RAG | backend={backend.upper()} source={args.source}")

            output_dir = Path(args.output_dir) if args.output_dir else None
            result = ask_typed_question(
                args.question,
                data_type,
                rerank=args.rerank,
                use_llm=not args.no_llm,
                save_artifacts=not args.no_save,
                output_dir=output_dir,
            )

            print(f"\n🏷️  Question Type: {result.question_type}")
            print("\n🎯 Final Answer:\n")
            print(result.answer)

            if result.aspects:
                print("\n📚 Aspect Answers:")
                for aspect in result.aspects:
                    aspect_label = aspect.get("aspect", "aspect")
                    answer = aspect.get("answer", "")
                    print(f"\n- {aspect_label}:")
                    print(answer)

            if not args.no_save:
                target = output_dir or Path("./output")
                print(f"\n💾 Artifacts saved to: {target.resolve()}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
