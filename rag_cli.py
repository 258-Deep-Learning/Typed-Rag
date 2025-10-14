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
from dotenv import load_dotenv

load_dotenv()

# Import the new core module
from typed_rag.rag_system import DataType, create_index, ask_question, detect_default_backend


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
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
