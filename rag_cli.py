#!/usr/bin/env python3
"""
Typed-RAG CLI - Simple interface for building indices and asking questions.

Usage examples:
  # Build indices
  python rag_cli.py build
  python rag_cli.py build --backend pinecone
  python rag_cli.py build --backend faiss --rebuild

  # Ask questions
  python rag_cli.py ask "What is Amazon's revenue?"
  python rag_cli.py ask --backend pinecone "Summarize the design."
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import the new core module
from typed_rag.rag_system import RAGSystem, RAGConfig


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
        "--rebuild",
        action="store_true",
        help="Force re-ingest and re-index"
    )
    build_parser.add_argument(
        "--docs-dir",
        type=Path,
        help="Custom documents directory (default: my-documents/)"
    )

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question (no building)")
    ask_parser.add_argument(
        "--backend", "-b",
        choices=["pinecone", "faiss"],
        help="Backend to query (default: auto-detect)"
    )
    ask_parser.add_argument(
        "question",
        help="The question to ask"
    )

    args = parser.parse_args()

    # Initialize RAG system
    config = RAGConfig.default()
    if hasattr(args, 'docs_dir') and args.docs_dir:
        config.docs_dir = args.docs_dir
    
    rag_system = RAGSystem(config)

    # Execute command
    try:
        if args.command == "build":
            print(f"\n🔨 Building {args.backend.upper()} index...")
            rag_system.build_index(
                backend=args.backend,
                force_rebuild=args.rebuild
            )
            print(f"\n✅ Successfully built {args.backend.upper()} index!")
            print(f"\nNow you can ask questions:")
            print(f'  python rag_cli.py ask "Your question here"')
        
        elif args.command == "ask":
            backend = args.backend or rag_system.detect_default_backend()
            print(f"\n🔍 Using {backend.upper()} backend...")
            rag_system.ask(args.question, backend=backend)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
