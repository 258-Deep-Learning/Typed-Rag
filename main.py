#!/usr/bin/env python3
"""
Typed-RAG - Main entry point

This file provides both CLI and programmatic access to the RAG system.

CLI Usage:
  python main.py build                                 # Build FAISS index for own_docs
  python main.py build -b pinecone -s wikipedia        # Build Pinecone index for Wikipedia
  python main.py ask "Your question"                   # Ask with auto-detected backend
  python main.py ask -b faiss -s own_docs "Question"   # Ask with specific backend/source

Programmatic Usage:
  from main import DataType, create_index, ask_question
  
  # Build index
  data_type = DataType('pinecone', 'own_docs')
  create_index(data_type)
  
  # Ask question
  data_type = DataType('faiss', 'wikipedia')
  ask_question("What is quantum computing?", data_type)
"""

import sys
from pathlib import Path

# Ensure imports work
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Import the CLI
from rag_cli import main as cli_main

# ============================================================================
# Public API - Export core functionality
# ============================================================================
from typed_rag.rag_system import (
    DataType,           # Main data type class
    create_index,       # Build index function
    ask_question,       # Query function
    detect_default_backend,  # Helper function
    RAGConfig,          # Configuration (advanced usage)
    IndexBuilder,       # Builder class (advanced usage)
    QueryEngine,        # Query engine (advanced usage)
)

# Make these available when importing main
__all__ = [
    'DataType',
    'create_index',
    'ask_question',
    'detect_default_backend',
    'RAGConfig',
    'IndexBuilder',
    'QueryEngine',
]


def _prompt_source() -> str:
    """Prompt the user to choose a data source."""
    while True:
        print("\nChoose data source:")
        print("  1) own_docs")
        print("  2) wikipedia")
        choice = input("> ").strip()
        if choice in {"1", "own_docs", "docs"}:
            return "own_docs"
        if choice in {"2", "wikipedia", "wiki"}:
            return "wikipedia"
        print("Invalid choice. Please enter 1 or 2.")


def _interactive_menu() -> int:
    """Run an interactive menu for common tasks."""
    print("\nTyped-RAG Interactive Menu")
    print("==========================")
    print("1) Use Pinecone and build index")
    print("2) Use FAISS and build index")
    print("3) Use Pinecone and ask question")
    print("4) Use FAISS and ask question")
    print("q) Quit")

    choice = input("> ").strip().lower()

    try:
        if choice == "1":
            source = _prompt_source()
            data_type = DataType("pinecone", source)
            print(f"\n🔨 Building PINECONE index for {source}...")
            create_index(data_type, rebuild=False)
            print("\n✅ Build complete.")
            return 0

        if choice == "2":
            source = _prompt_source()
            data_type = DataType("faiss", source)
            print(f"\n🔨 Building FAISS index for {source}...")
            create_index(data_type, rebuild=False)
            print("\n✅ Build complete.")
            return 0

        if choice == "3":
            source = _prompt_source()
            question = input("\nEnter your question:\n> ").strip()
            if not question:
                print("Question cannot be empty.")
                return 1
            data_type = DataType("pinecone", source)
            print(f"\n🔍 Using PINECONE for {source}...")
            ask_question(question, data_type)
            return 0

        if choice == "4":
            source = _prompt_source()
            question = input("\nEnter your question:\n> ").strip()
            if not question:
                print("Question cannot be empty.")
                return 1
            data_type = DataType("faiss", source)
            print(f"\n🔍 Using FAISS for {source}...")
            ask_question(question, data_type)
            return 0

        if choice in {"q", "quit", "exit"}:
            return 0

        print("Invalid choice. Exiting.")
        return 2
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    # If no arguments provided, run interactive menu; otherwise use the CLI.
    if len(sys.argv) <= 1:
        exit(_interactive_menu())
    exit(cli_main())
