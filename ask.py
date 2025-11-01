#!/usr/bin/env python3
"""
Simple script to ask a single question to your RAG system (LangChain version)
Usage: python3 ask.py "What is Amazon's revenue?"
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "typed_rag"))
from retrieval.pipeline import BGEEmbedder, PineconeDenseStore, LCBGEEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS as LCFAISS  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


# ---------------------------
# Defaults and configuration
# ---------------------------
DEFAULT_VECTOR_STORE = "pinecone"
DEFAULT_PINECONE_INDEX = "typedrag-own"
DEFAULT_PINECONE_NAMESPACE = "own_docs"
DEFAULT_FAISS_DIR = os.path.join(os.path.dirname(__file__), "typed_rag", "indexes", "faiss")
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
RETRIEVAL_TOP_K = 5


def validate_api_keys(vector_store: str) -> str:
    """Ensure required API keys are present; exit with a clear message if not.

    Returns the Google API key (always required).
    """
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        print("❌ GOOGLE_API_KEY not set!")
        sys.exit(1)

    if vector_store == "pinecone":
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            print("❌ PINECONE_API_KEY not set!")
            sys.exit(1)

    return google_key


def load_faiss_store(faiss_dir: str, embedder: "BGEEmbedder") -> "LCFAISS":
    """Load a local FAISS index via LangChain, or exit with a friendly message."""
    if not os.path.isdir(faiss_dir):
        print(
            "❌ FAISS directory not found. Expected at:\n"
            f"   dir: {faiss_dir}\n"
            "Build it first (e.g., via main.py FAISS flow)."
        )
        sys.exit(1)

    lc_embeddings = LCBGEEmbeddings(embedder)
    try:
        return LCFAISS.load_local(faiss_dir, lc_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Failed to load FAISS store from {faiss_dir}: {e}")
        sys.exit(1)


def print_passages(results: list) -> None:
    """Pretty-print retrieved passages for quick inspection."""
    print("📄 Retrieved Passages:")
    print("-" * 80)
    for i, passage in enumerate(results, 1):
        md = passage.get("metadata", {}) or {}
        title = md.get("title", "Untitled")
        score = float(passage.get("score", 0.0))
        text = md.get("text", "[No text available]")
        print(f"\n[{i}] {title}")
        print(f"Score: {score:.4f}")
        print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
    print("-" * 80)
    print()


def build_prompt(question: str, results: list) -> str:
    """Construct the LLM prompt from retrieved passages and the question."""
    context_parts = []
    for i, p in enumerate(results):
        md = p.get("metadata", {}) or {}
        text = md.get("text", "")
        context_parts.append(f"[{i+1}] {text}")
    context = "\n\n".join(context_parts)

    return (
        "Answer the question based on the following passages.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (be concise and cite passage numbers):"
    )


def get_env_config() -> dict:
    """Read environment-driven configuration with sensible defaults."""
    return {
        "vector_store": os.environ.get("VECTOR_STORE", DEFAULT_VECTOR_STORE).lower(),
        "pinecone_index": os.environ.get("PINECONE_INDEX", DEFAULT_PINECONE_INDEX),
        "pinecone_namespace": os.environ.get("PINECONE_NAMESPACE", DEFAULT_PINECONE_NAMESPACE),
        "faiss_dir": os.environ.get("FAISS_DIR") or DEFAULT_FAISS_DIR,
        "model_name": DEFAULT_MODEL_NAME,
    }


def make_retrieval_fn(
    embedder: "BGEEmbedder",
    vector_store: str,
    pinecone_index: str,
    pinecone_namespace: str,
    faiss_dir: str,
):
    """Create a retrieval function for the selected vector store.

    The returned function has the signature: (question: str, top_k: int) -> list
    """
    if vector_store == "faiss":
        lc_store = load_faiss_store(faiss_dir, embedder)

        def retrieve(question: str, top_k: int) -> list:
            try:
                lc_results = lc_store.similarity_search_with_score(question, k=top_k)
            except Exception as e:
                print(f"❌ FAISS query failed: {e}")
                sys.exit(1)
            results = []
            for doc, score in lc_results:
                md = dict(doc.metadata or {})
                text = doc.page_content or ""
                if "text" not in md:
                    md["text"] = text
                rec_id = md.get("id") or md.get("doc_id") or ""
                results.append({"id": rec_id, "score": float(score), "metadata": md})
            return results

        return retrieve

    store = PineconeDenseStore(
        index_name=pinecone_index,
        namespace=pinecone_namespace,
        create_if_missing=False,
    )

    def retrieve(question: str, top_k: int) -> list:
        query_vec = embedder.encode_queries([question])
        return store.query(query_vec, top_k=top_k)

    return retrieve


def main(question):
    # Get question from command line
    if len(sys.argv) == 2:
        question = sys.argv[1]
    elif len(sys.argv) < 2 and not question:
        print("Usage: python3 ask.py \"Your question here\"")
        print("Example: python3 ask.py \"What is Amazon's revenue?\"")
        sys.exit(1)




    k = RETRIEVAL_TOP_K  # Number of passages to retrieve

    # Read environment-driven config
    cfg = get_env_config()
    vector_store = cfg["vector_store"]
    pinecone_index = cfg["pinecone_index"]
    pinecone_namespace = cfg["pinecone_namespace"]
    faiss_dir = cfg["faiss_dir"]
    model_name = cfg["model_name"]
    
    print(f"🔍 Question: {question}")
    print()
    # Check API keys
    google_key = validate_api_keys(vector_store)

    # 1. Initialize retrieval using in-repo PineconeDenseStore + BGEEmbedder
    print("📚 Retrieving relevant passages...")
    embedder = BGEEmbedder()
    retrieve = make_retrieval_fn(
        embedder,
        vector_store=vector_store,
        pinecone_index=pinecone_index,
        pinecone_namespace=pinecone_namespace,
        faiss_dir=faiss_dir,
    )

    # Retrieve passages (dense query with scores + metadata)
    results = retrieve(question, k)

    print(f"✓ Retrieved {len(results)} passages")
    print()

    # Display retrieved passages
    print_passages(results)

    # 2. Generate answer with Gemini via LangChain
    print("💡 Generating answer...")

    # Build prompt
    prompt = build_prompt(question, results)

    # Call Gemini through LangChain
    chat = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_key)
    response = chat.invoke(prompt)

    print("✓ Answer generated")
    print()

    # Display answer
    print("🎯 Answer:")
    print("=" * 80)
    # response.content can be str or list depending on provider; str() is safe
    print(str(response.content))
    print("=" * 80)


if __name__ == "__main__":
    main("data engineer")
