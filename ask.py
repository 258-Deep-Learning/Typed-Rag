#!/usr/bin/env python3
"""
Simple script to ask a single question to your RAG system (LangChain version)
Usage: python3 ask.py "What is Amazon's revenue?"
"""

import os
import sys

# Use in-repo retriever components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "typed_rag"))
from retrieval.pipeline import BGEEmbedder, PineconeDenseStore, FAISSDenseStore, LCBGEEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS as LCFAISS  # type: ignore

# LangChain (LLM only)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def main():
    # Get question from command line
    if len(sys.argv) < 2:
        print("Usage: python3 ask.py \"Your question here\"")
        print("Example: python3 ask.py \"What is Amazon's revenue?\"")
        sys.exit(1)

    question = sys.argv[1]
    k = 5  # Number of passages to retrieve

    # Vector store selection
    vector_store = os.environ.get("VECTOR_STORE", "pinecone").lower()

    # Pinecone config (used when VECTOR_STORE=pinecone)
    pinecone_index = os.environ.get("PINECONE_INDEX", "typedrag-own")
    pinecone_namespace = os.environ.get("PINECONE_NAMESPACE", "own_docs")

    # FAISS config (used when VECTOR_STORE=faiss)
    faiss_dir = os.environ.get("FAISS_DIR") or os.path.join(
        os.path.dirname(__file__), "typed_rag", "indexes", "faiss"
    )
    # LangChain FAISS saves index.faiss + index.pkl
    model_name = "gemini-2.5-flash"

    print(f"🔍 Question: {question}")
    print()

    # Check API keys
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        print("❌ GOOGLE_API_KEY not set!")
        sys.exit(1)

    if vector_store == "pinecone":
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            print("❌ PINECONE_API_KEY not set!")
            sys.exit(1)

    # 1. Initialize retrieval using in-repo PineconeDenseStore + BGEEmbedder
    print("📚 Retrieving relevant passages...")

    embedder = BGEEmbedder()
    if vector_store == "faiss":
        # Load LangChain FAISS store
        if not os.path.isdir(faiss_dir):
            print(
                "❌ FAISS directory not found. Expected at:\n"
                f"   dir: {faiss_dir}\n"
                "Build it first (e.g., via main.py FAISS flow)."
            )
            sys.exit(1)
        lc_embeddings = LCBGEEmbeddings(embedder)
        try:
            lc_store = LCFAISS.load_local(faiss_dir, lc_embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"❌ Failed to load FAISS store from {faiss_dir}: {e}")
            sys.exit(1)
    else:
        store = PineconeDenseStore(
            index_name=pinecone_index,
            namespace=pinecone_namespace,
            create_if_missing=False,
        )

    # Retrieve passages (dense query with scores + metadata)
    if vector_store == "faiss":
        # Query via LangChain vectorstore
        try:
            lc_results = lc_store.similarity_search_with_score(question, k=k)
        except Exception as e:
            print(f"❌ FAISS query failed: {e}")
            sys.exit(1)
        # Convert to our expected shape
        results = []
        for doc, score in lc_results:
            md = dict(doc.metadata or {})
            text = doc.page_content or ""
            # ensure text present in metadata for display
            if "text" not in md:
                md["text"] = text
            # build id
            rec_id = md.get("id") or md.get("doc_id") or ""
            results.append({"id": rec_id, "score": float(score), "metadata": md})
    else:
        query_vec = embedder.encode_queries([question])
        results = store.query(query_vec, top_k=k)

    print(f"✓ Retrieved {len(results)} passages")
    print()

    # Display retrieved passages
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

    # 2. Generate answer with Gemini via LangChain
    print("💡 Generating answer...")

    # Build context from passages
    context_parts = []
    for i, p in enumerate(results):
        md = p.get("metadata", {}) or {}
        text = md.get("text", "")
        context_parts.append(f"[{i+1}] {text}")
    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = (
        "Answer the question based on the following passages.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (be concise and cite passage numbers):"
    )

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
    main()
