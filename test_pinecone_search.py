#!/usr/bin/env python3
"""
Test Pinecone retrieval with your resume data
"""

import sys
import os

sys.path.insert(0, 'typed_rag')

from retrieval.pipeline import BGEEmbedder, PineconeDenseStore

def test_search(query: str, top_k: int = 5):
    """Search Pinecone index and display results."""
    
    print(f"🔍 Searching for: '{query}'")
    print("=" * 60)
    print()
    
    # Check API key
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not set!")
        print("Run: export PINECONE_API_KEY='your-key'")
        return
    
    try:
        # Initialize
        print("Initializing embedder...")
        embedder = BGEEmbedder()
        
        print("Connecting to Pinecone...")
        store = PineconeDenseStore(
            index_name='typedrag-own',
            namespace='own_docs',
            create_if_missing=False
        )
        
        # Search
        print(f"Searching for top {top_k} results...")
        print()
        
        qv = embedder.encode_queries([query])
        results = store.query(qv, top_k=top_k)
        
        if not results:
            print("No results found!")
            return
        
        print(f"✅ Found {len(results)} results:")
        print()
        
        for i, r in enumerate(results, 1):
            score = r['score']
            metadata = r['metadata']
            
            print(f"Result #{i} (Score: {score:.4f})")
            print(f"  Document: {metadata.get('title', 'Unknown')}")
            print(f"  Source: {metadata.get('url', 'Unknown')}")
            print(f"  Chunk ID: {metadata.get('id', r['id'])}")
            
            # Try to get the text from chunks.jsonl
            chunk_id = r['id']
            try:
                import json
                with open('typed_rag/data/chunks.jsonl', 'r') as f:
                    for line in f:
                        chunk = json.loads(line)
                        if chunk['id'] == chunk_id:
                            text = chunk['text'][:200]  # First 200 chars
                            print(f"  Preview: {text}...")
                            break
            except:
                pass
            
            print()
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run interactive search or use command line query."""
    
    if len(sys.argv) > 1:
        # Use command line query
        query = " ".join(sys.argv[1:])
        test_search(query)
    else:
        # Interactive mode
        print("🚀 Pinecone Search Test")
        print("=" * 60)
        print()
        print("Example queries for your resume:")
        print("  - What experience does Indraneel have with data?")
        print("  - Tell me about Toyota projects")
        print("  - What skills are mentioned?")
        print("  - Machine learning experience")
        print()
        
        while True:
            try:
                query = input("Enter search query (or 'quit' to exit): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print()
                test_search(query)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()

