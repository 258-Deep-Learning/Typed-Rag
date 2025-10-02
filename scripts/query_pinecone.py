# scripts/query_pinecone.py
import json
import os
import typer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

app = typer.Typer()

@app.command()
def main(
    q: str = typer.Option(..., "--q", help="Query string"),
    k: int = typer.Option(5, "--k", help="Number of results"),
    api_key: str = None,
    index_name: str = "dl-rag",
    model_name: str = "BAAI/bge-small-en-v1.5",
    pretty: bool = typer.Option(True, "--pretty", help="Pretty print output")
):
    """
    Query Pinecone index.
    
    Usage:
        python scripts/query_pinecone.py --q "What is deep learning?" --k 3
    """
    
    # Get API key
    if not api_key:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY or use --api-key")
    
    # Connect to Pinecone
    print(f"🔌 Connecting to Pinecone index: {index_name}...", flush=True)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Load embedding model
    print(f"📦 Loading embedding model...", flush=True)
    model = SentenceTransformer(model_name)
    
    # Encode query
    print(f"🔍 Searching for: {q}", flush=True)
    query_embedding = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding[0].tolist(),
        top_k=k,
        include_metadata=True
    )
    
    # Format results
    formatted_results = []
    for match in results['matches']:
        result = {
            "id": match['id'],
            "score": match['score'],
            "metadata": match.get('metadata', {})
        }
        formatted_results.append(result)
    
    # Output
    print("\n" + "="*60)
    if pretty:
        print(json.dumps(formatted_results, indent=2))
    else:
        print(json.dumps(formatted_results))
    
    print(f"\n✅ Found {len(formatted_results)} results from Pinecone cloud")

if __name__ == "__main__":
    app()

