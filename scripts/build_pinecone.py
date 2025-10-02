# scripts/build_pinecone.py
import json
import os
import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

app = typer.Typer()

@app.command()
def main(
    input: str = "data/passages.jsonl",
    api_key: str = None,
    index_name: str = "typed-rag",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 100,
    cloud: str = "aws",
    region: str = "us-east-1"
):
    """
    Build Pinecone index from documents.
    
    Usage:
        python scripts/build_pinecone.py --input data/dl_docs.jsonl --index-name dl-rag
    """
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY or use --api-key")
    
    # Initialize Pinecone with new API
    print(f"🔌 Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Load embedding model
    print(f"📦 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dimension = 384  # BGE-small dimension
    
    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"✨ Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        print(f"⏳ Waiting for index to be ready...")
        import time
        time.sleep(10)  # Wait for index initialization
    else:
        print(f"📌 Using existing index: {index_name}")
    
    index = pc.Index(index_name)
    
    # Load documents
    print(f"📄 Loading documents from {input}...")
    documents = []
    with open(input, 'r') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Process and upload in batches
    print(f"🚀 Encoding and uploading to Pinecone...")
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading"):
        batch_docs = documents[i:i+batch_size]
        
        # Extract texts and IDs
        texts = [doc.get('chunk_text', '') for doc in batch_docs]
        ids = [doc.get('id', f'doc_{i+j}') for j, doc in enumerate(batch_docs)]
        
        # Encode texts
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        
        # Prepare metadata (Pinecone metadata has size limits, keep it minimal)
        metadata = [
            {
                'title': doc.get('title', '')[:500],  # Limit size
                'url': doc.get('url', '')[:500],
                'text': doc.get('chunk_text', '')[:1000],  # First 1000 chars
            }
            for doc in batch_docs
        ]
        
        # Prepare vectors for upsert
        vectors = [
            (ids[j], embeddings[j].tolist(), metadata[j])
            for j in range(len(batch_docs))
        ]
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
    
    # Check final stats
    stats = index.describe_index_stats()
    print(f"\n✅ Pinecone index built successfully!")
    print(f"📊 Total vectors: {stats['total_vector_count']}")
    print(f"🏷️  Index name: {index_name}")
    print(f"\n🎯 Next: Query with --backend pinecone")

if __name__ == "__main__":
    app()

