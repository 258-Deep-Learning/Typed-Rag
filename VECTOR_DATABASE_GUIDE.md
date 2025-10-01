# Vector Database Integration Guide

## 🗄️ Replacing FAISS with Vector Databases

This guide shows you how to replace the local FAISS index with popular vector databases like Pinecone, Weaviate, Qdrant, or Milvus.

---

## 📊 Comparison: FAISS vs Vector Databases

### Current Setup (FAISS Local)

**Pros:**
- ✅ Simple, no external services
- ✅ Fast (218ms median latency)
- ✅ Good for <100K documents
- ✅ No API costs
- ✅ Complete control

**Cons:**
- ❌ Loads all in memory (~500MB)
- ❌ No distributed search
- ❌ Manual index updates
- ❌ Not scalable beyond 1M vectors

### Vector Databases

**Pros:**
- ✅ Scalable (millions/billions of vectors)
- ✅ Real-time updates
- ✅ Metadata filtering
- ✅ Multi-tenancy support
- ✅ Distributed architecture
- ✅ Built-in persistence

**Cons:**
- ❌ External dependency
- ❌ API latency overhead
- ❌ Potential costs (for managed services)
- ❌ More complex setup

---

## 🎯 When to Use Vector Databases?

| Factor | Use FAISS | Use Vector DB |
|--------|-----------|---------------|
| **Document Count** | <100K | >100K |
| **Update Frequency** | Batch (daily/weekly) | Real-time |
| **Deployment** | Single machine | Distributed/Cloud |
| **Budget** | Cost-sensitive | Have infrastructure budget |
| **Filtering** | Simple | Complex metadata filtering |
| **Multi-tenancy** | Single user/app | Multiple users/apps |

---

## 🚀 Option 1: Pinecone (Managed, Easy)

### Why Pinecone?
- Easiest to set up (fully managed)
- No infrastructure management
- Great for prototypes and production
- Free tier available

### Setup

#### 1. Install Pinecone

```bash
pip install pinecone-client
```

#### 2. Get API Key

1. Sign up at https://www.pinecone.io/
2. Create a project
3. Copy your API key and environment

#### 3. Create Modified Retrieval Module

Create `retrieval/hybrid_pinecone.py`:

```python
# retrieval/hybrid_pinecone.py
import json
import numpy as np
import joblib
import pinecone
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Literal
from scipy.stats import zscore


class HybridRetrieverPinecone:
    """
    Hybrid retrieval system with Pinecone vector database.
    """
    
    def __init__(self, 
                 bm25_path: str = "indexes/bm25_rank.pkl",
                 meta_path: str = "indexes/meta.jsonl",
                 pinecone_api_key: str = None,
                 pinecone_environment: str = None,
                 pinecone_index_name: str = "typed-rag",
                 embedding_model: str = "BAAI/bge-small-en-v1.5"):
        
        self.bm25_path = bm25_path
        self.meta_path = meta_path
        self.pinecone_index_name = pinecone_index_name
        
        # Initialize components
        self._load_bm25()
        self._load_pinecone(pinecone_api_key, pinecone_environment)
        self._load_metadata()
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ Embedding model loaded: {embedding_model}")
    
    def _load_bm25(self):
        """Load BM25 index and build ranker from cached tokens."""
        try:
            data = joblib.load(self.bm25_path)
            self.bm25_tokens = data["tokens"]
            self.bm25_ranker = BM25Okapi(self.bm25_tokens)
            print(f"✅ BM25 loaded: {len(self.bm25_tokens)} docs")
        except Exception as e:
            print(f"❌ Failed to load BM25: {e}")
            self.bm25_ranker = None
            self.bm25_tokens = []
    
    def _load_pinecone(self, api_key: str, environment: str):
        """Initialize Pinecone connection."""
        try:
            pinecone.init(api_key=api_key, environment=environment)
            
            # Check if index exists, create if not
            if self.pinecone_index_name not in pinecone.list_indexes():
                print(f"⚠️  Index '{self.pinecone_index_name}' not found. Create it first!")
                self.pinecone_index = None
            else:
                self.pinecone_index = pinecone.Index(self.pinecone_index_name)
                stats = self.pinecone_index.describe_index_stats()
                print(f"✅ Pinecone loaded: {stats['total_vector_count']} vectors")
        except Exception as e:
            print(f"❌ Failed to load Pinecone: {e}")
            self.pinecone_index = None
    
    def _load_metadata(self):
        """Load document metadata for result formatting."""
        try:
            self.metadata = {}
            with open(self.meta_path, 'r') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        doc = json.loads(line)
                        # Store by ID for lookup
                        self.metadata[doc.get('id', str(idx))] = doc
            print(f"✅ Metadata loaded: {len(self.metadata)} docs")
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
            self.metadata = {}
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query using same method as BM25 index."""
        import re
        TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
        return TOKEN_RE.findall(query.lower())
    
    def _search_bm25(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using BM25 only."""
        if not self.bm25_ranker:
            return []
        
        query_tokens = self._tokenize_query(query)
        scores = self.bm25_ranker.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            # Get metadata by index position
            doc_id = list(self.metadata.keys())[idx] if idx < len(self.metadata) else None
            if doc_id:
                result = self.metadata[doc_id].copy()
                result["score"] = float(scores[idx])
                results.append(result)
        
        return results
    
    def _search_pinecone(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using Pinecone only."""
        if not self.pinecone_index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        # Search Pinecone
        results = self.pinecone_index.query(
            vector=query_embedding[0].tolist(),
            top_k=k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            doc_id = match['id']
            if doc_id in self.metadata:
                result = self.metadata[doc_id].copy()
                result["score"] = float(match['score'])
                formatted_results.append(result)
        
        return formatted_results
    
    def _search_hybrid(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using hybrid BM25 + Pinecone with z-score normalization."""
        if not self.bm25_ranker or not self.pinecone_index:
            print("⚠️  Hybrid mode requires both BM25 and Pinecone.")
            if self.bm25_ranker:
                return self._search_bm25(query, k)
            elif self.pinecone_index:
                return self._search_pinecone(query, k)
            else:
                return []
        
        # Get larger candidate set for better score normalization
        candidate_k = min(k * 3, len(self.metadata))
        
        # BM25 search
        query_tokens = self._tokenize_query(query)
        bm25_scores = self.bm25_ranker.get_scores(query_tokens)
        
        # Pinecone search
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        pinecone_results = self.pinecone_index.query(
            vector=query_embedding[0].tolist(),
            top_k=candidate_k,
            include_metadata=True
        )
        
        # Create score dictionaries by document ID
        n_docs = len(self.metadata)
        doc_ids = list(self.metadata.keys())
        
        bm25_score_dict = {}
        for idx, doc_id in enumerate(doc_ids):
            if idx < len(bm25_scores):
                bm25_score_dict[doc_id] = bm25_scores[idx]
        
        pinecone_score_dict = {}
        for match in pinecone_results['matches']:
            pinecone_score_dict[match['id']] = match['score']
        
        # Combine scores with z-score normalization
        all_bm25 = list(bm25_score_dict.values())
        all_pinecone = list(pinecone_score_dict.values())
        
        # Z-score normalization
        z_bm25_dict = {}
        z_pinecone_dict = {}
        
        if len(all_bm25) > 0 and np.std(all_bm25) > 0:
            bm25_mean, bm25_std = np.mean(all_bm25), np.std(all_bm25)
            z_bm25_dict = {k: (v - bm25_mean) / bm25_std for k, v in bm25_score_dict.items()}
        
        if len(all_pinecone) > 0 and np.std(all_pinecone) > 0:
            pin_mean, pin_std = np.mean(all_pinecone), np.std(all_pinecone)
            z_pinecone_dict = {k: (v - pin_mean) / pin_std for k, v in pinecone_score_dict.items()}
        
        # Combine scores
        hybrid_scores = {}
        all_doc_ids = set(list(bm25_score_dict.keys()) + list(pinecone_score_dict.keys()))
        
        for doc_id in all_doc_ids:
            z_bm25 = z_bm25_dict.get(doc_id, 0)
            z_pin = z_pinecone_dict.get(doc_id, 0)
            hybrid_scores[doc_id] = z_bm25 + z_pin
        
        # Get top-k results
        top_doc_ids = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:k]
        
        results = []
        for doc_id in top_doc_ids:
            if doc_id in self.metadata:
                result = self.metadata[doc_id].copy()
                result["score"] = float(hybrid_scores[doc_id])
                result["bm25_score"] = float(bm25_score_dict.get(doc_id, 0))
                result["pinecone_score"] = float(pinecone_score_dict.get(doc_id, 0))
                results.append(result)
        
        return results
    
    def retrieve(self, 
                query: str, 
                k: int = 5, 
                mode: Literal["bm25", "pinecone", "hybrid"] = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a given query.
        
        Args:
            query: Search query string
            k: Number of results to return
            mode: Search mode - 'bm25', 'pinecone', or 'hybrid'
            
        Returns:
            List of documents with metadata and scores
        """
        if mode == "bm25":
            return self._search_bm25(query, k)
        elif mode == "pinecone":
            return self._search_pinecone(query, k)
        elif mode == "hybrid":
            return self._search_hybrid(query, k)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bm25', 'pinecone', or 'hybrid'")
    
    def health_check(self) -> Dict[str, bool]:
        """Check the health of all components."""
        return {
            "bm25_loaded": self.bm25_ranker is not None,
            "pinecone_loaded": self.pinecone_index is not None,
            "metadata_loaded": len(self.metadata) > 0,
            "embedding_model_loaded": self.embedding_model is not None
        }
```

#### 4. Create Pinecone Index Building Script

Create `scripts/build_pinecone.py`:

```python
# scripts/build_pinecone.py
import json
import os
import pinecone
import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

app = typer.Typer()

@app.command()
def main(
    input: str = "data/passages.jsonl",
    api_key: str = None,
    environment: str = None,
    index_name: str = "typed-rag",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 100
):
    """
    Build Pinecone index from documents.
    
    Usage:
        python scripts/build_pinecone.py --api-key YOUR_KEY --environment YOUR_ENV
    """
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY or use --api-key")
    
    if not environment:
        environment = os.environ.get("PINECONE_ENVIRONMENT")
        if not environment:
            raise ValueError("Pinecone environment required. Set PINECONE_ENVIRONMENT or use --environment")
    
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)
    
    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dimension = 384  # BGE-small dimension
    
    # Create index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        print(f"Creating new index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )
    else:
        print(f"Using existing index: {index_name}")
    
    index = pinecone.Index(index_name)
    
    # Load documents
    print("Loading documents...")
    documents = []
    with open(input, 'r') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    
    print(f"Loaded {len(documents)} documents")
    
    # Process and upload in batches
    print("Encoding and uploading to Pinecone...")
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
    print(f"Total vectors: {stats['total_vector_count']}")
    print(f"Index name: {index_name}")

if __name__ == "__main__":
    app()
```

#### 5. Build Your Pinecone Index

```bash
# Set environment variables
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_ENVIRONMENT="your-environment-here"  # e.g., "us-west1-gcp"

# Build Pinecone index
python scripts/build_pinecone.py --input data/passages.jsonl --index-name typed-rag

# Build BM25 index (still needed for hybrid)
python scripts/build_bm25.py --input data/passages.jsonl
```

#### 6. Query with Pinecone

Create `scripts/query_pinecone.py`:

```python
# scripts/query_pinecone.py
import json
import os
import typer
from typing import Literal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.hybrid_pinecone import HybridRetrieverPinecone

app = typer.Typer()

@app.command()
def main(
    q: str = typer.Option(..., "--q", help="Query string"),
    k: int = typer.Option(5, "--k", help="Number of results"),
    mode: Literal["bm25", "pinecone", "hybrid"] = typer.Option("hybrid", "--mode"),
    api_key: str = None,
    environment: str = None,
    index_name: str = "typed-rag"
):
    """Query the Pinecone-based retrieval system."""
    
    # Get credentials from environment if not provided
    if not api_key:
        api_key = os.environ.get("PINECONE_API_KEY")
    if not environment:
        environment = os.environ.get("PINECONE_ENVIRONMENT")
    
    # Initialize retriever
    retriever = HybridRetrieverPinecone(
        pinecone_api_key=api_key,
        pinecone_environment=environment,
        pinecone_index_name=index_name
    )
    
    # Perform search
    results = retriever.retrieve(query=q, k=k, mode=mode)
    
    # Display results
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()
```

**Usage:**

```bash
python scripts/query_pinecone.py --q "What are neural networks?" --k 5 --mode hybrid
```

---

## 🚀 Option 2: Weaviate (Open Source, Feature-Rich)

### Why Weaviate?
- Open source
- Can self-host or use cloud
- Built-in text processing
- Great documentation
- Free tier available

### Quick Setup

```bash
# Install client
pip install weaviate-client

# Run Weaviate with Docker
docker run -d \
  -p 8080:8080 \
  --name weaviate \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

### Integration Example

```python
# retrieval/hybrid_weaviate.py (simplified)
import weaviate

class HybridRetrieverWeaviate:
    def __init__(self, weaviate_url="http://localhost:8080"):
        self.client = weaviate.Client(weaviate_url)
        # ... similar structure to Pinecone version
    
    def _search_weaviate(self, query: str, k: int):
        result = (
            self.client.query
            .get("Document", ["id", "title", "url", "chunk_text"])
            .with_near_text({"concepts": [query]})
            .with_limit(k)
            .do()
        )
        return result['data']['Get']['Document']
```

---

## 🚀 Option 3: Qdrant (Modern, High-Performance)

### Why Qdrant?
- Modern Rust-based (very fast)
- Easy to deploy
- Great for production
- Generous free tier

### Quick Setup

```bash
# Install client
pip install qdrant-client

# Run Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant
```

### Integration Example

```python
# retrieval/hybrid_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class HybridRetrieverQdrant:
    def __init__(self, qdrant_url="localhost", port=6333, collection_name="typed-rag"):
        self.client = QdrantClient(host=qdrant_url, port=port)
        self.collection_name = collection_name
        # ... similar structure
    
    def _search_qdrant(self, query: str, k: int):
        query_vector = self.embedding_model.encode([query])[0].tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        return results
```

---

## 🚀 Option 4: Milvus (Enterprise-Scale)

### Why Milvus?
- Built for massive scale
- Industry standard for large deployments
- LF AI & Data Foundation project
- Excellent for >10M vectors

### Quick Setup

```bash
# Install client
pip install pymilvus

# Run Milvus with Docker Compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

---

## 📊 Performance Comparison

| Vector DB | Setup Difficulty | Query Latency | Max Vectors | Cost (Managed) |
|-----------|------------------|---------------|-------------|----------------|
| **FAISS (Local)** | ⭐ Easy | 50-100ms | 100K-1M | Free |
| **Pinecone** | ⭐ Easy | 100-200ms | Billions | $70+/mo |
| **Weaviate** | ⭐⭐ Medium | 80-150ms | Billions | Free (self-host) |
| **Qdrant** | ⭐⭐ Medium | 50-100ms | Billions | Free (self-host) |
| **Milvus** | ⭐⭐⭐ Hard | 50-100ms | Billions | Free (self-host) |

---

## 🎯 Recommendation

**For your situation:**

1. **Starting out / Prototype** → Stick with **FAISS** (current setup)
   - Simple, fast, free
   - Works great for <100K documents

2. **Need real-time updates** → Use **Qdrant**
   - Modern, fast, easy to set up
   - Great documentation
   - Can self-host for free

3. **Want managed service** → Use **Pinecone**
   - Easiest to manage
   - No infrastructure overhead
   - Generous free tier

4. **Enterprise/Large scale** → Use **Milvus**
   - Proven at massive scale
   - Industry standard
   - Best for >10M vectors

---

## 🔧 Migration Checklist

When migrating to a vector database:

- [ ] Choose vector database based on requirements
- [ ] Set up database (Docker/Cloud)
- [ ] Install client library
- [ ] Create modified retrieval module (`hybrid_[db_name].py`)
- [ ] Create index building script (`build_[db_name].py`)
- [ ] Upload your vectors with metadata
- [ ] Test queries and performance
- [ ] Update RAG scripts to use new retriever
- [ ] Monitor costs (if using managed service)

---

## 💡 Pro Tips

1. **Start local**: Test with Docker before moving to cloud
2. **Monitor costs**: Managed services can get expensive at scale
3. **Batch operations**: Upload vectors in batches (100-1000 at a time)
4. **Metadata limits**: Most services have metadata size limits (check docs)
5. **Keep BM25**: Hybrid search still works well with vector DBs
6. **Test thoroughly**: Ensure query latency meets your SLA

---

## 📚 Additional Resources

- **Pinecone Docs**: https://docs.pinecone.io/
- **Weaviate Docs**: https://weaviate.io/developers/weaviate
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Milvus Docs**: https://milvus.io/docs

---

**Ready to scale!** 🚀

Choose the database that fits your needs and follow the integration pattern above.

