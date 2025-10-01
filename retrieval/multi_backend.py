# retrieval/multi_backend.py
"""
Multi-backend retrieval system supporting both local FAISS and vector databases.

Allows users to choose backend at runtime:
- 'faiss': Local FAISS index (default, fast)
- 'pinecone': Pinecone vector database
- 'qdrant': Qdrant vector database
- 'weaviate': Weaviate vector database
"""

import json
import numpy as np
import joblib
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Literal, Optional
from scipy.stats import zscore
import os


class MultiBackendRetriever:
    """
    Unified retrieval system with pluggable vector database backends.
    
    Supports:
    - Local FAISS (default)
    - Pinecone (managed cloud)
    - Qdrant (self-hosted or cloud)
    - Weaviate (self-hosted or cloud)
    """
    
    def __init__(self, 
                 bm25_path: str = "indexes/bm25_rank.pkl",
                 meta_path: str = "indexes/meta.jsonl",
                 vector_backend: Literal["faiss", "pinecone", "qdrant", "weaviate"] = "faiss",
                 # FAISS settings
                 faiss_dir: str = "indexes/faiss_bge_small",
                 # Pinecone settings
                 pinecone_api_key: Optional[str] = None,
                 pinecone_environment: Optional[str] = None,
                 pinecone_index_name: str = "typed-rag",
                 # Qdrant settings
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_collection_name: str = "typed-rag",
                 # Weaviate settings
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_class_name: str = "Document",
                 # Common settings
                 embedding_model: str = "BAAI/bge-small-en-v1.5"):
        
        self.bm25_path = bm25_path
        self.meta_path = meta_path
        self.vector_backend = vector_backend
        
        # Initialize components
        self._load_bm25()
        self._load_metadata()
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize vector backend
        if vector_backend == "faiss":
            self._init_faiss(faiss_dir)
        elif vector_backend == "pinecone":
            self._init_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name)
        elif vector_backend == "qdrant":
            self._init_qdrant(qdrant_url, qdrant_port, qdrant_collection_name)
        elif vector_backend == "weaviate":
            self._init_weaviate(weaviate_url, weaviate_class_name)
        else:
            raise ValueError(f"Unknown backend: {vector_backend}")
    
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
    
    def _load_metadata(self):
        """Load document metadata for result formatting."""
        try:
            self.metadata = []
            self.metadata_by_id = {}
            with open(self.meta_path, 'r') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        doc = json.loads(line)
                        self.metadata.append(doc)
                        doc_id = doc.get('id', str(idx))
                        self.metadata_by_id[doc_id] = doc
            print(f"✅ Metadata loaded: {len(self.metadata)} docs")
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
            self.metadata = []
            self.metadata_by_id = {}
    
    def _init_faiss(self, faiss_dir: str):
        """Initialize FAISS backend."""
        try:
            import faiss
            index_path = Path(faiss_dir) / "index.flatip"
            self.faiss_index = faiss.read_index(str(index_path))
            self.vector_index = self.faiss_index
            print(f"✅ FAISS loaded: {self.faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Failed to load FAISS: {e}")
            self.vector_index = None
    
    def _init_pinecone(self, api_key: Optional[str], environment: Optional[str], index_name: str):
        """Initialize Pinecone backend."""
        try:
            import pinecone
            
            # Get credentials from environment if not provided
            api_key = api_key or os.environ.get("PINECONE_API_KEY")
            environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
            
            if not api_key or not environment:
                raise ValueError("Pinecone requires API key and environment")
            
            pinecone.init(api_key=api_key, environment=environment)
            
            if index_name not in pinecone.list_indexes():
                raise ValueError(f"Pinecone index '{index_name}' not found")
            
            self.pinecone_index = pinecone.Index(index_name)
            self.vector_index = self.pinecone_index
            
            stats = self.pinecone_index.describe_index_stats()
            print(f"✅ Pinecone loaded: {stats['total_vector_count']} vectors")
        except ImportError:
            print(f"❌ Pinecone not installed. Install with: pip install pinecone-client")
            self.vector_index = None
        except Exception as e:
            print(f"❌ Failed to load Pinecone: {e}")
            self.vector_index = None
    
    def _init_qdrant(self, url: str, port: int, collection_name: str):
        """Initialize Qdrant backend."""
        try:
            from qdrant_client import QdrantClient
            
            self.qdrant_client = QdrantClient(host=url, port=port)
            self.qdrant_collection = collection_name
            self.vector_index = self.qdrant_client
            
            collection_info = self.qdrant_client.get_collection(collection_name)
            print(f"✅ Qdrant loaded: {collection_info.points_count} vectors")
        except ImportError:
            print(f"❌ Qdrant not installed. Install with: pip install qdrant-client")
            self.vector_index = None
        except Exception as e:
            print(f"❌ Failed to load Qdrant: {e}")
            self.vector_index = None
    
    def _init_weaviate(self, url: str, class_name: str):
        """Initialize Weaviate backend."""
        try:
            import weaviate
            
            self.weaviate_client = weaviate.Client(url)
            self.weaviate_class = class_name
            self.vector_index = self.weaviate_client
            
            schema = self.weaviate_client.schema.get(class_name)
            print(f"✅ Weaviate loaded: class '{class_name}'")
        except ImportError:
            print(f"❌ Weaviate not installed. Install with: pip install weaviate-client")
            self.vector_index = None
        except Exception as e:
            print(f"❌ Failed to load Weaviate: {e}")
            self.vector_index = None
    
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
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[idx])
                results.append(result)
        
        return results
    
    def _search_vector(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using vector backend (auto-detects which one)."""
        if self.vector_backend == "faiss":
            return self._search_faiss(query, k)
        elif self.vector_backend == "pinecone":
            return self._search_pinecone(query, k)
        elif self.vector_backend == "qdrant":
            return self._search_qdrant(query, k)
        elif self.vector_backend == "weaviate":
            return self._search_weaviate(query, k)
        else:
            return []
    
    def _search_faiss(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using FAISS."""
        if not self.vector_index:
            return []
        
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).astype('float32')
        
        scores, indices = self.vector_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results
    
    def _search_pinecone(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using Pinecone."""
        if not self.vector_index:
            return []
        
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        results_raw = self.vector_index.query(
            vector=query_embedding[0].tolist(),
            top_k=k,
            include_metadata=True
        )
        
        results = []
        for match in results_raw['matches']:
            doc_id = match['id']
            if doc_id in self.metadata_by_id:
                result = self.metadata_by_id[doc_id].copy()
                result["score"] = float(match['score'])
                results.append(result)
        
        return results
    
    def _search_qdrant(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using Qdrant."""
        if not self.vector_index:
            return []
        
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        results_raw = self.vector_index.search(
            collection_name=self.qdrant_collection,
            query_vector=query_embedding[0].tolist(),
            limit=k
        )
        
        results = []
        for hit in results_raw:
            doc_id = hit.payload.get('id')
            if doc_id in self.metadata_by_id:
                result = self.metadata_by_id[doc_id].copy()
                result["score"] = float(hit.score)
                results.append(result)
        
        return results
    
    def _search_weaviate(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using Weaviate."""
        if not self.vector_index:
            return []
        
        results_raw = (
            self.vector_index.query
            .get(self.weaviate_class, ["id", "title", "url", "chunk_text"])
            .with_near_text({"concepts": [query]})
            .with_limit(k)
            .do()
        )
        
        results = []
        for item in results_raw['data']['Get'][self.weaviate_class]:
            doc_id = item.get('id')
            if doc_id in self.metadata_by_id:
                result = self.metadata_by_id[doc_id].copy()
                result["score"] = 1.0  # Weaviate doesn't return scores by default
                results.append(result)
        
        return results
    
    def _search_hybrid(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using hybrid BM25 + Vector with z-score normalization."""
        if not self.bm25_ranker or not self.vector_index:
            print("⚠️  Hybrid mode requires both BM25 and vector backend.")
            if self.bm25_ranker:
                return self._search_bm25(query, k)
            elif self.vector_index:
                return self._search_vector(query, k)
            else:
                return []
        
        candidate_k = min(k * 3, len(self.metadata))
        
        # BM25 search
        query_tokens = self._tokenize_query(query)
        bm25_scores = self.bm25_ranker.get_scores(query_tokens)
        
        # Vector search
        vector_results = self._search_vector(query, candidate_k)
        
        # Create score dictionaries
        n_docs = len(self.metadata)
        combined_bm25 = np.zeros(n_docs)
        combined_vector = np.zeros(n_docs)
        
        # Fill BM25 scores
        combined_bm25[:len(bm25_scores)] = bm25_scores
        
        # Fill vector scores
        for result in vector_results:
            # Find index by matching ID
            doc_id = result.get('id')
            for idx, meta in enumerate(self.metadata):
                if meta.get('id') == doc_id:
                    combined_vector[idx] = result.get('score', 0)
                    break
        
        # Z-score normalization
        z_bm25 = zscore(combined_bm25) if np.std(combined_bm25) > 0 else combined_bm25
        z_vector = zscore(combined_vector) if np.std(combined_vector) > 0 else combined_vector
        
        z_bm25 = np.nan_to_num(z_bm25)
        z_vector = np.nan_to_num(z_vector)
        
        # Combine scores
        hybrid_scores = z_bm25 + z_vector
        
        # Get top-k
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(hybrid_scores[idx])
                result["bm25_score"] = float(combined_bm25[idx])
                result["vector_score"] = float(combined_vector[idx])
                results.append(result)
        
        return results
    
    def retrieve(self, 
                query: str, 
                k: int = 5, 
                mode: Literal["bm25", "vector", "hybrid"] = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a given query.
        
        Args:
            query: Search query string
            k: Number of results to return
            mode: Search mode - 'bm25', 'vector', or 'hybrid'
            
        Returns:
            List of documents with metadata and scores
        """
        if mode == "bm25":
            return self._search_bm25(query, k)
        elif mode == "vector":
            return self._search_vector(query, k)
        elif mode == "hybrid":
            return self._search_hybrid(query, k)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bm25', 'vector', or 'hybrid'")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of all components."""
        return {
            "bm25_loaded": self.bm25_ranker is not None,
            "vector_backend": self.vector_backend,
            "vector_loaded": self.vector_index is not None,
            "metadata_loaded": len(self.metadata) > 0,
            "embedding_model_loaded": self.embedding_model is not None
        }


# Convenience function for quick usage
def search(query: str, 
          k: int = 5, 
          mode: str = "hybrid",
          backend: str = "faiss",
          index_dir: str = "indexes") -> List[Dict[str, Any]]:
    """Quick search function with configurable backend."""
    retriever = MultiBackendRetriever(
        bm25_path=f"{index_dir}/bm25_rank.pkl",
        meta_path=f"{index_dir}/meta.jsonl",
        vector_backend=backend,
        faiss_dir=f"{index_dir}/faiss_bge_small"
    )
    return retriever.retrieve(query, k, mode)


if __name__ == "__main__":
    import sys
    
    # Quick test
    backend = sys.argv[1] if len(sys.argv) > 1 else "faiss"
    print(f"Testing with backend: {backend}")
    
    retriever = MultiBackendRetriever(vector_backend=backend)
    print("Health check:", retriever.health_check())
    
    if retriever.health_check()["metadata_loaded"]:
        results = retriever.retrieve("test query", k=3, mode="hybrid")
        print(f"\nFound {len(results)} results")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('title', 'No title')} (score: {result.get('score', 0):.3f})")

