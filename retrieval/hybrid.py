# retrieval/hybrid.py
import json
import numpy as np
import joblib
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Literal
from scipy.stats import zscore


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 (keyword) and FAISS (semantic) search.
    
    Supports three modes:
    - 'bm25': Keyword-based retrieval only
    - 'faiss': Vector-based retrieval only  
    - 'hybrid': Combined scoring with z-score normalization
    """
    
    def __init__(self, 
                 bm25_path: str = "indexes/bm25_rank.pkl",
                 faiss_dir: str = "indexes/faiss_bge_small",
                 meta_path: str = "indexes/meta.jsonl"):
        
        self.bm25_path = bm25_path
        self.faiss_dir = faiss_dir
        self.meta_path = meta_path
        
        # Initialize components
        self._load_bm25()
        self._load_faiss()
        self._load_metadata()
    
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
    
    def _load_faiss(self):
        """Load FAISS index and embedding model."""
        try:
            index_path = Path(self.faiss_dir) / "index.flatip"
            model_path = Path(self.faiss_dir) / "model.txt"
            
            self.faiss_index = faiss.read_index(str(index_path))
            
            with open(model_path, 'r') as f:
                model_name = f.read().strip()
            self.embedding_model = SentenceTransformer(model_name)
            
            print(f"✅ FAISS loaded: {self.faiss_index.ntotal} vectors, model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load FAISS: {e}")
            self.faiss_index = None
            self.embedding_model = None
    
    def _load_metadata(self):
        """Load document metadata for result formatting."""
        try:
            self.metadata = []
            with open(self.meta_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.metadata.append(json.loads(line))
            print(f"✅ Metadata loaded: {len(self.metadata)} docs")
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
            self.metadata = []
    
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
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[idx])
                results.append(result)
        
        return results
    
    def _search_faiss(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using FAISS only."""
        if not self.faiss_index or not self.embedding_model:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).astype('float32')
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):  # -1 means no result found
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results
    
    def _search_hybrid(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search using hybrid BM25 + FAISS with z-score normalization."""
        if not self.bm25_ranker or not self.faiss_index:
            print("⚠️  Hybrid mode requires both BM25 and FAISS. Falling back to available method.")
            if self.bm25_ranker:
                return self._search_bm25(query, k)
            elif self.faiss_index:
                return self._search_faiss(query, k)
            else:
                return []
        
        # Get larger candidate set for better score normalization
        candidate_k = min(k * 3, len(self.metadata))
        
        # BM25 search
        query_tokens = self._tokenize_query(query)
        bm25_scores = self.bm25_ranker.get_scores(query_tokens)
        
        # FAISS search  
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype('float32')
        faiss_scores, faiss_indices = self.faiss_index.search(query_embedding, candidate_k)
        
        # Create score arrays aligned by document index
        n_docs = len(self.metadata)
        combined_bm25 = np.zeros(n_docs)
        combined_faiss = np.zeros(n_docs)
        
        # Fill BM25 scores
        combined_bm25[:len(bm25_scores)] = bm25_scores
        
        # Fill FAISS scores
        for score, idx in zip(faiss_scores[0], faiss_indices[0]):
            if idx != -1 and idx < n_docs:
                combined_faiss[idx] = score
        
        # Z-score normalization (handle edge case where std=0)
        z_bm25 = zscore(combined_bm25) if np.std(combined_bm25) > 0 else combined_bm25
        z_faiss = zscore(combined_faiss) if np.std(combined_faiss) > 0 else combined_faiss
        
        # Replace NaN with 0 (can happen with zscore)
        z_bm25 = np.nan_to_num(z_bm25)
        z_faiss = np.nan_to_num(z_faiss)
        
        # Combine scores (equal weighting)
        hybrid_scores = z_bm25 + z_faiss
        
        # Get top-k results
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(hybrid_scores[idx])
                result["bm25_score"] = float(combined_bm25[idx])
                result["faiss_score"] = float(combined_faiss[idx])
                results.append(result)
        
        return results
    
    def retrieve(self, 
                query: str, 
                k: int = 5, 
                mode: Literal["bm25", "faiss", "hybrid"] = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a given query.
        
        Args:
            query: Search query string
            k: Number of results to return
            mode: Search mode - 'bm25', 'faiss', or 'hybrid'
            
        Returns:
            List of documents with metadata and scores
        """
        if mode == "bm25":
            return self._search_bm25(query, k)
        elif mode == "faiss":
            return self._search_faiss(query, k)
        elif mode == "hybrid":
            return self._search_hybrid(query, k)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bm25', 'faiss', or 'hybrid'")
    
    def health_check(self) -> Dict[str, bool]:
        """Check the health of all components."""
        return {
            "bm25_loaded": self.bm25_ranker is not None,
            "faiss_loaded": self.faiss_index is not None,
            "metadata_loaded": len(self.metadata) > 0,
            "embedding_model_loaded": self.embedding_model is not None
        }


# Convenience function for quick usage
def search(query: str, k: int = 5, mode: str = "hybrid") -> List[Dict[str, Any]]:
    """Quick search function using default paths."""
    retriever = HybridRetriever()
    return retriever.retrieve(query, k, mode)


if __name__ == "__main__":
    # Quick test
    retriever = HybridRetriever()
    print("Health check:", retriever.health_check())
    
    # Test query
    if retriever.health_check()["metadata_loaded"]:
        results = retriever.retrieve("test query", k=3, mode="hybrid")
        print(f"Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('title', 'No title')} (score: {result.get('score', 0):.3f})")
