# src/retrieval/vector_store.py

"""
Vector store implementation using FAISS for semantic search.
Handles index building, persistence, and querying operations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.is_built = False
        
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")
    
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Build FAISS index from embeddings and metadata.
        
        Args:
            embeddings: numpy array of embeddings (n_items x embedding_dim)
            metadata: List of metadata dictionaries for each embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        logger.info(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # Create appropriate index based on type
        if self.index_type == "flat":
            # Simple flat index (exact search)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            nlist = min(100, max(10, len(embeddings) // 10))  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index if we have enough data
            if len(embeddings) >= nlist:
                self.index.train(embeddings)
        else:
            # Default to flat index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity (if using inner product)
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = metadata.copy()
        self.is_built = True
        
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add new vectors to existing index.
        
        Args:
            embeddings: numpy array of new embeddings
            metadata: List of metadata for new embeddings
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add metadata
        self.metadata.extend(metadata)
        
        logger.info(f"✅ Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding (1 x embedding_dim)
            k: Number of results to return
            
        Returns:
            Tuple of (similarity_scores, metadata_list)
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Ensure query is float32 and normalized
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than available
        similarities, indices = self.index.search(query_embedding, k)
        
        # Get results
        scores = similarities[0].tolist()
        results = []
        
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return scores, results
    
    def save_index(self, save_dir: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            save_dir: Directory to save index and metadata
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Cannot save empty index.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "num_vectors": self.index.ntotal,
            "saved_at": datetime.now().isoformat()
        }
        
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Saved FAISS index to {save_dir}")
    
    def load_index(self, load_dir: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            load_dir: Directory containing saved index and metadata
        """
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index directory not found: {load_dir}")
        
        # Load configuration
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.embedding_dim = config["embedding_dim"]
            self.index_type = config["index_type"]
        
        # Load FAISS index
        index_path = load_path / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.is_built = True
        
        logger.info(f"✅ Loaded FAISS index from {load_dir} ({self.index.ntotal} vectors)")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_built:
            return {
                "is_built": False,
                "num_vectors": 0,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }
        
        return {
            "is_built": True,
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metadata_count": len(self.metadata)
        }


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: FAISS vector store for semantic search
            embedding_manager: Embedding manager for encoding queries
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        # Initialize TF-IDF for keyword search
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build TF-IDF index for keyword search."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if not self.vector_store.is_built:
                logger.warning("Vector store not built. Keyword index will be empty.")
                return
            
            # Extract text from metadata for TF-IDF
            self.documents = []
            for meta in self.vector_store.metadata:
                # Combine relevant text fields
                text_parts = []
                
                if meta.get("text"):
                    text_parts.append(meta["text"])
                
                if meta.get("resolution"):
                    text_parts.append(meta["resolution"])
                
                if meta.get("resolution_steps"):
                    text_parts.append(meta["resolution_steps"])
                
                if meta.get("category"):
                    text_parts.append(meta["category"])
                
                combined_text = " ".join(text_parts)
                self.documents.append(combined_text)
            
            if self.documents:
                # Build TF-IDF index
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
                logger.info(f"✅ Built TF-IDF index with {len(self.documents)} documents")
            
        except ImportError:
            logger.warning("scikit-learn not available. Keyword search disabled.")
    
    def semantic_search(self, query_text: str, k: int = 10) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            Tuple of (similarity_scores, metadata_list)
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.encode_single_text(query_text)
        
        # Search vector store
        return self.vector_store.search(query_embedding, k)
    
    def keyword_search(self, query_text: str, k: int = 10) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Perform keyword search using TF-IDF.
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            Tuple of (similarity_scores, metadata_list)
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not available")
            return [], []
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k results
            top_indices = similarities.argsort()[-k:][::-1]
            top_scores = similarities[top_indices].tolist()
            
            # Get corresponding metadata
            results = []
            for idx in top_indices:
                if 0 <= idx < len(self.vector_store.metadata):
                    results.append(self.vector_store.metadata[idx])
            
            return top_scores, results
            
        except ImportError:
            logger.warning("scikit-learn not available for keyword search")
            return [], []
    
    def hybrid_search(self, query_text: str, k: int = 10, 
                     semantic_weight: float = 0.7, keyword_weight: float = 0.3,
                     rerank_by_success_rate: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword results.
        
        Args:
            query_text: Query text
            k: Number of final results to return
            semantic_weight: Weight for semantic search scores
            keyword_weight: Weight for keyword search scores
            rerank_by_success_rate: Whether to rerank by resolution success rate
            
        Returns:
            List of result dictionaries with combined scores
        """
        # Perform both searches
        sem_scores, sem_results = self.semantic_search(query_text, k * 2)
        kw_scores, kw_results = self.keyword_search(query_text, k * 2)
        
        # Combine results
        combined_results = {}
        
        # Add semantic results
        for score, result in zip(sem_scores, sem_results):
            key = result.get("resolution_id", str(hash(str(result))))
            combined_results[key] = {
                **result,
                "semantic_score": score,
                "keyword_score": 0.0,
                "combined_score": semantic_weight * score
            }
        
        # Add keyword results
        for score, result in zip(kw_scores, kw_results):
            key = result.get("resolution_id", str(hash(str(result))))
            
            if key in combined_results:
                # Update existing result
                combined_results[key]["keyword_score"] = score
                combined_results[key]["combined_score"] += keyword_weight * score
            else:
                # Add new result
                combined_results[key] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": score,
                    "combined_score": keyword_weight * score
                }
        
        # Convert to list and sort by combined score
        final_results = list(combined_results.values())
        
        # Optional reranking by success rate
        if rerank_by_success_rate:
            for result in final_results:
                success_rate = result.get("success_rate", 0.5)
                # Boost score based on success rate
                result["combined_score"] *= (0.5 + success_rate)
        
        # Sort by combined score (descending)
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top-k results
        return final_results[:k]


if __name__ == "__main__":
    # Test the vector store
    from .embedding_manager import EmbeddingManager, generate_mock_resolutions
    
    try:
        # Initialize components
        embedding_manager = EmbeddingManager()
        embedding_manager.load_model()
        
        # Generate test data
        resolutions = generate_mock_resolutions(20)
        
        # Prepare texts and embeddings
        resolution_texts = []
        metadata = []
        
        for res in resolutions:
            text = embedding_manager.prepare_resolution_text(res)
            resolution_texts.append(text)
            
            metadata.append({
                **res,
                "text": text
            })
        
        # Generate embeddings
        embeddings = embedding_manager.encode_texts(resolution_texts)
        
        # Build vector store
        vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        vector_store.build_index(embeddings, metadata)
        
        # Test search
        query = "authentication login failed timeout"
        query_embedding = embedding_manager.encode_single_text(query)
        scores, results = vector_store.search(query_embedding, k=5)
        
        print(f"Query: {query}")
        print("\nTop 5 results:")
        for i, (score, result) in enumerate(zip(scores, results)):
            print(f"{i+1}. Score: {score:.3f} | {result['resolution'][:100]}...")
        
        # Test hybrid retriever
        hybrid = HybridRetriever(vector_store, embedding_manager)
        hybrid_results = hybrid.hybrid_search(query, k=3)
        
        print("\nHybrid search results:")
        for i, result in enumerate(hybrid_results):
            print(f"{i+1}. Combined: {result['combined_score']:.3f} | "
                  f"Semantic: {result['semantic_score']:.3f} | "
                  f"Keyword: {result['keyword_score']:.3f}")
            print(f"   {result['resolution'][:100]}...")
        
        # Test save/load
        vector_store.save_index("test_vector_store")
        print("\n✅ Vector store saved and tested successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to install required packages:")
        print("   pip install sentence-transformers faiss-cpu scikit-learn")