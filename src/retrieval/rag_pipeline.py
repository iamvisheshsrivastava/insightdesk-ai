# src/retrieval/rag_pipeline.py

"""
Main RAG (Retrieval-Augmented Generation) pipeline implementation.
Provides high-level interface for building and querying the knowledge base.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from .embedding_manager import EmbeddingManager, generate_mock_resolutions
from .vector_store import VectorStore, HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline for intelligent support system."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_dir: str = "vector_store",
                 index_type: str = "flat"):
        """
        Initialize RAG pipeline.
        
        Args:
            model_name: Sentence transformer model name
            vector_store_dir: Directory for persisting vector store
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        """
        self.model_name = model_name
        self.vector_store_dir = Path(vector_store_dir)
        self.index_type = index_type
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(model_name)
        self.vector_store = None
        self.hybrid_retriever = None
        
        # State tracking
        self.is_initialized = False
        self.knowledge_base_size = 0
        
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the RAG pipeline.
        
        Args:
            force_rebuild: Whether to force rebuild even if existing index found
        """
        logger.info("Initializing RAG pipeline...")
        
        # Load embedding model
        self.embedding_manager.load_model()
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_dim=self.embedding_manager.embedding_dim,
            index_type=self.index_type
        )
        
        # Try to load existing index
        index_path = self.vector_store_dir / "faiss_index"
        
        if index_path.exists() and not force_rebuild:
            try:
                self.vector_store.load_index(str(index_path))
                self.knowledge_base_size = self.vector_store.index.ntotal
                logger.info(f"✅ Loaded existing index with {self.knowledge_base_size} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                logger.info("Will build new index...")
                force_rebuild = True
        
        if not index_path.exists() or force_rebuild:
            # Build new index with mock data for demonstration
            self._build_initial_index()
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.embedding_manager)
        
        self.is_initialized = True
        logger.info("✅ RAG pipeline initialized successfully")
    
    def _build_initial_index(self):
        """Build initial index with mock resolution data."""
        logger.info("Building initial knowledge base with mock data...")
        
        # Generate mock resolutions
        mock_resolutions = generate_mock_resolutions(200)
        
        # Prepare embeddings
        resolution_texts = []
        metadata = []
        
        for res in mock_resolutions:
            text = self.embedding_manager.prepare_resolution_text(res)
            resolution_texts.append(text)
            
            metadata.append({
                **res,
                "text": text,
                "type": "resolution"
            })
        
        # Generate embeddings
        embeddings = self.embedding_manager.encode_texts(resolution_texts)
        
        # Build index
        self.vector_store.build_index(embeddings, metadata)
        
        # Save index
        self.save_index()
        
        self.knowledge_base_size = len(mock_resolutions)
        logger.info(f"✅ Built initial index with {self.knowledge_base_size} resolutions")
    
    def add_ticket_resolutions(self, tickets_data: List[Dict[str, Any]]):
        """
        Add ticket resolution data to the knowledge base.
        
        Args:
            tickets_data: List of ticket dictionaries with resolution information
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"Adding {len(tickets_data)} ticket resolutions to knowledge base...")
        
        # Prepare texts and metadata
        texts = []
        metadata = []
        
        for ticket in tickets_data:
            # Create resolution entry from ticket
            if ticket.get("resolution") or ticket.get("status") == "resolved":
                text = self.embedding_manager.prepare_ticket_text(ticket)
                texts.append(text)
                
                resolution_meta = {
                    "ticket_id": ticket.get("ticket_id"),
                    "category": ticket.get("category"),
                    "product": ticket.get("product"),
                    "resolution": ticket.get("resolution", "Resolution applied"),
                    "resolution_steps": ticket.get("resolution_steps", ""),
                    "success_rate": 0.8,  # Default success rate
                    "usage_count": 1,
                    "text": text,
                    "type": "ticket_resolution",
                    "created_date": ticket.get("created_date"),
                    "resolved_date": ticket.get("resolved_date")
                }
                
                metadata.append(resolution_meta)
        
        if texts:
            # Generate embeddings
            embeddings = self.embedding_manager.encode_texts(texts)
            
            # Add to vector store
            self.vector_store.add_vectors(embeddings, metadata)
            
            # Update hybrid retriever
            self.hybrid_retriever._build_keyword_index()
            
            self.knowledge_base_size += len(texts)
            logger.info(f"✅ Added {len(texts)} ticket resolutions. Total: {self.knowledge_base_size}")
    
    def add_kb_articles(self, kb_articles: List[Dict[str, Any]]):
        """
        Add knowledge base articles to the vector store.
        
        Args:
            kb_articles: List of KB article dictionaries
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"Adding {len(kb_articles)} KB articles to knowledge base...")
        
        # Prepare texts and metadata
        texts = []
        metadata = []
        
        for article in kb_articles:
            # Combine article content
            text_parts = []
            
            if article.get("title"):
                text_parts.append(f"Title: {article['title']}")
            
            if article.get("content"):
                text_parts.append(f"Content: {article['content']}")
            
            if article.get("tags"):
                text_parts.append(f"Tags: {', '.join(article['tags'])}")
            
            if article.get("category"):
                text_parts.append(f"Category: {article['category']}")
            
            text = " | ".join(text_parts)
            texts.append(text)
            
            article_meta = {
                "article_id": article.get("article_id"),
                "title": article.get("title"),
                "category": article.get("category"),
                "tags": article.get("tags", []),
                "resolution": article.get("content", ""),
                "success_rate": article.get("helpful_score", 0.7),
                "usage_count": article.get("view_count", 0),
                "text": text,
                "type": "kb_article",
                "created_date": article.get("created_date"),
                "last_updated": article.get("last_updated")
            }
            
            metadata.append(article_meta)
        
        if texts:
            # Generate embeddings
            embeddings = self.embedding_manager.encode_texts(texts)
            
            # Add to vector store
            self.vector_store.add_vectors(embeddings, metadata)
            
            # Update hybrid retriever
            self.hybrid_retriever._build_keyword_index()
            
            self.knowledge_base_size += len(texts)
            logger.info(f"✅ Added {len(texts)} KB articles. Total: {self.knowledge_base_size}")
    
    def query_solutions(self, 
                       ticket_data: Dict[str, Any], 
                       k: int = 10,
                       search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Query for solutions based on ticket data.
        
        Args:
            ticket_data: Ticket information dictionary
            k: Number of solutions to return
            search_type: Type of search ('semantic', 'keyword', 'hybrid')
            
        Returns:
            List of solution dictionaries with similarity scores
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Prepare query text from ticket
        query_text = self.embedding_manager.prepare_ticket_text(ticket_data)
        
        logger.info(f"Querying solutions for: {query_text[:100]}...")
        
        if search_type == "semantic":
            scores, results = self.hybrid_retriever.semantic_search(query_text, k)
            
            # Add score information
            for i, result in enumerate(results):
                result["similarity_score"] = scores[i]
                result["search_type"] = "semantic"
            
            return results
            
        elif search_type == "keyword":
            scores, results = self.hybrid_retriever.keyword_search(query_text, k)
            
            # Add score information
            for i, result in enumerate(results):
                result["similarity_score"] = scores[i]
                result["search_type"] = "keyword"
            
            return results
            
        else:  # hybrid
            results = self.hybrid_retriever.hybrid_search(query_text, k)
            
            # Add search type information
            for result in results:
                result["similarity_score"] = result["combined_score"]
                result["search_type"] = "hybrid"
            
            return results
    
    def update_success_rates(self, feedback_data: List[Dict[str, Any]]):
        """
        Update success rates based on user feedback.
        
        Args:
            feedback_data: List of feedback dictionaries with resolution_id and success flag
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"Updating success rates based on {len(feedback_data)} feedback entries...")
        
        # Update metadata
        updated_count = 0
        
        for feedback in feedback_data:
            resolution_id = feedback.get("resolution_id")
            was_successful = feedback.get("success", False)
            
            # Find and update corresponding metadata
            for meta in self.vector_store.metadata:
                if meta.get("resolution_id") == resolution_id:
                    # Simple exponential moving average update
                    current_rate = meta.get("success_rate", 0.5)
                    current_count = meta.get("usage_count", 1)
                    
                    # Update usage count
                    meta["usage_count"] = current_count + 1
                    
                    # Update success rate (weighted average)
                    weight = 0.1  # Learning rate
                    new_rate = current_rate * (1 - weight) + (1.0 if was_successful else 0.0) * weight
                    meta["success_rate"] = max(0.1, min(0.99, new_rate))  # Clamp between 0.1 and 0.99
                    
                    updated_count += 1
                    break
        
        logger.info(f"✅ Updated success rates for {updated_count} resolutions")
    
    def save_index(self):
        """Save the vector store index to disk."""
        if not self.is_initialized or not self.vector_store.is_built:
            logger.warning("No index to save")
            return
        
        index_path = self.vector_store_dir / "faiss_index"
        self.vector_store.save_index(str(index_path))
        
        # Save pipeline metadata
        pipeline_meta = {
            "model_name": self.model_name,
            "index_type": self.index_type,
            "knowledge_base_size": self.knowledge_base_size,
            "last_saved": datetime.now().isoformat()
        }
        
        meta_path = self.vector_store_dir / "pipeline_metadata.json"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        with open(meta_path, 'w') as f:
            json.dump(pipeline_meta, f, indent=2)
        
        logger.info(f"✅ Saved RAG pipeline to {self.vector_store_dir}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        vector_stats = self.vector_store.get_index_stats()
        
        # Count different types of content
        type_counts = {}
        for meta in self.vector_store.metadata:
            content_type = meta.get("type", "unknown")
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        return {
            "status": "initialized",
            "model_name": self.model_name,
            "knowledge_base_size": self.knowledge_base_size,
            "vector_store_stats": vector_stats,
            "content_types": type_counts,
            "embedding_dimension": self.embedding_manager.embedding_dim
        }
    
    # Future extensibility functions (placeholders)
    
    def setup_graph_rag(self, neo4j_config: Optional[Dict[str, Any]] = None):
        """
        TODO: Setup Graph-RAG integration with Neo4j.
        
        Args:
            neo4j_config: Neo4j connection configuration
        """
        logger.info("TODO: Graph-RAG integration with Neo4j")
        # Placeholder for Neo4j graph database integration
        # This would create knowledge graphs from ticket relationships
        pass
    
    def rerank_with_category_predictions(self, results: List[Dict[str, Any]], 
                                       predicted_category: str) -> List[Dict[str, Any]]:
        """
        TODO: Rerank results based on category predictions from ML models.
        
        Args:
            results: List of retrieved results
            predicted_category: Predicted category from classification model
            
        Returns:
            Reranked results list
        """
        logger.info(f"TODO: Rerank results with predicted category: {predicted_category}")
        # Placeholder for reranking based on ML model predictions
        # This would boost results that match the predicted category
        return results


def build_index_from_tickets(tickets_file: str, 
                           output_dir: str = "vector_store",
                           force_rebuild: bool = False) -> RAGPipeline:
    """
    Utility function to build RAG index from tickets file.
    
    Args:
        tickets_file: Path to tickets JSON file
        output_dir: Output directory for vector store
        force_rebuild: Whether to force rebuild
        
    Returns:
        Initialized RAG pipeline
    """
    logger.info(f"Building RAG index from {tickets_file}...")
    
    # Initialize pipeline
    pipeline = RAGPipeline(vector_store_dir=output_dir)
    pipeline.initialize(force_rebuild=force_rebuild)
    
    # Load tickets data if file exists
    tickets_path = Path(tickets_file)
    if tickets_path.exists():
        with open(tickets_path, 'r') as f:
            tickets_data = json.load(f)
        
        # Add ticket resolutions to knowledge base
        pipeline.add_ticket_resolutions(tickets_data)
        
        # Save the updated index
        pipeline.save_index()
        
        logger.info(f"✅ Built RAG index with {len(tickets_data)} tickets")
    else:
        logger.warning(f"Tickets file not found: {tickets_file}")
        logger.info("Using mock data for demonstration")
    
    return pipeline


if __name__ == "__main__":
    # Test the RAG pipeline
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        pipeline.initialize()
        
        # Test query
        test_ticket = {
            "subject": "Cannot login to application",
            "description": "User gets timeout error when trying to log in",
            "error_logs": "Authentication service timeout after 30 seconds",
            "product": "web_application",
            "category": "authentication"
        }
        
        # Query solutions
        solutions = pipeline.query_solutions(test_ticket, k=5)
        
        print(f"Query: {test_ticket['subject']}")
        print(f"\nFound {len(solutions)} solutions:")
        
        for i, solution in enumerate(solutions):
            print(f"\n{i+1}. Score: {solution['similarity_score']:.3f}")
            print(f"   Type: {solution.get('type', 'unknown')}")
            print(f"   Category: {solution.get('category', 'N/A')}")
            print(f"   Resolution: {solution.get('resolution', 'N/A')[:100]}...")
            print(f"   Success Rate: {solution.get('success_rate', 0):.2f}")
        
        # Test stats
        stats = pipeline.get_stats()
        print(f"\nPipeline Stats:")
        print(f"- Knowledge Base Size: {stats['knowledge_base_size']}")
        print(f"- Content Types: {stats['content_types']}")
        print(f"- Embedding Dimension: {stats['embedding_dimension']}")
        
        print("\n✅ RAG pipeline test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to install required packages:")
        print("   pip install sentence-transformers faiss-cpu scikit-learn")