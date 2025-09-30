# src/retrieval/hybrid_rag_pipeline.py

"""
Hybrid RAG Pipeline combining FAISS semantic search with Neo4j graph retrieval.

This module enhances the existing RAG system with graph-based knowledge
retrieval and provides intelligent ranking of results.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .rag_pipeline import RAGPipeline
from .graph_manager import Neo4jGraphManager, GraphResult

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search combining semantic and graph retrieval."""
    semantic_results: List[Dict[str, Any]]
    graph_results: GraphResult
    combined_results: List[Dict[str, Any]]
    ranking_scores: Dict[str, float]
    search_strategy: str
    total_time_ms: float


@dataclass
class RelevanceWeights:
    """Weights for different relevance factors."""
    semantic_similarity: float = 0.4
    graph_relevance: float = 0.3
    success_rate: float = 0.2
    recency: float = 0.1


class HybridRAGPipeline:
    """Enhanced RAG pipeline with graph-based retrieval."""
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        graph_manager: Optional[Neo4jGraphManager] = None,
        relevance_weights: Optional[RelevanceWeights] = None
    ):
        """Initialize hybrid RAG pipeline."""
        self.rag_pipeline = rag_pipeline
        self.graph_manager = graph_manager
        self.relevance_weights = relevance_weights or RelevanceWeights()
        
        # Initialize components if not provided
        if not self.rag_pipeline:
            try:
                self.rag_pipeline = RAGPipeline()
                self.rag_pipeline.initialize()
                logger.info("✅ RAG pipeline initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize RAG pipeline: {e}")
                self.rag_pipeline = None
        
        if not self.graph_manager:
            try:
                self.graph_manager = Neo4jGraphManager()
                logger.info("✅ Graph manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize graph manager: {e}")
                self.graph_manager = None
        
        # Validate components
        self.semantic_available = self.rag_pipeline is not None
        self.graph_available = self.graph_manager is not None
        
        logger.info(f"Hybrid RAG Pipeline initialized - "
                   f"Semantic: {self.semantic_available}, Graph: {self.graph_available}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_semantic: bool = True,
        use_graph: bool = True,
        rerank: bool = True
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining semantic and graph retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_semantic: Whether to use semantic search
            use_graph: Whether to use graph search
            rerank: Whether to rerank combined results
            
        Returns:
            HybridSearchResult with combined results
        """
        start_time = datetime.now()
        
        semantic_results = []
        graph_results = GraphResult([], [], [], [], 0.0, 0.0)
        
        # Determine search strategy
        if use_semantic and use_graph and self.semantic_available and self.graph_available:
            search_strategy = "hybrid"
        elif use_semantic and self.semantic_available:
            search_strategy = "semantic_only"
        elif use_graph and self.graph_available:
            search_strategy = "graph_only"
        else:
            search_strategy = "fallback"
        
        # Perform semantic search
        if use_semantic and self.semantic_available:
            try:
                semantic_results = self._perform_semantic_search(query, top_k)
                logger.debug(f"Semantic search returned {len(semantic_results)} results")
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        
        # Perform graph search
        if use_graph and self.graph_available:
            try:
                graph_results = self._perform_graph_search(query, top_k)
                logger.debug(f"Graph search returned {len(graph_results.paths)} paths")
            except Exception as e:
                logger.error(f"Graph search failed: {e}")
        
        # Combine and rank results
        if rerank:
            combined_results, ranking_scores = self._combine_and_rank_results(
                semantic_results, graph_results, query, top_k
            )
        else:
            combined_results = self._simple_combine_results(
                semantic_results, graph_results, top_k
            )
            ranking_scores = {}
        
        # Calculate total time
        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return HybridSearchResult(
            semantic_results=semantic_results,
            graph_results=graph_results,
            combined_results=combined_results,
            ranking_scores=ranking_scores,
            search_strategy=search_strategy,
            total_time_ms=total_time_ms
        )
    
    def _perform_semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search using FAISS."""
        if not self.rag_pipeline:
            return []
        
        # Use existing RAG pipeline search
        search_results = self.rag_pipeline.search_similar_solutions(query, top_k=top_k)
        
        # Convert to standardized format
        results = []
        for result in search_results:
            results.append({
                "id": result.get("ticket_id", ""),
                "title": result.get("subject", ""),
                "content": result.get("description", ""),
                "resolution": result.get("resolution", ""),
                "category": result.get("category", ""),
                "similarity_score": result.get("similarity_score", 0.0),
                "source": "semantic",
                "metadata": {
                    "priority": result.get("priority", "medium"),
                    "resolved": result.get("resolved", False),
                    "resolution_time": result.get("resolution_time_hours", 0)
                }
            })
        
        return results
    
    def _perform_graph_search(self, query: str, top_k: int) -> GraphResult:
        """Perform graph search using Neo4j."""
        if not self.graph_manager:
            return GraphResult([], [], [], [], 0.0, 0.0)
        
        return self.graph_manager.query_graph(query, limit=top_k)
    
    def _combine_and_rank_results(
        self,
        semantic_results: List[Dict[str, Any]],
        graph_results: GraphResult,
        query: str,
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Combine and rank results from both sources."""
        combined_results = []
        ranking_scores = {}
        
        # Convert graph results to standardized format
        graph_converted = self._convert_graph_results(graph_results)
        
        # Create unified result set
        all_results = semantic_results + graph_converted
        
        # Remove duplicates based on content similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Calculate comprehensive ranking scores
        for result in unique_results:
            score = self._calculate_comprehensive_score(result, graph_results, query)
            result["combined_score"] = score
            ranking_scores[result.get("id", "")] = score
        
        # Sort by combined score
        unique_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top_k results
        return unique_results[:top_k], ranking_scores
    
    def _simple_combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        graph_results: GraphResult,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Simple combination without reranking."""
        combined = []
        
        # Add semantic results first
        combined.extend(semantic_results[:top_k//2])
        
        # Add graph results
        graph_converted = self._convert_graph_results(graph_results)
        combined.extend(graph_converted[:top_k//2])
        
        return combined[:top_k]
    
    def _convert_graph_results(self, graph_results: GraphResult) -> List[Dict[str, Any]]:
        """Convert graph results to standardized format."""
        converted = []
        
        for path in graph_results.paths:
            issue = path.get("issue", {})
            resolution = path.get("resolution", {})
            product = path.get("product", {})
            
            converted.append({
                "id": f"graph_{issue.get('id', '')}_to_{resolution.get('id', '')}",
                "title": f"{product.get('name', '')} - {issue.get('type', '')}",
                "content": issue.get("description", ""),
                "resolution": resolution.get("description", ""),
                "category": issue.get("category", ""),
                "similarity_score": graph_results.relevance_score,
                "source": "graph",
                "metadata": {
                    "product": product.get("name", ""),
                    "issue_type": issue.get("type", ""),
                    "resolution_type": resolution.get("type", ""),
                    "success_rate": resolution.get("success_rate", 0.0),
                    "avg_resolution_time": resolution.get("avg_resolution_time", 0),
                    "related_tickets": len(path.get("related_tickets", []))
                }
            })
        
        return converted
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Create content hash for deduplication
            content_key = self._create_content_hash(result)
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _create_content_hash(self, result: Dict[str, Any]) -> str:
        """Create hash for content deduplication."""
        # Use title + first 100 chars of content for comparison
        title = result.get("title", "").lower().strip()
        content = result.get("content", "").lower().strip()[:100]
        return f"{title}_{content}".replace(" ", "_")
    
    def _calculate_comprehensive_score(
        self,
        result: Dict[str, Any],
        graph_results: GraphResult,
        query: str
    ) -> float:
        """Calculate comprehensive ranking score."""
        score = 0.0
        weights = self.relevance_weights
        
        # Semantic similarity score
        semantic_score = result.get("similarity_score", 0.0)
        score += weights.semantic_similarity * semantic_score
        
        # Graph relevance score
        if result.get("source") == "graph":
            graph_score = graph_results.relevance_score
        else:
            # Calculate graph relevance for semantic results
            graph_score = self._calculate_graph_relevance_for_semantic(result)
        
        score += weights.graph_relevance * graph_score
        
        # Success rate score
        success_rate = result.get("metadata", {}).get("success_rate", 0.5)
        score += weights.success_rate * success_rate
        
        # Recency score (if timestamp available)
        recency_score = self._calculate_recency_score(result)
        score += weights.recency * recency_score
        
        return min(score, 1.0)  # Normalize to 0-1 range
    
    def _calculate_graph_relevance_for_semantic(self, result: Dict[str, Any]) -> float:
        """Calculate graph relevance for semantic search results."""
        # Simple implementation - could be enhanced with actual graph queries
        category = result.get("category", "")
        if category in ["technical", "bug", "configuration"]:
            return 0.8
        elif category in ["billing", "account"]:
            return 0.6
        else:
            return 0.4
    
    def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
        """Calculate recency score based on result timestamp."""
        # Simple implementation - could use actual timestamps
        metadata = result.get("metadata", {})
        resolution_time = metadata.get("avg_resolution_time", 24)
        
        # Prefer solutions with faster resolution times
        if resolution_time <= 4:
            return 1.0
        elif resolution_time <= 24:
            return 0.8
        elif resolution_time <= 72:
            return 0.6
        else:
            return 0.4
    
    def get_search_capabilities(self) -> Dict[str, Any]:
        """Get available search capabilities."""
        return {
            "semantic_search": self.semantic_available,
            "graph_search": self.graph_available,
            "hybrid_search": self.semantic_available and self.graph_available,
            "components": {
                "rag_pipeline": self.rag_pipeline is not None,
                "graph_manager": self.graph_manager is not None
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "semantic_stats": {},
            "graph_stats": {},
            "hybrid_enabled": self.semantic_available and self.graph_available
        }
        
        # Get RAG pipeline stats
        if self.rag_pipeline:
            try:
                stats["semantic_stats"] = {
                    "vector_store_size": len(self.rag_pipeline.vector_store.docstore._dict) if hasattr(self.rag_pipeline, 'vector_store') else 0,
                    "embedding_model": getattr(self.rag_pipeline.embedding_manager, 'model_name', 'unknown')
                }
            except Exception as e:
                logger.warning(f"Failed to get semantic stats: {e}")
        
        # Get graph stats
        if self.graph_manager:
            try:
                stats["graph_stats"] = self.graph_manager.get_graph_stats()
            except Exception as e:
                logger.warning(f"Failed to get graph stats: {e}")
        
        return stats
    
    def close(self):
        """Close connections and cleanup resources."""
        if self.graph_manager:
            self.graph_manager.close()
        
        logger.info("Hybrid RAG Pipeline closed")


class GraphRAGConfig:
    """Configuration for Graph-RAG system."""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        enable_semantic: bool = True,
        enable_graph: bool = True,
        default_top_k: int = 10,
        relevance_weights: Optional[RelevanceWeights] = None
    ):
        """Initialize Graph-RAG configuration."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.enable_semantic = enable_semantic
        self.enable_graph = enable_graph
        self.default_top_k = default_top_k
        self.relevance_weights = relevance_weights or RelevanceWeights()
    
    @classmethod
    def from_env(cls) -> "GraphRAGConfig":
        """Create configuration from environment variables."""
        import os
        
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            enable_semantic=os.getenv("ENABLE_SEMANTIC", "true").lower() == "true",
            enable_graph=os.getenv("ENABLE_GRAPH", "true").lower() == "true",
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "10"))
        )