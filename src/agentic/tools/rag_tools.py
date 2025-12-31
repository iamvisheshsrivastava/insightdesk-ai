# src/agentic/tools/rag_tools.py
from typing import Any, Dict, List
import logging
from src.retrieval.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class RAGSearchTool:
    name = "solution_retriever"
    description = "Searches the knowledge base for existing solutions to similar tickets."

    def __init__(self, pipeline: RAGPipeline = None):
        if pipeline:
            self.pipeline = pipeline
        else:
            # Initialize with default settings if not provided
            # Note: This might be slow if initialized per request, better to pass singleton
            self.pipeline = RAGPipeline()
            # We don't call initialize() here to avoid overhead during import/init
            # It should be initialized by the orchestrator or dependency injector

    def run(self, query_data: Dict[str, Any], k: int = 3) -> Dict[str, Any]:
        """
        Search for solutions.
        
        Args:
            query_data: Dictionary with query info (subject, description) or just text
            k: Number of results
        """
        logger.info(f"Running RAGSearchTool for query: {query_data}")
        
        if not self.pipeline.is_initialized:
             logger.info("Initializing RAG pipeline within tool...")
             self.pipeline.initialize(force_rebuild=False)

        results = self.pipeline.query_solutions(query_data, k=k, search_type="hybrid")
        
        return {
            "found_solutions": results,
            "count": len(results)
        }
