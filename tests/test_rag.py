# tests/test_rag.py

"""
Unit tests for the RAG (Retrieval-Augmented Generation) pipeline.
Tests embedding generation, vector search, hybrid retrieval, and API integration.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock imports for testing without dependencies
class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        # Return mock embeddings with consistent shape
        if isinstance(texts, str):
            texts = [texts]
        return np.random.random((len(texts), 384)).astype(np.float32)

class MockFAISS:
    class IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
            self.ntotal = 0
            self._vectors = []
        
        def add(self, vectors):
            self._vectors.extend(vectors.tolist())
            self.ntotal = len(self._vectors)
        
        def search(self, query, k):
            # Return mock search results
            similarities = np.random.random(min(k, self.ntotal))
            indices = np.arange(min(k, self.ntotal))
            return similarities.reshape(1, -1), indices.reshape(1, -1)
        
        def train(self, vectors):
            pass
    
    @staticmethod
    def normalize_L2(vectors):
        # Mock normalization
        pass
    
    @staticmethod
    def write_index(index, path):
        # Mock save operation
        pass
    
    @staticmethod
    def read_index(path):
        # Mock load operation
        mock_index = MockFAISS.IndexFlatIP(384)
        mock_index.ntotal = 10
        return mock_index

class MockTfidfVectorizer:
    def __init__(self, **kwargs):
        pass
    
    def fit_transform(self, documents):
        # Return mock TF-IDF matrix
        return np.random.random((len(documents), 100))
    
    def transform(self, documents):
        # Return mock query vector
        return np.random.random((1, 100))

class MockCosineSimilarity:
    @staticmethod
    def cosine_similarity(X, Y):
        # Return mock similarities
        return np.random.random((X.shape[0], Y.shape[0]))

# Patch imports before importing our modules
with patch.dict('sys.modules', {
    'sentence_transformers': Mock(SentenceTransformer=MockSentenceTransformer),
    'faiss': MockFAISS,
    'sklearn.feature_extraction.text': Mock(TfidfVectorizer=MockTfidfVectorizer),
    'sklearn.metrics.pairwise': MockCosineSimilarity
}):
    from src.retrieval.embedding_manager import EmbeddingManager, generate_mock_resolutions
    from src.retrieval.vector_store import VectorStore, HybridRetriever
    from src.retrieval.rag_pipeline import RAGPipeline


class TestEmbeddingManager:
    """Test the EmbeddingManager class."""
    
    def test_embedding_manager_initialization(self):
        """Test EmbeddingManager initialization."""
        manager = EmbeddingManager()
        assert manager.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert manager.model is None
        assert not manager.is_loaded
    
    def test_load_model(self):
        """Test model loading."""
        manager = EmbeddingManager()
        manager.load_model()
        
        assert manager.is_loaded
        assert manager.model is not None
        assert manager.embedding_dim == 384
    
    def test_encode_texts(self):
        """Test text encoding."""
        manager = EmbeddingManager()
        manager.load_model()
        
        texts = ["Hello world", "Test message"]
        embeddings = manager.encode_texts(texts)
        
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32
    
    def test_encode_single_text(self):
        """Test single text encoding."""
        manager = EmbeddingManager()
        manager.load_model()
        
        text = "Single test message"
        embedding = manager.encode_single_text(text)
        
        assert embedding.shape == (1, 384)
        assert embedding.dtype == np.float32
    
    def test_prepare_ticket_text(self):
        """Test ticket text preparation."""
        manager = EmbeddingManager()
        
        ticket_data = {
            "subject": "Login issue",
            "description": "Cannot log in",
            "error_logs": "Timeout error",
            "product": "web_app",
            "category": "authentication"
        }
        
        text = manager.prepare_ticket_text(ticket_data)
        
        assert "Subject: Login issue" in text
        assert "Description: Cannot log in" in text
        assert "Error: Timeout error" in text
        assert "Product: web_app" in text
        assert "Category: authentication" in text
    
    def test_prepare_resolution_text(self):
        """Test resolution text preparation."""
        manager = EmbeddingManager()
        
        resolution_data = {
            "resolution": "Reset password",
            "resolution_steps": "Step 1: Go to login page",
            "category": "authentication",
            "product": "web_app"
        }
        
        text = manager.prepare_resolution_text(resolution_data)
        
        assert "Resolution: Reset password" in text
        assert "Steps: Step 1: Go to login page" in text
        assert "Category: authentication" in text
        assert "Product: web_app" in text
    
    def test_generate_mock_resolutions(self):
        """Test mock resolution generation."""
        resolutions = generate_mock_resolutions(10)
        
        assert len(resolutions) == 10
        
        for resolution in resolutions:
            assert "resolution_id" in resolution
            assert "category" in resolution
            assert "resolution" in resolution
            assert "success_rate" in resolution
            assert 0.7 <= resolution["success_rate"] <= 0.98


class TestVectorStore:
    """Test the VectorStore class."""
    
    def test_vector_store_initialization(self):
        """Test VectorStore initialization."""
        store = VectorStore(embedding_dim=384)
        
        assert store.embedding_dim == 384
        assert store.index_type == "flat"
        assert not store.is_built
        assert store.metadata == []
    
    def test_build_index(self):
        """Test index building."""
        store = VectorStore(embedding_dim=384)
        
        # Create mock embeddings and metadata
        embeddings = np.random.random((5, 384)).astype(np.float32)
        metadata = [{"id": i, "text": f"text {i}"} for i in range(5)]
        
        store.build_index(embeddings, metadata)
        
        assert store.is_built
        assert len(store.metadata) == 5
        assert store.index.ntotal == 5
    
    def test_add_vectors(self):
        """Test adding vectors to existing index."""
        store = VectorStore(embedding_dim=384)
        
        # Build initial index
        embeddings1 = np.random.random((3, 384)).astype(np.float32)
        metadata1 = [{"id": i, "text": f"text {i}"} for i in range(3)]
        store.build_index(embeddings1, metadata1)
        
        # Add more vectors
        embeddings2 = np.random.random((2, 384)).astype(np.float32)
        metadata2 = [{"id": i+3, "text": f"text {i+3}"} for i in range(2)]
        store.add_vectors(embeddings2, metadata2)
        
        assert len(store.metadata) == 5
        assert store.index.ntotal == 5
    
    def test_search(self):
        """Test vector search."""
        store = VectorStore(embedding_dim=384)
        
        # Build index
        embeddings = np.random.random((10, 384)).astype(np.float32)
        metadata = [{"id": i, "text": f"text {i}"} for i in range(10)]
        store.build_index(embeddings, metadata)
        
        # Search
        query = np.random.random((1, 384)).astype(np.float32)
        scores, results = store.search(query, k=5)
        
        assert len(scores) == 5
        assert len(results) == 5
        assert all(isinstance(score, float) for score in scores)
    
    def test_get_index_stats(self):
        """Test index statistics."""
        store = VectorStore(embedding_dim=384)
        
        # Test unbuilt index
        stats = store.get_index_stats()
        assert not stats["is_built"]
        assert stats["num_vectors"] == 0
        
        # Build index and test again
        embeddings = np.random.random((5, 384)).astype(np.float32)
        metadata = [{"id": i} for i in range(5)]
        store.build_index(embeddings, metadata)
        
        stats = store.get_index_stats()
        assert stats["is_built"]
        assert stats["num_vectors"] == 5
        assert stats["metadata_count"] == 5


class TestHybridRetriever:
    """Test the HybridRetriever class."""
    
    def test_hybrid_retriever_initialization(self):
        """Test HybridRetriever initialization."""
        # Create vector store
        store = VectorStore(embedding_dim=384)
        embeddings = np.random.random((5, 384)).astype(np.float32)
        metadata = [{"text": f"test text {i}", "resolution": f"solution {i}"} for i in range(5)]
        store.build_index(embeddings, metadata)
        
        # Create embedding manager
        manager = EmbeddingManager()
        manager.load_model()
        
        # Create hybrid retriever
        retriever = HybridRetriever(store, manager)
        
        assert retriever.vector_store == store
        assert retriever.embedding_manager == manager
    
    def test_semantic_search(self):
        """Test semantic search."""
        # Setup components
        store = VectorStore(embedding_dim=384)
        embeddings = np.random.random((5, 384)).astype(np.float32)
        metadata = [{"text": f"test text {i}"} for i in range(5)]
        store.build_index(embeddings, metadata)
        
        manager = EmbeddingManager()
        manager.load_model()
        
        retriever = HybridRetriever(store, manager)
        
        # Test search
        scores, results = retriever.semantic_search("test query", k=3)
        
        assert len(scores) <= 3
        assert len(results) <= 3
    
    def test_keyword_search(self):
        """Test keyword search."""
        # Setup components
        store = VectorStore(embedding_dim=384)
        embeddings = np.random.random((5, 384)).astype(np.float32)
        metadata = [{"text": f"test text {i}", "resolution": f"solution {i}"} for i in range(5)]
        store.build_index(embeddings, metadata)
        
        manager = EmbeddingManager()
        manager.load_model()
        
        retriever = HybridRetriever(store, manager)
        
        # Test search
        scores, results = retriever.keyword_search("test query", k=3)
        
        # Note: Might return empty if TF-IDF is not properly mocked
        assert isinstance(scores, list)
        assert isinstance(results, list)
    
    def test_hybrid_search(self):
        """Test hybrid search combining semantic and keyword."""
        # Setup components
        store = VectorStore(embedding_dim=384)
        embeddings = np.random.random((5, 384)).astype(np.float32)
        metadata = [
            {"text": f"test text {i}", "resolution": f"solution {i}", "success_rate": 0.8}
            for i in range(5)
        ]
        store.build_index(embeddings, metadata)
        
        manager = EmbeddingManager()
        manager.load_model()
        
        retriever = HybridRetriever(store, manager)
        
        # Test search
        results = retriever.hybrid_search("test query", k=3)
        
        assert isinstance(results, list)
        for result in results:
            assert "combined_score" in result
            assert "semantic_score" in result
            assert "keyword_score" in result


class TestRAGPipeline:
    """Test the RAGPipeline class."""
    
    def test_rag_pipeline_initialization(self):
        """Test RAGPipeline initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            
            assert pipeline.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert pipeline.vector_store_dir == Path(temp_dir)
            assert not pipeline.is_initialized
    
    def test_initialize_pipeline(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            pipeline.initialize()
            
            assert pipeline.is_initialized
            assert pipeline.embedding_manager.is_loaded
            assert pipeline.vector_store.is_built
            assert pipeline.hybrid_retriever is not None
    
    def test_query_solutions(self):
        """Test solution querying."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            pipeline.initialize()
            
            ticket_data = {
                "subject": "Login issue",
                "description": "Cannot authenticate",
                "error_logs": "Timeout error",
                "product": "web_app"
            }
            
            results = pipeline.query_solutions(ticket_data, k=5)
            
            assert isinstance(results, list)
            assert len(results) <= 5
            
            for result in results:
                assert "similarity_score" in result
                assert "search_type" in result
    
    def test_add_ticket_resolutions(self):
        """Test adding ticket resolutions to knowledge base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            pipeline.initialize()
            
            initial_size = pipeline.knowledge_base_size
            
            tickets_data = [
                {
                    "ticket_id": "TK-001",
                    "subject": "Login issue",
                    "description": "Cannot authenticate",
                    "resolution": "Reset password",
                    "status": "resolved"
                },
                {
                    "ticket_id": "TK-002",
                    "subject": "Database error",
                    "description": "Connection timeout",
                    "resolution": "Restart database service",
                    "status": "resolved"
                }
            ]
            
            pipeline.add_ticket_resolutions(tickets_data)
            
            assert pipeline.knowledge_base_size > initial_size
    
    def test_add_kb_articles(self):
        """Test adding knowledge base articles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            pipeline.initialize()
            
            initial_size = pipeline.knowledge_base_size
            
            kb_articles = [
                {
                    "article_id": "KB-001",
                    "title": "How to reset password",
                    "content": "Steps to reset your password",
                    "category": "authentication",
                    "helpful_score": 0.9
                },
                {
                    "article_id": "KB-002",
                    "title": "Database troubleshooting",
                    "content": "Common database issues and solutions",
                    "category": "database",
                    "helpful_score": 0.85
                }
            ]
            
            pipeline.add_kb_articles(kb_articles)
            
            assert pipeline.knowledge_base_size > initial_size
    
    def test_get_stats(self):
        """Test pipeline statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            
            # Test uninitialized stats
            stats = pipeline.get_stats()
            assert stats["status"] == "not_initialized"
            
            # Initialize and test again
            pipeline.initialize()
            stats = pipeline.get_stats()
            
            assert stats["status"] == "initialized"
            assert "knowledge_base_size" in stats
            assert "vector_store_stats" in stats
            assert "content_types" in stats
            assert "embedding_dimension" in stats
    
    def test_update_success_rates(self):
        """Test updating success rates based on feedback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGPipeline(vector_store_dir=temp_dir)
            pipeline.initialize()
            
            # Add some resolutions first
            tickets_data = [
                {
                    "ticket_id": "TK-001",
                    "subject": "Test issue",
                    "resolution": "Test solution",
                    "status": "resolved"
                }
            ]
            pipeline.add_ticket_resolutions(tickets_data)
            
            # Update success rates
            feedback_data = [
                {
                    "resolution_id": "RES-0001",
                    "success": True
                }
            ]
            
            # Should not raise an error
            pipeline.update_success_rates(feedback_data)


class TestRAGAPIIntegration:
    """Test RAG integration with FastAPI."""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create a mock RAG pipeline for testing."""
        mock_pipeline = Mock()
        mock_pipeline.is_initialized = True
        mock_pipeline.query_solutions.return_value = [
            {
                "resolution_id": "RES-001",
                "category": "authentication",
                "resolution": "Reset password to resolve login issues",
                "similarity_score": 0.95,
                "semantic_score": 0.9,
                "keyword_score": 0.1,
                "success_rate": 0.85,
                "type": "resolution",
                "search_type": "hybrid"
            }
        ]
        return mock_pipeline
    
    def test_retrieval_request_validation(self):
        """Test RetrievalRequest validation."""
        from src.api.main import RetrievalRequest
        
        # Valid request
        valid_request = RetrievalRequest(
            subject="Login issue",
            description="Cannot authenticate",
            k=5,
            search_type="hybrid"
        )
        
        assert valid_request.subject == "Login issue"
        assert valid_request.k == 5
        assert valid_request.search_type == "hybrid"
    
    def test_solution_result_model(self):
        """Test SolutionResult model."""
        from src.api.main import SolutionResult
        
        result = SolutionResult(
            resolution="Test resolution",
            similarity_score=0.95,
            search_type="hybrid"
        )
        
        assert result.resolution == "Test resolution"
        assert result.similarity_score == 0.95
        assert result.search_type == "hybrid"
    
    def test_retrieval_response_model(self):
        """Test RetrievalResponse model."""
        from src.api.main import RetrievalResponse, SolutionResult
        
        solutions = [
            SolutionResult(
                resolution="Test solution",
                similarity_score=0.9,
                search_type="hybrid"
            )
        ]
        
        response = RetrievalResponse(
            query_summary="Test query",
            solutions=solutions,
            total_found=1,
            search_type="hybrid",
            processing_time=0.5
        )
        
        assert response.query_summary == "Test query"
        assert len(response.solutions) == 1
        assert response.total_found == 1


class TestRAGUtilities:
    """Test RAG utility functions."""
    
    def test_build_index_from_tickets(self):
        """Test building index from tickets file."""
        from src.retrieval.rag_pipeline import build_index_from_tickets
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock tickets file
            tickets_data = [
                {
                    "ticket_id": "TK-001",
                    "subject": "Login issue",
                    "description": "Cannot authenticate",
                    "resolution": "Reset password",
                    "status": "resolved"
                }
            ]
            
            tickets_file = Path(temp_dir) / "tickets.json"
            with open(tickets_file, 'w') as f:
                json.dump(tickets_data, f)
            
            # Build index
            pipeline = build_index_from_tickets(
                str(tickets_file),
                output_dir=temp_dir
            )
            
            assert pipeline.is_initialized
            assert pipeline.knowledge_base_size > 0


# Integration test fixtures
@pytest.fixture
def sample_ticket_data():
    """Sample ticket data for testing."""
    return {
        "subject": "Cannot login to application",
        "description": "User is unable to authenticate with correct credentials",
        "error_logs": "Authentication timeout after 30 seconds",
        "product": "web_application",
        "category": "authentication"
    }


@pytest.fixture
def sample_resolutions():
    """Sample resolution data for testing."""
    return [
        {
            "resolution_id": "RES-001",
            "category": "authentication",
            "resolution": "Reset user password and clear cache",
            "success_rate": 0.9,
            "usage_count": 15
        },
        {
            "resolution_id": "RES-002",
            "category": "database",
            "resolution": "Restart database service",
            "success_rate": 0.85,
            "usage_count": 8
        }
    ]


def test_end_to_end_rag_workflow(sample_ticket_data):
    """Test complete RAG workflow end-to-end."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize pipeline
        pipeline = RAGPipeline(vector_store_dir=temp_dir)
        pipeline.initialize()
        
        # Query solutions
        results = pipeline.query_solutions(sample_ticket_data, k=3)
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # Check result structure
        for result in results:
            assert "similarity_score" in result
            assert "search_type" in result
            assert "resolution" in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])