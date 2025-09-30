# scripts/build_rag_index.py

"""
Script to build RAG (Retrieval-Augmented Generation) index from support tickets data.
This script initializes the RAG pipeline and builds the vector store index.
"""

import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline, build_index_from_tickets
from src.retrieval.embedding_manager import generate_mock_resolutions
from src.utils import setup_logging

# Setup logging
logger = setup_logging()


def main():
    """Build RAG index from tickets data."""
    print("🔍 Building RAG Index for InsightDesk AI")
    print("=" * 50)
    
    # Configuration
    tickets_file = "data/support_tickets.json"
    vector_store_dir = "vector_store"
    force_rebuild = False
    
    try:
        # Check if tickets file exists
        tickets_path = Path(tickets_file)
        
        if tickets_path.exists():
            print(f"📁 Found tickets file: {tickets_file}")
            
            # Load tickets data to check size
            with open(tickets_path, 'r') as f:
                tickets_data = json.load(f)
            
            print(f"📊 Found {len(tickets_data)} tickets in data file")
            
            # Build index from tickets
            print("🏗️ Building RAG index from tickets data...")
            pipeline = build_index_from_tickets(
                tickets_file,
                output_dir=vector_store_dir,
                force_rebuild=force_rebuild
            )
            
        else:
            print(f"⚠️ Tickets file not found: {tickets_file}")
            print("🔧 Creating RAG index with mock data for demonstration...")
            
            # Initialize pipeline with mock data
            pipeline = RAGPipeline(vector_store_dir=vector_store_dir)
            pipeline.initialize(force_rebuild=force_rebuild)
        
        # Get pipeline statistics
        stats = pipeline.get_stats()
        
        print("\n📈 RAG Pipeline Statistics:")
        print(f"   • Status: {stats['status']}")
        print(f"   • Knowledge Base Size: {stats['knowledge_base_size']}")
        print(f"   • Embedding Dimension: {stats['embedding_dimension']}")
        print(f"   • Content Types: {stats['content_types']}")
        
        # Test the pipeline with a sample query
        print("\n🧪 Testing RAG pipeline with sample query...")
        
        test_ticket = {
            "subject": "Cannot login to application",
            "description": "User is unable to authenticate with correct credentials. Getting timeout error.",
            "error_logs": "Authentication timeout after 30 seconds",
            "product": "web_application",
            "category": "authentication"
        }
        
        # Test all search types
        search_types = ["semantic", "keyword", "hybrid"]
        
        for search_type in search_types:
            print(f"\n🔍 Testing {search_type} search:")
            
            try:
                results = pipeline.query_solutions(
                    test_ticket,
                    k=3,
                    search_type=search_type
                )
                
                print(f"   Found {len(results)} solutions:")
                
                for i, result in enumerate(results[:3], 1):
                    score = result.get("similarity_score", 0)
                    resolution = result.get("resolution", "No resolution")
                    content_type = result.get("type", "unknown")
                    
                    print(f"   {i}. Score: {score:.3f} | Type: {content_type}")
                    print(f"      Resolution: {resolution[:80]}...")
                    
                    # Show detailed scores for hybrid search
                    if search_type == "hybrid":
                        sem_score = result.get("semantic_score", 0)
                        kw_score = result.get("keyword_score", 0)
                        print(f"      Semantic: {sem_score:.3f} | Keyword: {kw_score:.3f}")
            
            except Exception as e:
                print(f"   ❌ Error testing {search_type} search: {e}")
        
        # Add some sample KB articles for demonstration
        print("\n📚 Adding sample knowledge base articles...")
        
        sample_kb_articles = [
            {
                "article_id": "KB-AUTH-001",
                "title": "Authentication Troubleshooting Guide",
                "content": "This guide covers common authentication issues including timeout errors, password reset procedures, and session management. Start by checking user credentials and then verify authentication service status.",
                "category": "authentication",
                "tags": ["login", "password", "timeout", "authentication"],
                "helpful_score": 0.92,
                "view_count": 156,
                "created_date": "2024-01-15",
                "last_updated": "2025-09-29"
            },
            {
                "article_id": "KB-DB-001",
                "title": "Database Connection Issues",
                "content": "Common database connectivity problems and their solutions. Includes connection timeout troubleshooting, connection pool management, and database service restart procedures.",
                "category": "database",
                "tags": ["database", "connection", "timeout", "performance"],
                "helpful_score": 0.88,
                "view_count": 203,
                "created_date": "2024-02-01",
                "last_updated": "2025-09-29"
            },
            {
                "article_id": "KB-API-001",
                "title": "API Integration Guide",
                "content": "Complete guide for API integration including authentication tokens, request/response formats, error handling, and rate limiting. Covers both REST and GraphQL endpoints.",
                "category": "api",
                "tags": ["api", "integration", "tokens", "rest", "graphql"],
                "helpful_score": 0.95,
                "view_count": 89,
                "created_date": "2024-03-10",
                "last_updated": "2025-09-29"
            }
        ]
        
        pipeline.add_kb_articles(sample_kb_articles)
        print(f"   ✅ Added {len(sample_kb_articles)} knowledge base articles")
        
        # Save the updated index
        pipeline.save_index()
        
        # Get updated statistics
        final_stats = pipeline.get_stats()
        print(f"\n📈 Updated Knowledge Base Size: {final_stats['knowledge_base_size']}")
        print(f"📈 Content Types: {final_stats['content_types']}")
        
        print("\n✅ RAG index building completed successfully!")
        print(f"💾 Index saved to: {vector_store_dir}")
        print("\n🚀 You can now start the API server to use the RAG pipeline:")
        print("   python -m uvicorn src.api.main:app --reload")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to build RAG index: {e}")
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to install required dependencies:")
        print("   pip install sentence-transformers faiss-cpu scikit-learn")
        return False


def add_sample_resolutions():
    """Add sample resolution data to the RAG pipeline."""
    print("\n🔧 Generating additional sample resolutions...")
    
    try:
        # Generate more comprehensive mock resolutions
        mock_resolutions = generate_mock_resolutions(50)
        
        # Initialize pipeline
        pipeline = RAGPipeline(vector_store_dir="vector_store")
        pipeline.initialize()
        
        # Convert resolutions to ticket format for adding
        resolution_tickets = []
        for res in mock_resolutions:
            ticket = {
                "ticket_id": f"TK-{res['resolution_id'].split('-')[1]}",
                "subject": f"Issue related to {res['category']} in {res['product']}",
                "description": f"Support ticket for {res['category']} issues",
                "resolution": res["resolution"],
                "resolution_steps": res["resolution_steps"],
                "category": res["category"],
                "product": res["product"],
                "status": "resolved",
                "created_date": res["created_date"],
                "resolved_date": res["last_updated"]
            }
            resolution_tickets.append(ticket)
        
        # Add to pipeline
        pipeline.add_ticket_resolutions(resolution_tickets)
        
        # Save updated index
        pipeline.save_index()
        
        print(f"   ✅ Added {len(resolution_tickets)} sample resolutions")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add sample resolutions: {e}")
        return False


def test_rag_performance():
    """Test RAG pipeline performance with various queries."""
    print("\n⚡ Testing RAG pipeline performance...")
    
    try:
        pipeline = RAGPipeline(vector_store_dir="vector_store")
        pipeline.initialize()
        
        # Test queries covering different categories
        test_queries = [
            {
                "name": "Authentication Issue",
                "ticket": {
                    "subject": "Login timeout error",
                    "description": "User cannot login due to timeout",
                    "error_logs": "Authentication service timeout",
                    "category": "authentication"
                }
            },
            {
                "name": "Database Problem", 
                "ticket": {
                    "subject": "Database connection failed",
                    "description": "Cannot connect to database server",
                    "error_logs": "Connection timeout after 30 seconds",
                    "category": "database"
                }
            },
            {
                "name": "API Integration",
                "ticket": {
                    "subject": "API response error",
                    "description": "Getting 500 error from API endpoint",
                    "error_logs": "Internal server error",
                    "category": "api"
                }
            },
            {
                "name": "Payment Issue",
                "ticket": {
                    "subject": "Payment processing failed", 
                    "description": "Credit card payment not working",
                    "error_logs": "Payment gateway error",
                    "category": "payment"
                }
            }
        ]
        
        import time
        
        for query_info in test_queries:
            print(f"\n🧪 Testing: {query_info['name']}")
            
            start_time = time.time()
            results = pipeline.query_solutions(
                query_info['ticket'],
                k=5,
                search_type="hybrid"
            )
            query_time = time.time() - start_time
            
            print(f"   Query Time: {query_time:.3f}s")
            print(f"   Results Found: {len(results)}")
            
            if results:
                best_result = results[0]
                print(f"   Best Match: {best_result.get('similarity_score', 0):.3f} score")
                print(f"   Category: {best_result.get('category', 'N/A')}")
        
        print("\n✅ Performance testing completed")
        return True
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build RAG index for InsightDesk AI")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild existing index")
    parser.add_argument("--add-samples", action="store_true", help="Add additional sample resolutions")
    parser.add_argument("--test-performance", action="store_true", help="Test RAG pipeline performance")
    parser.add_argument("--tickets-file", default="data/support_tickets.json", help="Path to tickets JSON file")
    parser.add_argument("--output-dir", default="vector_store", help="Output directory for vector store")
    
    args = parser.parse_args()
    
    success = main()
    
    if success and args.add_samples:
        add_sample_resolutions()
    
    if success and args.test_performance:
        test_rag_performance()
    
    if success:
        print("\n🎉 RAG index building process completed successfully!")
    else:
        print("\n💥 RAG index building failed. Check logs for details.")
        sys.exit(1)