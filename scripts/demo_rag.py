# scripts/demo_rag.py

"""
Comprehensive demonstration of the RAG (Retrieval-Augmented Generation) pipeline.
Shows embedding generation, vector search, hybrid retrieval, and API integration.
"""

import sys
import json
import time
import requests
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging

# Setup logging
logger = setup_logging()


def test_rag_pipeline():
    """Test the RAG pipeline components directly."""
    print("🧪 Testing RAG Pipeline Components")
    print("=" * 50)
    
    try:
        from src.retrieval.rag_pipeline import RAGPipeline
        from src.retrieval.embedding_manager import EmbeddingManager
        
        print("1️⃣ Initializing RAG Pipeline...")
        
        # Initialize pipeline
        pipeline = RAGPipeline(vector_store_dir="vector_store")
        pipeline.initialize()
        
        # Get stats
        stats = pipeline.get_stats()
        print(f"   ✅ Pipeline Status: {stats['status']}")
        print(f"   📊 Knowledge Base Size: {stats['knowledge_base_size']}")
        print(f"   🧠 Embedding Dimension: {stats['embedding_dimension']}")
        print(f"   📁 Content Types: {stats['content_types']}")
        
        print("\n2️⃣ Testing Embedding Generation...")
        
        # Test embedding generation
        test_text = "User cannot login due to authentication timeout error"
        start_time = time.time()
        embedding = pipeline.embedding_manager.encode_single_text(test_text)
        embed_time = time.time() - start_time
        
        print(f"   ✅ Generated embedding shape: {embedding.shape}")
        print(f"   ⏱️ Embedding time: {embed_time:.3f}s")
        
        print("\n3️⃣ Testing Solution Retrieval...")
        
        # Test queries with different scenarios
        test_scenarios = [
            {
                "name": "Authentication Issue",
                "ticket": {
                    "subject": "Cannot login to application",
                    "description": "User is unable to authenticate with correct credentials. Getting timeout error after 30 seconds.",
                    "error_logs": "Authentication service timeout: connection refused",
                    "product": "web_application",
                    "category": "authentication"
                }
            },
            {
                "name": "Database Connection Problem",
                "ticket": {
                    "subject": "Database connection failed",
                    "description": "Application cannot connect to database server. Users seeing 500 error.",
                    "error_logs": "MySQL connection timeout after 30 seconds",
                    "product": "api_server",
                    "category": "database"
                }
            },
            {
                "name": "Payment Processing Issue",
                "ticket": {
                    "subject": "Credit card payment not working",
                    "description": "Payment fails during checkout process. Customer gets error message.",
                    "error_logs": "Payment gateway error: invalid card details",
                    "product": "payment_gateway",
                    "category": "payment"
                }
            },
            {
                "name": "API Integration Error",
                "ticket": {
                    "subject": "Third-party API integration failing",
                    "description": "External API calls returning 401 unauthorized errors",
                    "error_logs": "API authentication failed: invalid token",
                    "product": "api_server",
                    "category": "api"
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n   🎯 Scenario: {scenario['name']}")
            
            # Test different search types
            search_types = ["semantic", "keyword", "hybrid"]
            
            for search_type in search_types:
                try:
                    start_time = time.time()
                    results = pipeline.query_solutions(
                        scenario['ticket'],
                        k=3,
                        search_type=search_type
                    )
                    query_time = time.time() - start_time
                    
                    print(f"      🔍 {search_type.capitalize()} Search:")
                    print(f"         Results: {len(results)} | Time: {query_time:.3f}s")
                    
                    if results:
                        best_result = results[0]
                        score = best_result.get("similarity_score", 0)
                        category = best_result.get("category", "N/A")
                        resolution = best_result.get("resolution", "")[:60] + "..."
                        
                        print(f"         Best: {score:.3f} | {category} | {resolution}")
                        
                        # Show detailed scores for hybrid search
                        if search_type == "hybrid" and len(results) > 0:
                            sem_score = best_result.get("semantic_score", 0)
                            kw_score = best_result.get("keyword_score", 0)
                            success_rate = best_result.get("success_rate", 0)
                            print(f"         Details: Semantic={sem_score:.3f}, Keyword={kw_score:.3f}, Success={success_rate:.2f}")
                
                except Exception as e:
                    print(f"         ❌ Error: {str(e)[:50]}...")
        
        print("\n4️⃣ Testing Knowledge Base Updates...")
        
        # Test adding new knowledge
        new_kb_articles = [
            {
                "article_id": "KB-DEMO-001",
                "title": "Troubleshooting Login Issues",
                "content": "When users experience login problems, first check if the authentication service is running. Common causes include expired passwords, locked accounts, or server timeouts.",
                "category": "authentication",
                "tags": ["login", "troubleshooting", "authentication"],
                "helpful_score": 0.95
            }
        ]
        
        initial_size = pipeline.knowledge_base_size
        pipeline.add_kb_articles(new_kb_articles)
        pipeline.save_index()
        
        print(f"   ✅ Added {len(new_kb_articles)} KB articles")
        print(f"   📈 Knowledge base grew from {initial_size} to {pipeline.knowledge_base_size}")
        
        print("\n✅ RAG Pipeline Testing Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG Pipeline Test Failed: {e}")
        logger.error(f"RAG pipeline test error: {e}")
        return False


def test_api_integration():
    """Test RAG integration with FastAPI."""
    print("\n🌐 Testing API Integration")
    print("=" * 30)
    
    # API base URL
    base_url = "http://localhost:8000"
    
    try:
        # Test health check
        print("1️⃣ Testing Health Check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API Status: {health_data.get('status', 'unknown')}")
            print(f"   🤖 Models Available: {health_data.get('models', {})}")
            print(f"   🔍 RAG Available: {health_data.get('rag_available', False)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test RAG endpoint
        print("\n2️⃣ Testing RAG Retrieval Endpoint...")
        
        test_requests = [
            {
                "name": "Authentication Problem",
                "payload": {
                    "subject": "User cannot login to the system",
                    "description": "Getting authentication timeout errors when trying to log in",
                    "error_logs": "Auth service timeout after 30 seconds",
                    "product": "web_application",
                    "category": "authentication",
                    "k": 5,
                    "search_type": "hybrid"
                }
            },
            {
                "name": "Database Issue",
                "payload": {
                    "subject": "Database connection errors",
                    "description": "Application cannot connect to the database",
                    "error_logs": "Connection pool exhausted",
                    "product": "api_server",
                    "k": 3,
                    "search_type": "semantic"
                }
            }
        ]
        
        for test_request in test_requests:
            print(f"\n   🧪 Testing: {test_request['name']}")
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/retrieve/solutions",
                    json=test_request['payload'],
                    timeout=30
                )
                api_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    print(f"      ✅ Request successful: {api_time:.3f}s")
                    print(f"      📊 Solutions found: {data.get('total_found', 0)}")
                    print(f"      🔍 Search type: {data.get('search_type', 'unknown')}")
                    print(f"      ⏱️ Processing time: {data.get('processing_time', 0):.3f}s")
                    
                    solutions = data.get('solutions', [])
                    if solutions:
                        best_solution = solutions[0]
                        print(f"      🎯 Best solution score: {best_solution.get('similarity_score', 0):.3f}")
                        print(f"      📝 Resolution: {best_solution.get('resolution', '')[:80]}...")
                
                else:
                    print(f"      ❌ Request failed: {response.status_code}")
                    print(f"         Error: {response.text}")
            
            except requests.exceptions.RequestException as e:
                print(f"      ❌ Request error: {e}")
        
        print("\n3️⃣ Testing Category Prediction Integration...")
        
        # Test prediction + RAG workflow
        prediction_payload = {
            "ticket_id": "DEMO-001",
            "subject": "Cannot login to application",
            "description": "User authentication failing with timeout errors",
            "error_logs": "Authentication service timeout",
            "product": "web_application"
        }
        
        try:
            # Get category prediction
            pred_response = requests.post(
                f"{base_url}/predict/category",
                json=prediction_payload,
                timeout=30
            )
            
            if pred_response.status_code == 200:
                pred_data = pred_response.json()
                print(f"   ✅ Category prediction successful")
                
                # Extract predicted category for RAG query
                predictions = pred_data.get('predictions', {})
                predicted_category = None
                
                if 'xgboost' in predictions:
                    predicted_category = predictions['xgboost'].get('predicted_category')
                elif 'tensorflow' in predictions:
                    predicted_category = predictions['tensorflow'].get('predicted_category')
                
                if predicted_category:
                    print(f"   🎯 Predicted category: {predicted_category}")
                    
                    # Use prediction for enhanced RAG query
                    rag_payload = {
                        "subject": prediction_payload["subject"],
                        "description": prediction_payload["description"],
                        "error_logs": prediction_payload["error_logs"],
                        "product": prediction_payload["product"],
                        "category": predicted_category,  # Use predicted category
                        "k": 5,
                        "search_type": "hybrid"
                    }
                    
                    rag_response = requests.post(
                        f"{base_url}/retrieve/solutions",
                        json=rag_payload,
                        timeout=30
                    )
                    
                    if rag_response.status_code == 200:
                        rag_data = rag_response.json()
                        print(f"   🔍 RAG retrieval with predicted category successful")
                        print(f"   📊 Enhanced solutions found: {rag_data.get('total_found', 0)}")
                    else:
                        print(f"   ⚠️ RAG with prediction failed: {rag_response.status_code}")
            
            else:
                print(f"   ⚠️ Category prediction failed: {pred_response.status_code}")
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Prediction+RAG test error: {e}")
        
        print("\n✅ API Integration Testing Completed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running:")
        print("   python -m uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ API Integration Test Failed: {e}")
        return False


def demonstrate_search_types():
    """Demonstrate different search types and their characteristics."""
    print("\n🔍 Demonstrating Search Types")
    print("=" * 35)
    
    try:
        from src.retrieval.rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline(vector_store_dir="vector_store")
        pipeline.initialize()
        
        # Test query that shows clear differences between search types
        test_ticket = {
            "subject": "Login authentication timeout",
            "description": "Users experiencing slow authentication process with timeout errors",
            "error_logs": "Authentication service response time: 35 seconds",
            "product": "web_application",
            "category": "authentication"
        }
        
        print(f"🎯 Test Query: {test_ticket['subject']}")
        print(f"📝 Description: {test_ticket['description']}")
        
        search_types = ["semantic", "keyword", "hybrid"]
        
        for search_type in search_types:
            print(f"\n🔍 {search_type.upper()} SEARCH RESULTS:")
            print("-" * 40)
            
            try:
                start_time = time.time()
                results = pipeline.query_solutions(
                    test_ticket,
                    k=5,
                    search_type=search_type
                )
                query_time = time.time() - start_time
                
                print(f"⏱️ Query Time: {query_time:.3f}s | Results: {len(results)}")
                
                for i, result in enumerate(results[:3], 1):
                    score = result.get("similarity_score", 0)
                    category = result.get("category", "N/A")
                    resolution = result.get("resolution", "No resolution")
                    content_type = result.get("type", "unknown")
                    
                    print(f"\n{i}. Score: {score:.3f} | Category: {category} | Type: {content_type}")
                    print(f"   Resolution: {resolution[:100]}...")
                    
                    # Show detailed scores for hybrid
                    if search_type == "hybrid":
                        sem_score = result.get("semantic_score", 0)
                        kw_score = result.get("keyword_score", 0)
                        success_rate = result.get("success_rate", 0)
                        print(f"   📊 Semantic: {sem_score:.3f} | Keyword: {kw_score:.3f} | Success: {success_rate:.2f}")
            
            except Exception as e:
                print(f"❌ Error with {search_type} search: {e}")
        
        # Show search type recommendations
        print(f"\n💡 SEARCH TYPE RECOMMENDATIONS:")
        print("📈 Semantic Search: Best for finding conceptually similar solutions")
        print("🔤 Keyword Search: Best for exact term matches and technical terms") 
        print("🎯 Hybrid Search: Best overall performance combining both approaches")
        
        return True
        
    except Exception as e:
        print(f"❌ Search type demonstration failed: {e}")
        return False


def performance_benchmark():
    """Benchmark RAG pipeline performance."""
    print("\n⚡ Performance Benchmark")
    print("=" * 25)
    
    try:
        from src.retrieval.rag_pipeline import RAGPipeline
        import statistics
        
        pipeline = RAGPipeline(vector_store_dir="vector_store")
        pipeline.initialize()
        
        # Test queries
        test_queries = [
            "Authentication timeout error",
            "Database connection failed", 
            "Payment processing issue",
            "API integration problem",
            "User interface bug"
        ]
        
        print(f"🧪 Testing {len(test_queries)} queries with different k values...")
        
        k_values = [1, 5, 10, 20]
        
        for k in k_values:
            print(f"\n📊 Testing with k={k}:")
            
            times = []
            
            for query in test_queries:
                test_ticket = {
                    "subject": query,
                    "description": f"Support ticket for {query}",
                    "error_logs": f"Error related to {query}"
                }
                
                start_time = time.time()
                results = pipeline.query_solutions(test_ticket, k=k, search_type="hybrid")
                query_time = time.time() - start_time
                
                times.append(query_time)
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"   ⏱️ Average: {avg_time:.3f}s | Min: {min_time:.3f}s | Max: {max_time:.3f}s")
        
        # Test batch performance
        print(f"\n🚀 Batch Processing Test:")
        
        batch_queries = [
            {"subject": f"Test query {i}", "description": f"Description {i}"}
            for i in range(10)
        ]
        
        start_time = time.time()
        for query in batch_queries:
            pipeline.query_solutions(query, k=5, search_type="hybrid")
        batch_time = time.time() - start_time
        
        print(f"   📦 Processed {len(batch_queries)} queries in {batch_time:.3f}s")
        print(f"   ⚡ Average per query: {batch_time/len(batch_queries):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Run comprehensive RAG demonstration."""
    print("🚀 InsightDesk AI - RAG Pipeline Demonstration")
    print("=" * 55)
    
    # Check if vector store exists
    vector_store_path = Path("vector_store")
    if not vector_store_path.exists():
        print("⚠️ Vector store not found. Building RAG index first...")
        
        try:
            from scripts.build_rag_index import main as build_index
            success = build_index()
            if not success:
                print("❌ Failed to build RAG index. Exiting.")
                return False
        except Exception as e:
            print(f"❌ Error building index: {e}")
            return False
    
    # Run tests
    tests = [
        ("RAG Pipeline Components", test_rag_pipeline),
        ("Search Type Demonstration", demonstrate_search_types),
        ("Performance Benchmark", performance_benchmark),
        ("API Integration", test_api_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 DEMONSTRATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All RAG pipeline demonstrations completed successfully!")
        print("\n🚀 Next Steps:")
        print("   • Start the API server: python -m uvicorn src.api.main:app --reload")
        print("   • Visit API docs: http://localhost:8000/docs")
        print("   • Test RAG endpoint: POST /retrieve/solutions")
    else:
        print("⚠️ Some demonstrations failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)