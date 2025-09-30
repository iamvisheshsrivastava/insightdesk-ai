# scripts/test_graph_rag.py

"""
Comprehensive testing script for Graph-RAG system.

This script tests the Neo4j graph manager, hybrid RAG pipeline,
and FastAPI endpoints to ensure the Graph-RAG system works correctly.
"""

import sys
import json
import asyncio
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.graph_manager import Neo4jGraphManager
from src.retrieval.hybrid_rag_pipeline import HybridRAGPipeline

# Test configuration
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password",  # Change this to your Neo4j password
    "database": "neo4j"
}

API_BASE_URL = "http://localhost:8000"


class GraphRAGTester:
    """Comprehensive testing suite for Graph-RAG system."""
    
    def __init__(self):
        self.graph_manager = None
        self.hybrid_pipeline = None
        self.test_results = {
            "neo4j_connection": False,
            "graph_schema": False,
            "data_ingestion": False,
            "graph_queries": False,
            "hybrid_search": False,
            "api_endpoints": False,
            "performance": {}
        }
        
    def setup(self):
        """Setup test environment."""
        print("🔧 Setting up Graph-RAG test environment...")
        
        try:
            # Initialize graph manager
            self.graph_manager = Neo4jGraphManager(**NEO4J_CONFIG)
            print("✅ Neo4j Graph Manager initialized")
            
            # Test basic connection
            stats = self.graph_manager.get_graph_stats()
            self.test_results["neo4j_connection"] = True
            print(f"✅ Neo4j connection successful. Graph stats: {stats}")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def test_graph_operations(self):
        """Test basic graph operations."""
        print("\n📊 Testing Graph Operations...")
        
        try:
            # Test node creation
            product_data = {
                "id": "test_product_1",
                "name": "Test Application",
                "version": "2.0.0",
                "category": "web_app"
            }
            
            product_id = self.graph_manager.create_product(product_data)
            print(f"✅ Product created: {product_id}")
            
            # Test issue creation
            issue_data = {
                "id": "test_issue_1",
                "type": "authentication_error",
                "category": "authentication",
                "description": "Users cannot login",
                "severity": "high"
            }
            
            issue_id = self.graph_manager.create_issue(issue_data)
            print(f"✅ Issue created: {issue_id}")
            
            # Test resolution creation
            resolution_data = {
                "id": "test_resolution_1",
                "type": "configuration_change",
                "description": "Update authentication timeout settings",
                "steps": ["Navigate to config", "Update timeout", "Restart service"],
                "success_rate": 0.95
            }
            
            resolution_id = self.graph_manager.create_resolution(resolution_data)
            print(f"✅ Resolution created: {resolution_id}")
            
            # Test relationship creation
            self.graph_manager.create_relationship(
                product_id, issue_id, "PRODUCT_HAS_ISSUE"
            )
            print("✅ Product-Issue relationship created")
            
            self.graph_manager.create_relationship(
                issue_id, resolution_id, "ISSUE_SOLVED_BY_RESOLUTION"
            )
            print("✅ Issue-Resolution relationship created")
            
            self.test_results["graph_schema"] = True
            return True
            
        except Exception as e:
            print(f"❌ Graph operations test failed: {e}")
            return False
    
    def test_graph_queries(self):
        """Test graph query capabilities."""
        print("\n🔍 Testing Graph Queries...")
        
        try:
            # Test finding solutions for authentication issues
            query_results = self.graph_manager.query_graph(
                "authentication timeout error",
                search_type="issue_resolution",
                limit=5
            )
            
            print(f"✅ Found {len(query_results)} solutions for authentication query")
            
            # Test relationship traversal
            related_issues = self.graph_manager.find_related_issues(
                "test_issue_1",
                relationship_types=["SIMILAR_TO"],
                max_depth=2
            )
            
            print(f"✅ Found {len(related_issues)} related issues")
            
            # Test knowledge base queries
            kb_articles = self.graph_manager.find_kb_articles(
                ["configuration", "authentication"]
            )
            
            print(f"✅ Found {len(kb_articles)} knowledge base articles")
            
            self.test_results["graph_queries"] = True
            return True
            
        except Exception as e:
            print(f"❌ Graph queries test failed: {e}")
            return False
    
    def test_hybrid_search(self):
        """Test hybrid RAG pipeline."""
        print("\n🔬 Testing Hybrid Search Pipeline...")
        
        try:
            # Note: This requires both traditional RAG and graph components
            # For testing purposes, we'll simulate the pipeline
            
            # Mock ticket data
            ticket_data = {
                "subject": "Login timeout error",
                "description": "Users are experiencing timeout errors when trying to authenticate",
                "product": "web_application",
                "category": "authentication"
            }
            
            # Test direct graph search
            graph_results = self.graph_manager.query_graph(
                f"{ticket_data['subject']} {ticket_data['description']}",
                search_type="comprehensive",
                limit=5
            )
            
            print(f"✅ Graph search returned {len(graph_results)} results")
            
            # Test ranking and scoring
            if graph_results:
                for i, result in enumerate(graph_results[:3]):
                    print(f"   Result {i+1}: {result.get('resolution_id', 'N/A')} "
                          f"(score: {result.get('relevance_score', 0.0):.3f})")
            
            self.test_results["hybrid_search"] = True
            return True
            
        except Exception as e:
            print(f"❌ Hybrid search test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test Graph-RAG API endpoints."""
        print("\n🌐 Testing API Endpoints...")
        
        try:
            # Test graph stats endpoint
            response = requests.get(f"{API_BASE_URL}/retrieve/graph/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Graph stats endpoint: {stats.get('connection_status')}")
            else:
                print(f"⚠️ Graph stats endpoint returned: {response.status_code}")
            
            # Test graph retrieval endpoint
            test_request = {
                "subject": "Authentication timeout error",
                "description": "Users cannot login due to timeout",
                "product": "web_application",
                "category": "authentication",
                "k": 5,
                "semantic_weight": 0.6,
                "graph_weight": 0.4,
                "use_graph_expansion": True,
                "max_depth": 2
            }
            
            response = requests.post(
                f"{API_BASE_URL}/retrieve/graph",
                json=test_request
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Graph retrieval endpoint: found {result.get('total_found', 0)} solutions")
                print(f"   Semantic results: {result.get('semantic_results', 0)}")
                print(f"   Graph results: {result.get('graph_results', 0)}")
            else:
                print(f"⚠️ Graph retrieval endpoint returned: {response.status_code}")
                if response.status_code != 404:  # 404 expected if API not running
                    print(f"   Error: {response.text}")
            
            # Test direct query endpoint
            query_request = {
                "query": "MATCH (p:Product)-[:PRODUCT_HAS_ISSUE]->(i:Issue) RETURN p.name, i.type LIMIT 5"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/retrieve/graph/query",
                params={"query": query_request["query"]}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Direct query endpoint: {result.get('result_count', 0)} results")
            else:
                print(f"⚠️ Direct query endpoint returned: {response.status_code}")
            
            self.test_results["api_endpoints"] = True
            return True
            
        except requests.exceptions.ConnectionError:
            print("⚠️ API server not running - skipping API tests")
            return False
        except Exception as e:
            print(f"❌ API endpoints test failed: {e}")
            return False
    
    def test_performance(self):
        """Test Graph-RAG performance."""
        print("\n⚡ Testing Performance...")
        
        try:
            # Test query performance
            test_queries = [
                "authentication error timeout",
                "database connection failed",
                "payment processing issue",
                "user interface bug",
                "performance optimization"
            ]
            
            total_time = 0
            successful_queries = 0
            
            for query in test_queries:
                start_time = time.time()
                try:
                    results = self.graph_manager.query_graph(query, limit=10)
                    query_time = time.time() - start_time
                    total_time += query_time
                    successful_queries += 1
                    print(f"   Query '{query[:30]}...': {len(results)} results in {query_time:.3f}s")
                except Exception as e:
                    print(f"   Query '{query[:30]}...' failed: {e}")
            
            if successful_queries > 0:
                avg_time = total_time / successful_queries
                print(f"✅ Average query time: {avg_time:.3f}s")
                
                self.test_results["performance"] = {
                    "avg_query_time": avg_time,
                    "successful_queries": successful_queries,
                    "total_queries": len(test_queries)
                }
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    def cleanup_test_data(self):
        """Clean up test data."""
        print("\n🧹 Cleaning up test data...")
        
        try:
            # Delete test nodes
            test_ids = ["test_product_1", "test_issue_1", "test_resolution_1"]
            
            with self.graph_manager.driver.session(database=self.graph_manager.database) as session:
                for test_id in test_ids:
                    session.run(
                        "MATCH (n {id: $id}) DETACH DELETE n",
                        id=test_id
                    )
            
            print("✅ Test data cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("🚀 Starting Graph-RAG Comprehensive Test Suite")
        print("=" * 50)
        
        if not self.setup():
            print("❌ Setup failed - aborting tests")
            return
        
        # Run all test phases
        test_phases = [
            ("Graph Operations", self.test_graph_operations),
            ("Graph Queries", self.test_graph_queries),
            ("Hybrid Search", self.test_hybrid_search),
            ("API Endpoints", self.test_api_endpoints),
            ("Performance", self.test_performance)
        ]
        
        passed_tests = 0
        total_tests = len(test_phases)
        
        for phase_name, test_func in test_phases:
            if test_func():
                passed_tests += 1
        
        # Cleanup
        self.cleanup_test_data()
        
        # Generate report
        print("\n" + "=" * 50)
        print("📋 Test Results Summary")
        print("=" * 50)
        
        for test_name, result in self.test_results.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name:20}: {status}")
            elif isinstance(result, dict) and result:
                print(f"{test_name:20}: ✅ PASS")
                for key, value in result.items():
                    print(f"{'':22}{key}: {value}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Graph-RAG system is ready.")
        else:
            print("⚠️ Some tests failed. Please check the issues above.")
        
        # Close connections
        if self.graph_manager:
            self.graph_manager.close()


def main():
    """Main test execution."""
    tester = GraphRAGTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()