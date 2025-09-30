# scripts/demo_graph_rag.py

"""
Graph-RAG Demo Script

This script demonstrates the Graph-RAG capabilities including:
1. Knowledge graph visualization
2. Hybrid search comparisons
3. Interactive query examples
4. Performance demonstrations
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.graph_manager import Neo4jGraphManager
from src.retrieval.rag_pipeline import RAGPipeline


class GraphRAGDemo:
    """Interactive demonstration of Graph-RAG capabilities."""
    
    def __init__(self):
        self.graph_manager = None
        self.rag_pipeline = None
        
        # Neo4j configuration
        self.neo4j_config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",  # Change this to your Neo4j password
            "database": "neo4j"
        }
        
        # Demo queries
        self.demo_queries = [
            {
                "title": "Authentication Issues",
                "ticket": {
                    "subject": "Cannot login to application",
                    "description": "Users are unable to authenticate with correct credentials. Getting timeout error after 30 seconds.",
                    "product": "web_application",
                    "category": "authentication"
                }
            },
            {
                "title": "Database Connection Problems",
                "ticket": {
                    "subject": "Database connection timeout",
                    "description": "Application loses connection to database frequently. Users see error messages.",
                    "product": "database_service",
                    "category": "infrastructure"
                }
            },
            {
                "title": "Payment Processing Error",
                "ticket": {
                    "subject": "Payment fails with error code 500",
                    "description": "Customers cannot complete purchases. Payment gateway returns internal server error.",
                    "product": "payment_system",
                    "category": "payment"
                }
            },
            {
                "title": "UI Performance Issues",
                "ticket": {
                    "subject": "Application runs slowly",
                    "description": "Page load times are very slow, especially on the dashboard. Users are frustrated.",
                    "product": "web_application",
                    "category": "performance"
                }
            }
        ]
    
    def setup(self):
        """Initialize the demo environment."""
        print("🚀 Initializing Graph-RAG Demo Environment")
        print("=" * 50)
        
        try:
            # Initialize graph manager
            print("📊 Connecting to Neo4j...")
            self.graph_manager = Neo4jGraphManager(**self.neo4j_config)
            
            # Test connection
            stats = self.graph_manager.get_graph_stats()
            print(f"✅ Connected to Neo4j. Graph contains:")
            for node_type, count in stats.get("nodes", {}).items():
                print(f"   {node_type.title()}: {count} nodes")
            for rel_type, count in stats.get("relationships", {}).items():
                print(f"   {rel_type.replace('_', ' ').title()}: {count} relationships")
            
            # Initialize traditional RAG (if available)
            try:
                print("\n📚 Initializing traditional RAG pipeline...")
                self.rag_pipeline = RAGPipeline(vector_store_dir="vector_store")
                self.rag_pipeline.initialize()
                print("✅ Traditional RAG pipeline ready")
            except Exception as e:
                print(f"⚠️ Traditional RAG not available: {e}")
                self.rag_pipeline = None
            
            print("\n🎯 Demo environment ready!")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def display_graph_schema(self):
        """Display the knowledge graph schema."""
        print("\n🏗️ Knowledge Graph Schema")
        print("=" * 30)
        
        print("""
        📦 NODES:
        ├── Product (applications, services)
        ├── Issue (problems, bugs, errors)
        ├── Resolution (solutions, fixes)
        ├── Ticket (support requests)
        ├── KnowledgeBase (documentation)
        └── Category (classification)
        
        🔗 RELATIONSHIPS:
        ├── PRODUCT_HAS_ISSUE
        ├── ISSUE_SOLVED_BY_RESOLUTION
        ├── TICKET_LINKS_TO_ISSUE
        ├── RESOLUTION_REFERENCES_KB
        ├── SIMILAR_TO (between issues)
        └── HAS_CATEGORY
        """)
    
    def demonstrate_traditional_rag(self, ticket_data: Dict[str, Any]):
        """Demonstrate traditional RAG search."""
        print("\n🔍 Traditional RAG Search")
        print("-" * 25)
        
        if not self.rag_pipeline:
            print("⚠️ Traditional RAG not available")
            return []
        
        try:
            start_time = time.time()
            results = self.rag_pipeline.query_solutions(ticket_data, k=5)
            search_time = time.time() - start_time
            
            print(f"⏱️ Search time: {search_time:.3f}s")
            print(f"📊 Found {len(results)} results")
            
            for i, result in enumerate(results[:3], 1):
                print(f"\n{i}. Resolution: {result.get('resolution_id', 'N/A')}")
                print(f"   Category: {result.get('category', 'N/A')}")
                print(f"   Similarity: {result.get('similarity_score', 0.0):.3f}")
                print(f"   Description: {result.get('resolution', 'N/A')[:100]}...")
            
            return results
            
        except Exception as e:
            print(f"❌ Traditional RAG search failed: {e}")
            return []
    
    def demonstrate_graph_search(self, ticket_data: Dict[str, Any]):
        """Demonstrate graph-based search."""
        print("\n🕸️ Graph-Based Search")
        print("-" * 22)
        
        try:
            # Create search query from ticket
            search_terms = f"{ticket_data['subject']} {ticket_data['description']}"
            
            start_time = time.time()
            results = self.graph_manager.query_graph(
                search_terms,
                search_type="comprehensive",
                limit=5
            )
            search_time = time.time() - start_time
            
            print(f"⏱️ Search time: {search_time:.3f}s")
            print(f"📊 Found {len(results)} results")
            
            for i, result in enumerate(results[:3], 1):
                print(f"\n{i}. Resolution: {result.get('resolution_id', 'N/A')}")
                print(f"   Type: {result.get('resolution_type', 'N/A')}")
                print(f"   Relevance: {result.get('relevance_score', 0.0):.3f}")
                print(f"   Success Rate: {result.get('success_rate', 0.0):.1%}")
                if result.get('relationship_path'):
                    print(f"   Path: {' → '.join(result['relationship_path'][:3])}")
            
            return results
            
        except Exception as e:
            print(f"❌ Graph search failed: {e}")
            return []
    
    def demonstrate_graph_traversal(self, ticket_data: Dict[str, Any]):
        """Demonstrate graph relationship traversal."""
        print("\n🔗 Graph Relationship Traversal")
        print("-" * 32)
        
        try:
            # Find related products
            if ticket_data.get('product'):
                print(f"🔍 Finding issues for product: {ticket_data['product']}")
                
                product_issues = self.graph_manager.execute_query(
                    """
                    MATCH (p:Product {name: $product})-[:PRODUCT_HAS_ISSUE]->(i:Issue)
                    RETURN i.type as issue_type, i.category as category, COUNT(*) as count
                    ORDER BY count DESC
                    LIMIT 5
                    """,
                    {"product": ticket_data['product']}
                )
                
                for record in product_issues:
                    print(f"   {record['issue_type']} ({record['category']}): {record['count']} occurrences")
            
            # Find similar issues
            if ticket_data.get('category'):
                print(f"\n🔍 Finding similar issues in category: {ticket_data['category']}")
                
                similar_issues = self.graph_manager.execute_query(
                    """
                    MATCH (i:Issue {category: $category})-[:ISSUE_SOLVED_BY_RESOLUTION]->(r:Resolution)
                    RETURN i.type as issue_type, r.type as resolution_type, r.success_rate as success_rate
                    ORDER BY r.success_rate DESC
                    LIMIT 5
                    """,
                    {"category": ticket_data['category']}
                )
                
                for record in similar_issues:
                    print(f"   {record['issue_type']} → {record['resolution_type']} "
                          f"({record['success_rate']:.1%} success)")
            
            # Find knowledge base articles
            print(f"\n📚 Finding related knowledge base articles...")
            
            kb_articles = self.graph_manager.execute_query(
                """
                MATCH (r:Resolution)-[:RESOLUTION_REFERENCES_KB]->(kb:KnowledgeBase)
                WHERE r.type CONTAINS $search_term OR kb.category CONTAINS $search_term
                RETURN kb.title as title, kb.category as category, kb.rating as rating
                ORDER BY kb.rating DESC
                LIMIT 3
                """,
                {"search_term": ticket_data.get('category', '')}
            )
            
            for record in kb_articles:
                print(f"   📖 {record['title']} ({record['category']}) - "
                      f"Rating: {record['rating']:.1f}/5.0")
        
        except Exception as e:
            print(f"❌ Graph traversal failed: {e}")
    
    def demonstrate_hybrid_ranking(self, rag_results: List[Dict], graph_results: List[Dict]):
        """Demonstrate hybrid ranking algorithm."""
        print("\n⚖️ Hybrid Ranking Demonstration")
        print("-" * 33)
        
        if not rag_results and not graph_results:
            print("⚠️ No results to rank")
            return
        
        print("🔄 Combining semantic similarity with graph relevance...")
        
        # Simulate hybrid scoring
        semantic_weight = 0.6
        graph_weight = 0.4
        
        print(f"   Semantic weight: {semantic_weight}")
        print(f"   Graph weight: {graph_weight}")
        
        # Show top results from each method
        print(f"\n📊 Top Traditional RAG Result:")
        if rag_results:
            top_rag = rag_results[0]
            print(f"   Resolution: {top_rag.get('resolution_id', 'N/A')}")
            print(f"   Semantic Score: {top_rag.get('similarity_score', 0.0):.3f}")
        
        print(f"\n📊 Top Graph Result:")
        if graph_results:
            top_graph = graph_results[0]
            print(f"   Resolution: {top_graph.get('resolution_id', 'N/A')}")
            print(f"   Graph Score: {top_graph.get('relevance_score', 0.0):.3f}")
            print(f"   Success Rate: {top_graph.get('success_rate', 0.0):.1%}")
        
        print(f"\n🎯 Hybrid ranking would combine these signals to provide")
        print(f"   the most contextually relevant solutions.")
    
    def run_interactive_demo(self):
        """Run an interactive demonstration."""
        print("\n🎮 Interactive Graph-RAG Demo")
        print("=" * 30)
        
        while True:
            print("\nAvailable demo scenarios:")
            for i, query in enumerate(self.demo_queries, 1):
                print(f"{i}. {query['title']}")
            print("0. Exit demo")
            
            try:
                choice = input("\nSelect a scenario (0-4): ").strip()
                
                if choice == "0":
                    break
                
                demo_idx = int(choice) - 1
                if 0 <= demo_idx < len(self.demo_queries):
                    self.run_scenario_demo(self.demo_queries[demo_idx])
                else:
                    print("❌ Invalid choice. Please try again.")
                
            except ValueError:
                print("❌ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
    
    def run_scenario_demo(self, scenario: Dict[str, Any]):
        """Run a complete demo for a specific scenario."""
        print(f"\n🎯 Scenario: {scenario['title']}")
        print("=" * (len(scenario['title']) + 12))
        
        ticket_data = scenario['ticket']
        
        print("📋 Ticket Information:")
        print(f"   Subject: {ticket_data['subject']}")
        print(f"   Description: {ticket_data['description']}")
        print(f"   Product: {ticket_data['product']}")
        print(f"   Category: {ticket_data['category']}")
        
        # Demonstrate traditional RAG
        rag_results = self.demonstrate_traditional_rag(ticket_data)
        
        # Demonstrate graph search
        graph_results = self.demonstrate_graph_search(ticket_data)
        
        # Demonstrate graph traversal
        self.demonstrate_graph_traversal(ticket_data)
        
        # Demonstrate hybrid ranking
        self.demonstrate_hybrid_ranking(rag_results, graph_results)
        
        input("\nPress Enter to continue...")
    
    def show_graph_analytics(self):
        """Show graph analytics and insights."""
        print("\n📈 Graph Analytics & Insights")
        print("=" * 30)
        
        try:
            # Top issues by frequency
            top_issues = self.graph_manager.execute_query(
                """
                MATCH (i:Issue)<-[:TICKET_LINKS_TO_ISSUE]-(t:Ticket)
                RETURN i.type as issue_type, i.category as category, COUNT(t) as ticket_count
                ORDER BY ticket_count DESC
                LIMIT 5
                """
            )
            
            print("🔥 Most Common Issues:")
            for record in top_issues:
                print(f"   {record['issue_type']} ({record['category']}): "
                      f"{record['ticket_count']} tickets")
            
            # Most effective resolutions
            effective_resolutions = self.graph_manager.execute_query(
                """
                MATCH (r:Resolution)
                WHERE r.success_rate IS NOT NULL
                RETURN r.type as resolution_type, r.success_rate as success_rate, COUNT(*) as usage_count
                ORDER BY r.success_rate DESC, usage_count DESC
                LIMIT 5
                """
            )
            
            print("\n✅ Most Effective Resolutions:")
            for record in effective_resolutions:
                print(f"   {record['resolution_type']}: "
                      f"{record['success_rate']:.1%} success rate "
                      f"({record['usage_count']} instances)")
            
            # Product health overview
            product_health = self.graph_manager.execute_query(
                """
                MATCH (p:Product)-[:PRODUCT_HAS_ISSUE]->(i:Issue)
                RETURN p.name as product, COUNT(i) as issue_count
                ORDER BY issue_count DESC
                LIMIT 5
                """
            )
            
            print("\n🏥 Product Health Overview:")
            for record in product_health:
                print(f"   {record['product']}: {record['issue_count']} known issues")
        
        except Exception as e:
            print(f"❌ Analytics failed: {e}")
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        if not self.setup():
            return
        
        try:
            # Display schema
            self.display_graph_schema()
            
            # Show analytics
            self.show_graph_analytics()
            
            # Run interactive demo
            self.run_interactive_demo()
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
        finally:
            if self.graph_manager:
                self.graph_manager.close()
                print("🔌 Connections closed")


def main():
    """Main demo execution."""
    print("🌟 Welcome to the Graph-RAG Interactive Demo!")
    print("This demo showcases the power of combining semantic search")
    print("with knowledge graph relationships for enhanced AI support.")
    print()
    
    demo = GraphRAGDemo()
    demo.run_full_demo()
    
    print("\n🎉 Thank you for exploring Graph-RAG!")
    print("For more information, check the documentation and API endpoints.")


if __name__ == "__main__":
    main()