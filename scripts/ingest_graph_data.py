# scripts/ingest_graph_data.py

"""
Data ingestion script for Graph-RAG with Neo4j.

This script loads tickets data and creates the knowledge graph
with products, issues, resolutions, and their relationships.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.graph_manager import Neo4jGraphManager, GraphDataIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tickets_data(file_path: str) -> List[Dict[str, Any]]:
    """Load tickets data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both direct list and nested structure
        if isinstance(data, list):
            tickets = data
        elif isinstance(data, dict) and 'tickets' in data:
            tickets = data['tickets']
        else:
            logger.error(f"Unexpected data structure in {file_path}")
            return []
        
        logger.info(f"Loaded {len(tickets)} tickets from {file_path}")
        return tickets
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading tickets data: {e}")
        return []


def validate_neo4j_connection(graph_manager: Neo4jGraphManager) -> bool:
    """Validate Neo4j connection and return status."""
    try:
        stats = graph_manager.get_graph_stats()
        logger.info(f"✅ Neo4j connection validated. Current stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        return False


def create_sample_kb_nodes(graph_manager: Neo4jGraphManager) -> int:
    """Create sample knowledge base nodes for RESOLUTION_REFERENCES_KB relationships."""
    kb_entries = [
        {
            "id": "kb_timeout_config",
            "title": "Timeout Configuration Guide",
            "content": "How to configure timeouts in various components",
            "category": "configuration",
            "rating": 4.5,
            "views": 1250
        },
        {
            "id": "kb_auth_troubleshooting", 
            "title": "Authentication Troubleshooting",
            "content": "Step-by-step guide to resolve authentication issues",
            "category": "authentication",
            "rating": 4.8,
            "views": 2100
        },
        {
            "id": "kb_performance_tuning",
            "title": "Performance Optimization Guide",
            "content": "Best practices for improving system performance",
            "category": "performance",
            "rating": 4.3,
            "views": 890
        },
        {
            "id": "kb_error_codes",
            "title": "Common Error Codes Reference",
            "content": "Reference guide for common error codes and their meanings",
            "category": "troubleshooting",
            "rating": 4.6,
            "views": 1875
        }
    ]
    
    created_count = 0
    
    try:
        with graph_manager.driver.session(database=graph_manager.database) as session:
            for kb in kb_entries:
                session.run(
                    "MERGE (kb:KnowledgeBase {id: $id}) "
                    "SET kb += $properties",
                    id=kb["id"],
                    properties=kb
                )
                created_count += 1
                
        logger.info(f"✅ Created {created_count} knowledge base entries")
        return created_count
        
    except Exception as e:
        logger.error(f"❌ Failed to create KB nodes: {e}")
        return 0


def create_kb_relationships(graph_manager: Neo4jGraphManager) -> int:
    """Create relationships between resolutions and knowledge base articles."""
    relationships = [
        ("resolution_configuration_change_standard", "kb_timeout_config"),
        ("resolution_configuration_change_automated", "kb_timeout_config"),
        ("resolution_standard_fix_agent_assisted", "kb_auth_troubleshooting"),
        ("resolution_permission_fix_agent_assisted", "kb_auth_troubleshooting"),
        ("resolution_software_update_automated", "kb_performance_tuning"),
        ("resolution_system_restart_automated", "kb_performance_tuning"),
        ("resolution_standard_fix_escalated", "kb_error_codes")
    ]
    
    created_count = 0
    
    try:
        with graph_manager.driver.session(database=graph_manager.database) as session:
            for resolution_id, kb_id in relationships:
                result = session.run(
                    "MATCH (r:Resolution {id: $resolution_id}) "
                    "MATCH (kb:KnowledgeBase {id: $kb_id}) "
                    "MERGE (r)-[rel:RESOLUTION_REFERENCES_KB]->(kb) "
                    "RETURN COUNT(rel) as created",
                    resolution_id=resolution_id,
                    kb_id=kb_id
                )
                
                if result.single()["created"] > 0:
                    created_count += 1
                    
        logger.info(f"✅ Created {created_count} KB relationships")
        return created_count
        
    except Exception as e:
        logger.error(f"❌ Failed to create KB relationships: {e}")
        return 0


def update_success_rates(graph_manager: Neo4jGraphManager) -> int:
    """Update resolution success rates based on resolution types."""
    success_rate_mapping = {
        "configuration_change": 0.92,
        "system_restart": 0.85,
        "software_update": 0.88,
        "permission_fix": 0.95,
        "documentation_provided": 0.75,
        "standard_fix": 0.80
    }
    
    updated_count = 0
    
    try:
        with graph_manager.driver.session(database=graph_manager.database) as session:
            for resolution_type, success_rate in success_rate_mapping.items():
                result = session.run(
                    "MATCH (r:Resolution) "
                    "WHERE r.type = $resolution_type "
                    "SET r.success_rate = $success_rate "
                    "RETURN COUNT(r) as updated",
                    resolution_type=resolution_type,
                    success_rate=success_rate
                )
                
                count = result.single()["updated"]
                updated_count += count
                logger.info(f"Updated {count} resolutions of type '{resolution_type}' with success rate {success_rate}")
                
        logger.info(f"✅ Updated success rates for {updated_count} resolutions")
        return updated_count
        
    except Exception as e:
        logger.error(f"❌ Failed to update success rates: {e}")
        return 0


def create_additional_relationships(graph_manager: Neo4jGraphManager) -> Dict[str, int]:
    """Create additional useful relationships in the graph."""
    stats = {"similar_issues": 0, "product_categories": 0}
    
    try:
        with graph_manager.driver.session(database=graph_manager.database) as session:
            # Create SIMILAR_TO relationships between issues of same type
            result = session.run("""
                MATCH (i1:Issue), (i2:Issue)
                WHERE i1.type = i2.type AND i1.id < i2.id
                MERGE (i1)-[r:SIMILAR_TO]->(i2)
                SET r.similarity_score = 0.8
                RETURN COUNT(r) as created
            """)
            stats["similar_issues"] = result.single()["created"]
            
            # Create HAS_CATEGORY relationships from products to categories
            result = session.run("""
                MATCH (p:Product)-[:PRODUCT_HAS_ISSUE]->(i:Issue)
                WITH p, i.category as category, COUNT(i) as issue_count
                MERGE (c:Category {name: category})
                MERGE (p)-[r:HAS_CATEGORY]->(c)
                SET r.issue_count = issue_count
                RETURN COUNT(r) as created
            """)
            stats["product_categories"] = result.single()["created"]
            
        logger.info(f"✅ Created additional relationships: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to create additional relationships: {e}")
        return stats


def analyze_graph_structure(graph_manager: Neo4jGraphManager) -> Dict[str, Any]:
    """Analyze the created graph structure."""
    analysis = {}
    
    try:
        with graph_manager.driver.session(database=graph_manager.database) as session:
            # Get node counts
            analysis["nodes"] = {}
            for label in ["Product", "Issue", "Resolution", "Ticket", "KnowledgeBase", "Category"]:
                result = session.run(f"MATCH (n:{label}) RETURN COUNT(n) as count")
                analysis["nodes"][label.lower()] = result.single()["count"]
            
            # Get relationship counts
            analysis["relationships"] = {}
            rel_types = [
                "PRODUCT_HAS_ISSUE", "ISSUE_SOLVED_BY_RESOLUTION", 
                "TICKET_LINKS_TO_ISSUE", "RESOLUTION_REFERENCES_KB",
                "SIMILAR_TO", "HAS_CATEGORY"
            ]
            
            for rel_type in rel_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count")
                analysis["relationships"][rel_type.lower()] = result.single()["count"]
            
            # Get top issues by ticket count
            result = session.run("""
                MATCH (i:Issue)<-[:TICKET_LINKS_TO_ISSUE]-(t:Ticket)
                RETURN i.type as issue_type, i.category as category, COUNT(t) as ticket_count
                ORDER BY ticket_count DESC
                LIMIT 5
            """)
            analysis["top_issues"] = [dict(record) for record in result]
            
            # Get resolution success rates
            result = session.run("""
                MATCH (r:Resolution)
                RETURN r.type as resolution_type, AVG(r.success_rate) as avg_success_rate, COUNT(r) as count
                ORDER BY avg_success_rate DESC
            """)
            analysis["resolution_stats"] = [dict(record) for record in result]
            
        logger.info(f"📊 Graph analysis completed: {analysis}")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze graph: {e}")
        return {}


def main():
    """Main ingestion process."""
    print("🚀 Graph-RAG Data Ingestion for Neo4j")
    print("=" * 40)
    
    # Configuration
    neo4j_config = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j", 
        "password": "password",  # Change this to your Neo4j password
        "database": "neo4j"
    }
    
    tickets_file = "data/support_tickets.json"
    
    try:
        # Initialize Graph Manager
        print("📋 Initializing Neo4j Graph Manager...")
        graph_manager = Neo4jGraphManager(**neo4j_config)
        
        # Validate connection
        if not validate_neo4j_connection(graph_manager):
            print("❌ Cannot proceed without valid Neo4j connection")
            return
        
        # Optional: Clear existing graph
        clear_existing = input("Clear existing graph data? (y/N): ").lower().strip()
        if clear_existing == 'y':
            print("🧹 Clearing existing graph...")
            graph_manager.clear_graph()
        
        # Load tickets data
        print(f"📂 Loading tickets from {tickets_file}...")
        tickets_data = load_tickets_data(tickets_file)
        
        if not tickets_data:
            print("❌ No tickets data loaded. Cannot proceed.")
            return
        
        # Initialize data ingestion
        print("🔄 Starting data ingestion...")
        ingestion = GraphDataIngestion(graph_manager)
        
        # Ingest tickets data
        ingestion_stats = ingestion.ingest_tickets_data(tickets_data)
        print(f"✅ Ingestion completed: {ingestion_stats}")
        
        # Create knowledge base entries
        print("📚 Creating knowledge base entries...")
        kb_count = create_sample_kb_nodes(graph_manager)
        
        # Create KB relationships
        print("🔗 Creating knowledge base relationships...")
        kb_rel_count = create_kb_relationships(graph_manager)
        
        # Update success rates
        print("📈 Updating resolution success rates...")
        success_rate_updates = update_success_rates(graph_manager)
        
        # Create additional relationships
        print("🌐 Creating additional relationships...")
        additional_rels = create_additional_relationships(graph_manager)
        
        # Analyze final graph structure
        print("📊 Analyzing graph structure...")
        analysis = analyze_graph_structure(graph_manager)
        
        # Final summary
        print("\n🎉 Graph-RAG Ingestion Complete!")
        print("=" * 40)
        print(f"📊 Final Statistics:")
        print(f"  Nodes: {analysis.get('nodes', {})}")
        print(f"  Relationships: {analysis.get('relationships', {})}")
        print(f"  Knowledge Base Entries: {kb_count}")
        print(f"  KB Relationships: {kb_rel_count}")
        print(f"  Additional Relationships: {additional_rels}")
        
        if analysis.get('top_issues'):
            print(f"\n🔥 Top Issues by Ticket Count:")
            for issue in analysis['top_issues']:
                print(f"  {issue['issue_type']} ({issue['category']}): {issue['ticket_count']} tickets")
        
        if analysis.get('resolution_stats'):
            print(f"\n✅ Resolution Success Rates:")
            for stat in analysis['resolution_stats']:
                print(f"  {stat['resolution_type']}: {stat['avg_success_rate']:.1%} ({stat['count']} resolutions)")
        
        print(f"\n🔍 Graph ready for queries!")
        print(f"   Access Neo4j Browser: http://localhost:7474")
        
    except KeyboardInterrupt:
        print("\n⏹️ Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        print(f"❌ Error: {e}")
    finally:
        try:
            graph_manager.close()
        except:
            pass


if __name__ == "__main__":
    main()