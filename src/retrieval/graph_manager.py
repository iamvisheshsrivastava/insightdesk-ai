# src/retrieval/graph_manager.py

"""
Neo4j Graph Manager for Graph-RAG implementation.

This module provides graph-based retrieval capabilities using Neo4j,
implementing knowledge graph for products, issues, resolutions, and tickets.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class GraphResult:
    """Result from graph query."""
    products: List[Dict[str, Any]]
    issues: List[Dict[str, Any]]
    resolutions: List[Dict[str, Any]]
    paths: List[Dict[str, Any]]
    relevance_score: float
    success_rate: float


class Neo4jGraphManager:
    """Neo4j-based knowledge graph manager."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """Initialize Neo4j connection."""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        
        self.connect()
        self.setup_schema()
        
        logger.info(f"Neo4j Graph Manager initialized: {uri}")
    
    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def setup_schema(self):
        """Create indexes and constraints for the graph schema."""
        if not self.driver:
            return
        
        schema_queries = [
            # Create constraints for unique identifiers
            "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT issue_id IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT resolution_id IF NOT EXISTS FOR (r:Resolution) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT ticket_id IF NOT EXISTS FOR (t:Ticket) REQUIRE t.id IS UNIQUE",
            
            # Create indexes for better query performance
            "CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)",
            "CREATE INDEX issue_type IF NOT EXISTS FOR (i:Issue) ON (i.type)",
            "CREATE INDEX issue_category IF NOT EXISTS FOR (i:Issue) ON (i.category)",
            "CREATE INDEX resolution_type IF NOT EXISTS FOR (r:Resolution) ON (r.type)",
            "CREATE INDEX ticket_category IF NOT EXISTS FOR (t:Ticket) ON (t.category)",
            "CREATE INDEX ticket_priority IF NOT EXISTS FOR (t:Ticket) ON (t.priority)"
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for query in schema_queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Constraint/index might already exist
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Schema query failed: {query} - {e}")
            
            logger.info("✅ Graph schema setup completed")
        except Exception as e:
            logger.error(f"❌ Failed to setup graph schema: {e}")
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False
    
    def create_product_node(self, product_id: str, properties: Dict[str, Any]) -> bool:
        """Create a Product node."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    "MERGE (p:Product {id: $id}) "
                    "SET p += $properties",
                    id=product_id,
                    properties=properties
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create product node {product_id}: {e}")
            return False
    
    def create_issue_node(self, issue_id: str, properties: Dict[str, Any]) -> bool:
        """Create an Issue node."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    "MERGE (i:Issue {id: $id}) "
                    "SET i += $properties",
                    id=issue_id,
                    properties=properties
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create issue node {issue_id}: {e}")
            return False
    
    def create_resolution_node(self, resolution_id: str, properties: Dict[str, Any]) -> bool:
        """Create a Resolution node."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    "MERGE (r:Resolution {id: $id}) "
                    "SET r += $properties",
                    id=resolution_id,
                    properties=properties
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create resolution node {resolution_id}: {e}")
            return False
    
    def create_ticket_node(self, ticket_id: str, properties: Dict[str, Any]) -> bool:
        """Create a Ticket node."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    "MERGE (t:Ticket {id: $id}) "
                    "SET t += $properties",
                    id=ticket_id,
                    properties=properties
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create ticket node {ticket_id}: {e}")
            return False
    
    def create_relationship(
        self, 
        source_id: str, 
        source_label: str,
        target_id: str, 
        target_label: str,
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two nodes."""
        if not self.driver:
            return False
        
        properties = properties or {}
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    f"MATCH (a:{source_label} {{id: $source_id}}) "
                    f"MATCH (b:{target_label} {{id: $target_id}}) "
                    f"MERGE (a)-[r:{relationship_type}]->(b) "
                    "SET r += $properties",
                    source_id=source_id,
                    target_id=target_id,
                    properties=properties
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship {source_id} -> {target_id}: {e}")
            return False
    
    def query_graph(self, issue_description: str, limit: int = 10) -> GraphResult:
        """Query graph for related products and resolutions based on issue description."""
        if not self.driver:
            return GraphResult([], [], [], [], 0.0, 0.0)
        
        # Clean and prepare search terms
        search_terms = self._extract_search_terms(issue_description)
        
        # Build Cypher query
        cypher_query = """
        MATCH (p:Product)-[:PRODUCT_HAS_ISSUE]->(i:Issue)-[:ISSUE_SOLVED_BY_RESOLUTION]->(r:Resolution)
        WHERE ANY(term IN $search_terms WHERE 
            toLower(i.description) CONTAINS toLower(term) OR
            toLower(i.type) CONTAINS toLower(term) OR
            toLower(i.category) CONTAINS toLower(term) OR
            toLower(p.name) CONTAINS toLower(term)
        )
        OPTIONAL MATCH (t:Ticket)-[:TICKET_LINKS_TO_ISSUE]->(i)
        OPTIONAL MATCH (r)-[:RESOLUTION_REFERENCES_KB]->(kb:KnowledgeBase)
        WITH p, i, r, kb, 
             COUNT(DISTINCT t) as ticket_count,
             COLLECT(DISTINCT t {.id, .category, .priority, .resolved}) as related_tickets
        RETURN DISTINCT p {.*, ticket_count: ticket_count} as product,
               i {.*, ticket_count: ticket_count} as issue,
               r {.*, success_rate: COALESCE(r.success_rate, 0.0)} as resolution,
               kb {.*} as knowledge_base,
               related_tickets,
               p.name + ' - ' + i.type + ' - ' + r.type as path_description
        ORDER BY ticket_count DESC, r.success_rate DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, search_terms=search_terms, limit=limit)
                
                products = []
                issues = []
                resolutions = []
                paths = []
                
                total_success_rate = 0.0
                result_count = 0
                
                for record in result:
                    products.append(record["product"])
                    issues.append(record["issue"])
                    resolutions.append(record["resolution"])
                    
                    paths.append({
                        "description": record["path_description"],
                        "product": record["product"],
                        "issue": record["issue"],
                        "resolution": record["resolution"],
                        "related_tickets": record["related_tickets"]
                    })
                    
                    total_success_rate += record["resolution"]["success_rate"]
                    result_count += 1
                
                # Calculate average success rate and relevance score
                avg_success_rate = total_success_rate / result_count if result_count > 0 else 0.0
                relevance_score = self._calculate_relevance_score(search_terms, paths)
                
                return GraphResult(
                    products=products,
                    issues=issues,
                    resolutions=resolutions,
                    paths=paths,
                    relevance_score=relevance_score,
                    success_rate=avg_success_rate
                )
                
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return GraphResult([], [], [], [], 0.0, 0.0)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session(database=self.database) as session:
                # Count nodes by type
                node_counts = {}
                for label in ["Product", "Issue", "Resolution", "Ticket", "KnowledgeBase"]:
                    result = session.run(f"MATCH (n:{label}) RETURN COUNT(n) as count")
                    node_counts[label.lower()] = result.single()["count"]
                
                # Count relationships by type
                rel_counts = {}
                rel_types = ["PRODUCT_HAS_ISSUE", "ISSUE_SOLVED_BY_RESOLUTION", 
                           "TICKET_LINKS_TO_ISSUE", "RESOLUTION_REFERENCES_KB"]
                
                for rel_type in rel_types:
                    result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count")
                    rel_counts[rel_type.lower()] = result.single()["count"]
                
                return {
                    "nodes": node_counts,
                    "relationships": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values())
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {}
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract search terms from text."""
        # Simple implementation - can be enhanced with NLP
        import re
        
        # Remove special characters and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        terms = text.split()
        
        # Filter out common stop words and short terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with'}
        terms = [term for term in terms if len(term) > 2 and term not in stop_words]
        
        return terms[:10]  # Limit to top 10 terms
    
    def _calculate_relevance_score(self, search_terms: List[str], paths: List[Dict[str, Any]]) -> float:
        """Calculate relevance score based on term matches."""
        if not paths or not search_terms:
            return 0.0
        
        total_score = 0.0
        
        for path in paths:
            score = 0.0
            # Check matches in different fields with different weights
            for term in search_terms:
                term_lower = term.lower()
                
                # Issue description (highest weight)
                if term_lower in path["issue"].get("description", "").lower():
                    score += 1.0
                
                # Issue type (high weight)
                if term_lower in path["issue"].get("type", "").lower():
                    score += 0.8
                
                # Product name (medium weight)
                if term_lower in path["product"].get("name", "").lower():
                    score += 0.6
                
                # Resolution type (medium weight)
                if term_lower in path["resolution"].get("type", "").lower():
                    score += 0.5
            
            total_score += score / len(search_terms)  # Normalize by number of terms
        
        return min(total_score / len(paths), 1.0)  # Normalize to 0-1 range


class GraphDataIngestion:
    """Handles data ingestion from tickets.json into Neo4j graph."""
    
    def __init__(self, graph_manager: Neo4jGraphManager):
        """Initialize with graph manager."""
        self.graph_manager = graph_manager
        self.logger = logging.getLogger(__name__)
    
    def ingest_tickets_data(self, tickets_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest tickets data into Neo4j graph."""
        stats = {
            "products": 0,
            "issues": 0,
            "resolutions": 0,
            "tickets": 0,
            "relationships": 0,
            "errors": 0
        }
        
        # Maps to track unique entities
        products_map = {}
        issues_map = {}
        resolutions_map = {}
        
        self.logger.info(f"Starting ingestion of {len(tickets_data)} tickets...")
        
        for i, ticket in enumerate(tickets_data):
            try:
                ticket_id = ticket.get("ticket_id", f"ticket_{i}")
                
                # Extract entities from ticket
                product_info = self._extract_product_info(ticket)
                issue_info = self._extract_issue_info(ticket)
                resolution_info = self._extract_resolution_info(ticket)
                
                # Create unique IDs
                product_id = f"product_{product_info['name'].replace(' ', '_')}"
                issue_id = f"issue_{issue_info['type']}_{issue_info['category']}"
                resolution_id = f"resolution_{resolution_info['type']}_{resolution_info.get('method', 'standard')}"
                
                # Create or update Product node
                if product_id not in products_map:
                    if self.graph_manager.create_product_node(product_id, product_info):
                        products_map[product_id] = product_info
                        stats["products"] += 1
                
                # Create or update Issue node
                if issue_id not in issues_map:
                    if self.graph_manager.create_issue_node(issue_id, issue_info):
                        issues_map[issue_id] = issue_info
                        stats["issues"] += 1
                
                # Create or update Resolution node
                if resolution_id not in resolutions_map:
                    if self.graph_manager.create_resolution_node(resolution_id, resolution_info):
                        resolutions_map[resolution_id] = resolution_info
                        stats["resolutions"] += 1
                
                # Create Ticket node
                ticket_properties = self._extract_ticket_properties(ticket)
                if self.graph_manager.create_ticket_node(ticket_id, ticket_properties):
                    stats["tickets"] += 1
                
                # Create relationships
                relationships = [
                    (product_id, "Product", issue_id, "Issue", "PRODUCT_HAS_ISSUE"),
                    (issue_id, "Issue", resolution_id, "Resolution", "ISSUE_SOLVED_BY_RESOLUTION"),
                    (ticket_id, "Ticket", issue_id, "Issue", "TICKET_LINKS_TO_ISSUE")
                ]
                
                for rel in relationships:
                    if self.graph_manager.create_relationship(*rel):
                        stats["relationships"] += 1
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(tickets_data)} tickets")
                    
            except Exception as e:
                self.logger.error(f"Error processing ticket {i}: {e}")
                stats["errors"] += 1
        
        self.logger.info(f"Ingestion completed: {stats}")
        return stats
    
    def _extract_product_info(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information from ticket."""
        product_name = ticket.get("product", "Unknown Product")
        product_module = ticket.get("product_module", "General")
        
        return {
            "name": product_name,
            "module": product_module,
            "description": f"{product_name} - {product_module}",
            "created_at": datetime.now().isoformat()
        }
    
    def _extract_issue_info(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Extract issue information from ticket."""
        # Determine issue type from ticket content
        issue_type = self._classify_issue_type(ticket)
        category = ticket.get("category", "general")
        
        return {
            "type": issue_type,
            "category": category,
            "description": ticket.get("description", ""),
            "severity": ticket.get("severity", "medium"),
            "priority": ticket.get("priority", "medium"),
            "frequency": 1,  # Will be updated later based on similar issues
            "created_at": datetime.now().isoformat()
        }
    
    def _extract_resolution_info(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Extract resolution information from ticket."""
        # Determine resolution type from ticket resolution
        resolution_type = self._classify_resolution_type(ticket)
        
        return {
            "type": resolution_type,
            "method": self._extract_resolution_method(ticket),
            "description": ticket.get("resolution", ""),
            "success_rate": 0.85,  # Default - will be updated based on feedback
            "avg_resolution_time": ticket.get("resolution_time_hours", 24.0),
            "requires_escalation": ticket.get("priority") in ["high", "critical"],
            "created_at": datetime.now().isoformat()
        }
    
    def _extract_ticket_properties(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ticket properties."""
        return {
            "subject": ticket.get("subject", ""),
            "description": ticket.get("description", ""),
            "category": ticket.get("category", "general"),
            "priority": ticket.get("priority", "medium"),
            "severity": ticket.get("severity", "medium"),
            "status": ticket.get("status", "open"),
            "resolved": ticket.get("resolved", False),
            "customer_tier": ticket.get("customer_tier", "standard"),
            "channel": ticket.get("channel", "email"),
            "created_at": ticket.get("created_at", datetime.now().isoformat())
        }
    
    def _classify_issue_type(self, ticket: Dict[str, Any]) -> str:
        """Classify issue type based on ticket content."""
        description = ticket.get("description", "").lower()
        subject = ticket.get("subject", "").lower()
        text = f"{subject} {description}"
        
        # Simple keyword-based classification
        if any(word in text for word in ["timeout", "slow", "performance", "latency"]):
            return "performance_issue"
        elif any(word in text for word in ["login", "authentication", "password"]):
            return "authentication_issue"
        elif any(word in text for word in ["error", "exception", "crash", "bug"]):
            return "application_error"
        elif any(word in text for word in ["config", "configuration", "setting"]):
            return "configuration_issue"
        elif any(word in text for word in ["integration", "api", "connection"]):
            return "integration_issue"
        else:
            return "general_issue"
    
    def _classify_resolution_type(self, ticket: Dict[str, Any]) -> str:
        """Classify resolution type based on ticket content."""
        resolution = ticket.get("resolution", "").lower()
        category = ticket.get("category", "").lower()
        
        if any(word in resolution for word in ["config", "configuration", "setting"]):
            return "configuration_change"
        elif any(word in resolution for word in ["restart", "reboot", "reload"]):
            return "system_restart"
        elif any(word in resolution for word in ["update", "patch", "upgrade"]):
            return "software_update"
        elif any(word in resolution for word in ["permission", "access", "role"]):
            return "permission_fix"
        elif any(word in resolution for word in ["documentation", "guide", "manual"]):
            return "documentation_provided"
        else:
            return "standard_fix"
    
    def _extract_resolution_method(self, ticket: Dict[str, Any]) -> str:
        """Extract resolution method from ticket."""
        resolution = ticket.get("resolution", "").lower()
        
        if "automatic" in resolution:
            return "automated"
        elif any(word in resolution for word in ["escalate", "tier", "specialist"]):
            return "escalated"
        elif "self" in resolution or "customer" in resolution:
            return "self_service"
        else:
            return "agent_assisted"