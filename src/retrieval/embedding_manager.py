# src/retrieval/embedding_manager.py

"""
Embedding manager for the RAG pipeline.
Handles sentence embedding generation using HuggingFace transformers.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import json
import joblib
from datetime import datetime

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings for the RAG pipeline."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            
            self.is_loaded = True
            logger.info(f"✅ Embedding model loaded. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings (n_texts x embedding_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        logger.info(f"✅ Generated {embeddings.shape[0]} embeddings")
        return embeddings
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            numpy array embedding (1 x embedding_dim)
        """
        return self.encode_texts([text])
    
    def prepare_ticket_text(self, ticket_data: Dict[str, Any]) -> str:
        """
        Prepare ticket text for embedding by combining relevant fields.
        
        Args:
            ticket_data: Dictionary containing ticket information
            
        Returns:
            Combined text string for embedding
        """
        text_parts = []
        
        # Add subject
        if ticket_data.get("subject"):
            text_parts.append(f"Subject: {ticket_data['subject']}")
        
        # Add description
        if ticket_data.get("description"):
            text_parts.append(f"Description: {ticket_data['description']}")
        
        # Add error logs
        if ticket_data.get("error_logs"):
            text_parts.append(f"Error: {ticket_data['error_logs']}")
        
        # Add stack trace (truncated if too long)
        if ticket_data.get("stack_trace"):
            stack_trace = ticket_data["stack_trace"]
            if len(stack_trace) > 500:
                stack_trace = stack_trace[:500] + "..."
            text_parts.append(f"Stack trace: {stack_trace}")
        
        # Add product context
        if ticket_data.get("product"):
            text_parts.append(f"Product: {ticket_data['product']}")
        
        # Add category if available
        if ticket_data.get("category"):
            text_parts.append(f"Category: {ticket_data['category']}")
        
        combined_text = " | ".join(text_parts)
        
        # Ensure text isn't too long (some models have limits)
        if len(combined_text) > 2000:
            combined_text = combined_text[:2000] + "..."
        
        return combined_text
    
    def prepare_resolution_text(self, resolution_data: Dict[str, Any]) -> str:
        """
        Prepare resolution text for embedding.
        
        Args:
            resolution_data: Dictionary containing resolution information
            
        Returns:
            Combined resolution text for embedding
        """
        text_parts = []
        
        # Add resolution description
        if resolution_data.get("resolution"):
            text_parts.append(f"Resolution: {resolution_data['resolution']}")
        
        # Add resolution steps if available
        if resolution_data.get("resolution_steps"):
            text_parts.append(f"Steps: {resolution_data['resolution_steps']}")
        
        # Add category context
        if resolution_data.get("category"):
            text_parts.append(f"Category: {resolution_data['category']}")
        
        # Add product context
        if resolution_data.get("product"):
            text_parts.append(f"Product: {resolution_data['product']}")
        
        combined_text = " | ".join(text_parts)
        
        # Ensure text isn't too long
        if len(combined_text) > 1500:
            combined_text = combined_text[:1500] + "..."
        
        return combined_text


def generate_mock_resolutions(num_resolutions: int = 100) -> List[Dict[str, Any]]:
    """
    Generate mock resolution data for testing and demonstration.
    
    Args:
        num_resolutions: Number of mock resolutions to generate
        
    Returns:
        List of resolution dictionaries
    """
    import random
    
    categories = [
        "authentication", "database", "payment", "api", "security", 
        "performance", "ui", "integration", "backup", "network"
    ]
    
    products = [
        "web_application", "mobile_app", "api_server", "database", 
        "payment_gateway", "user_portal", "admin_panel"
    ]
    
    resolution_templates = {
        "authentication": [
            "Clear user session cache and retry login",
            "Reset password and verify email address", 
            "Check authentication service logs for errors",
            "Verify user permissions and role assignments",
            "Update authentication tokens and refresh session"
        ],
        "database": [
            "Restart database service and check connections",
            "Optimize database queries and update indexes",
            "Check database disk space and clean up logs",
            "Review database connection pool settings",
            "Backup database and verify data integrity"
        ],
        "payment": [
            "Verify payment gateway configuration",
            "Check credit card validation rules",
            "Review payment processing logs",
            "Test payment endpoints with valid data",
            "Contact payment provider for status update"
        ],
        "api": [
            "Check API endpoint configuration",
            "Verify request/response format",
            "Review API rate limiting settings",
            "Test API authentication tokens",
            "Check API server resource usage"
        ],
        "security": [
            "Update security certificates",
            "Review firewall and access rules",
            "Scan for security vulnerabilities",
            "Update user access permissions",
            "Enable two-factor authentication"
        ]
    }
    
    resolutions = []
    
    for i in range(num_resolutions):
        category = random.choice(categories)
        product = random.choice(products)
        
        # Get resolution template
        if category in resolution_templates:
            resolution_base = random.choice(resolution_templates[category])
        else:
            resolution_base = "Standard troubleshooting procedure applied"
        
        # Add some variation
        resolution_steps = [
            "1. Identify the root cause of the issue",
            f"2. {resolution_base}",
            "3. Test the solution thoroughly",
            "4. Monitor for any recurring issues",
            "5. Document the resolution for future reference"
        ]
        
        resolution = {
            "resolution_id": f"RES-{i+1:04d}",
            "category": category,
            "product": product,
            "resolution": resolution_base,
            "resolution_steps": " | ".join(resolution_steps),
            "success_rate": random.uniform(0.7, 0.98),
            "usage_count": random.randint(1, 50),
            "created_date": "2024-01-01",
            "last_updated": "2025-09-29"
        }
        
        resolutions.append(resolution)
    
    return resolutions


def save_embeddings_metadata(embeddings: np.ndarray, metadata: List[Dict], output_dir: str):
    """
    Save embeddings and metadata to disk.
    
    Args:
        embeddings: numpy array of embeddings
        metadata: List of metadata dictionaries
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Saved {len(embeddings)} embeddings to {output_dir}")


def load_embeddings_metadata(input_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load embeddings and metadata from disk.
    
    Args:
        input_dir: Directory containing embeddings and metadata files
        
    Returns:
        Tuple of (embeddings array, metadata list)
    """
    input_path = Path(input_dir)
    
    # Load embeddings
    embeddings_path = input_path / "embeddings.npy"
    embeddings = np.load(embeddings_path)
    
    # Load metadata
    metadata_path = input_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"✅ Loaded {len(embeddings)} embeddings from {input_dir}")
    return embeddings, metadata


if __name__ == "__main__":
    # Test the embedding manager
    manager = EmbeddingManager()
    
    try:
        manager.load_model()
        
        # Test ticket text preparation
        test_ticket = {
            "subject": "Login authentication failure",
            "description": "User cannot login with correct credentials",
            "error_logs": "Authentication timeout after 30 seconds",
            "product": "web_application",
            "category": "authentication"
        }
        
        ticket_text = manager.prepare_ticket_text(test_ticket)
        print(f"Prepared ticket text: {ticket_text}")
        
        # Test embedding generation
        embeddings = manager.encode_texts([ticket_text])
        print(f"Generated embedding shape: {embeddings.shape}")
        
        # Test mock resolutions
        mock_resolutions = generate_mock_resolutions(5)
        print(f"Generated {len(mock_resolutions)} mock resolutions")
        
        for i, res in enumerate(mock_resolutions[:2]):
            res_text = manager.prepare_resolution_text(res)
            print(f"Resolution {i+1}: {res_text[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to install sentence-transformers: pip install sentence-transformers")