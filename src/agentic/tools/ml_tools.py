"""
ML Tools for Agentic AI System

This module provides machine learning tools that the agent can use during the
Plan-Act-Observe-Reflect (PAOR) loop. These tools enable the agent to analyze
and classify support tickets using ML models.

Current Tools:
    - ClassificationTool: Categorizes support tickets into predefined categories

Architecture Note:
    This is a mock implementation that uses keyword-based heuristics. In production,
    this would integrate with the actual ML pipeline from src.models.prediction_pipeline
    to leverage trained classification models.
"""

# src/agentic/tools/ml_tools.py
from typing import Any, Dict, List
import logging

# ========== Future Integration Points ==========
# In production, these imports would be active:
# from src.models.prediction_pipeline import PredictionPipeline
# from src.models.model_loader import load_classification_model
# 
# The models would be loaded during initialization and reused across
# multiple ticket classifications for efficiency.

logger = logging.getLogger(__name__)


class ClassificationTool:
    """
    Tool for classifying support tickets into categories using ML models.
    
    This tool is designed to be called by the agent orchestrator during the
    "Act" phase of the PAOR loop. It analyzes ticket content and predicts
    the most appropriate category, which helps route tickets and retrieve
    relevant solutions.
    
    Attributes:
        name (str): Unique identifier for this tool ("ticket_classifier")
        description (str): Human-readable description of what this tool does
    
    Categories Supported:
        - authentication: Login, password, and access issues
        - database: Database connectivity and query problems
        - billing: Payment and subscription issues
        - other: Miscellaneous or uncategorized tickets
    
    Integration:
        The agent planner uses this tool when it needs to understand the
        nature of a ticket before deciding on next steps (e.g., RAG retrieval).
    """
    
    name = "ticket_classifier"
    description = "Classifies a support ticket into a category (e.g., authentication, database) using ML models."

    def __init__(self):
        """
        Initialize the classification tool.
        
        In Production:
            This would load the trained ML models from disk:
            - Vectorizer (TF-IDF or similar) for text processing
            - Classification model (RandomForest, XGBoost, or Neural Network)
            - Label encoder for category mapping
        
        Current Implementation:
            Uses keyword-based heuristics as a placeholder until the
            full ML pipeline is integrated.
        """
        # ========== Production Implementation (Currently Disabled) ==========
        # from src.models.prediction_pipeline import PredictionPipeline
        # self.pipeline = PredictionPipeline()
        # self.pipeline.load_models()  # Load pre-trained models from disk
        pass

    def run(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a support ticket into a category.
        
        This method analyzes the ticket's subject and description to predict
        the most appropriate category. The classification helps the agent:
        1. Route tickets to the right team/specialist
        2. Retrieve relevant solutions from the knowledge base
        3. Prioritize tickets based on category-specific SLAs
        
        Args:
            ticket_data (Dict[str, Any]): Dictionary containing ticket information
                Required fields:
                    - ticket_id (str): Unique identifier for the ticket
                    - subject (str): Brief summary of the issue
                    - description (str): Detailed problem description
                Optional fields:
                    - priority (str): Urgency level (low, medium, high, critical)
                    - product (str): Which product/service is affected
                    - error_logs (str): Technical error messages
        
        Returns:
            Dict[str, Any]: Classification result containing:
                - predicted_category (str): The predicted category name
                - confidence (float): Confidence score (0.0 to 1.0)
                - reasoning (str): Explanation of why this category was chosen
        
        Example:
            >>> tool = ClassificationTool()
            >>> ticket = {
            ...     "ticket_id": "T-123",
            ...     "subject": "Cannot login to dashboard",
            ...     "description": "Getting 'invalid password' error"
            ... }
            >>> result = tool.run(ticket)
            >>> print(result)
            {
                "predicted_category": "authentication",
                "confidence": 0.92,
                "reasoning": "Detected keywords related to authentication"
            }
        """
        # Log the classification attempt for debugging and monitoring
        logger.info(f"Running ClassificationTool for ticket: {ticket_data.get('ticket_id')}")
        
        # ========== Production Implementation (Currently Disabled) ==========
        # In production, this would use the trained ML pipeline:
        # return self.pipeline.predict(ticket_data)
        
        # ========== Mock Implementation Using Keyword Heuristics ==========
        # Extract and normalize text fields for analysis
        subject = ticket_data.get("subject", "").lower()
        description = ticket_data.get("description", "").lower()
        
        # Default category for tickets that don't match any patterns
        predicted_category = "other"
        confidence = 0.6  # Lower confidence for default category
        
        # ========== Authentication Category Detection ==========
        # Keywords: login, auth, password, access, credentials, 2fa, mfa
        if "login" in subject or "auth" in subject or "password" in description:
            predicted_category = "authentication"
            confidence = 0.92  # High confidence for clear authentication issues
        
        # ========== Database Category Detection ==========
        # Keywords: database, sql, query, connection, timeout, db
        elif "database" in subject or "sql" in description:
            predicted_category = "database"
            confidence = 0.88  # High confidence for database-related keywords
        
        # ========== Billing Category Detection ==========
        # Keywords: payment, card, billing, subscription, invoice, charge
        elif "payment" in subject or "card" in description:
            predicted_category = "billing"
            confidence = 0.85  # High confidence for payment-related issues
        
        # ========== Return Classification Result ==========
        # This result will be stored in the agent's memory and used for
        # decision-making in subsequent steps of the PAOR loop
        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "reasoning": f"Detected keywords related to {predicted_category}"
        }
