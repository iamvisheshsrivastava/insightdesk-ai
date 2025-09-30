# src/feedback/__init__.py

"""
Feedback Loop System for the Intelligent Support System.

This module handles:
- Agent corrections and feedback
- Customer satisfaction feedback
- Feedback data storage and analysis
- Learning from feedback for model improvement
"""

from .feedback_models import (
    AgentCorrection,
    CustomerFeedback,
    FeedbackEvent,
    FeedbackSeverity,
    FeedbackType,
    FeedbackSummary
)

from .feedback_manager import (
    FeedbackManager
)

from .feedback_storage import (
    FeedbackStorage,
    JSONFeedbackStorage,
    SQLiteFeedbackStorage
)

__all__ = [
    # Models
    "AgentCorrection",
    "CustomerFeedback", 
    "FeedbackEvent",
    "FeedbackSeverity",
    "FeedbackType",
    "FeedbackSummary",
    
    # Manager
    "FeedbackManager",
    "FeedbackStats",
    "CorrectionAnalysis",
    
    # Storage
    "FeedbackStorage",
    "JSONFeedbackStorage",
    "SQLiteFeedbackStorage"
]