# src/feedback/feedback_models.py

"""
Data models for the Feedback Loop system.

This module defines the core data structures for capturing and managing
agent corrections and customer feedback.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback events."""
    AGENT_CORRECTION = "agent_correction"
    CUSTOMER_FEEDBACK = "customer_feedback"
    SYSTEM_FEEDBACK = "system_feedback"
    SYSTEM_EVENT = "system_event"


class FeedbackSeverity(Enum):
    """Severity levels for feedback events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionQuality(Enum):
    """Quality assessment for predictions."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INCORRECT = "incorrect"


@dataclass
class AgentCorrection:
    """Agent correction for model predictions."""
    
    # Identifiers
    correction_id: str
    ticket_id: str
    agent_id: str
    
    # Original prediction details
    original_prediction: str
    original_confidence: float
    model_type: str
    
    # Correction details
    corrected_label: str
    correction_reason: str
    correction_notes: Optional[str] = None
    
    # Quality assessment
    prediction_quality: PredictionQuality = PredictionQuality.POOR
    severity: FeedbackSeverity = FeedbackSeverity.MEDIUM
    
    # Metadata
    correction_timestamp: datetime = field(default_factory=datetime.utcnow)
    ticket_data: Optional[Dict[str, Any]] = None
    
    # Learning flags
    should_retrain: bool = True
    correction_confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result["correction_timestamp"] = self.correction_timestamp.isoformat()
        result["prediction_quality"] = self.prediction_quality.value
        result["severity"] = self.severity.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCorrection":
        """Create from dictionary."""
        # Handle enum fields
        if "prediction_quality" in data:
            data["prediction_quality"] = PredictionQuality(data["prediction_quality"])
        if "severity" in data:
            data["severity"] = FeedbackSeverity(data["severity"])
        
        # Handle datetime field
        if "correction_timestamp" in data and isinstance(data["correction_timestamp"], str):
            data["correction_timestamp"] = datetime.fromisoformat(data["correction_timestamp"])
        
        return cls(**data)


@dataclass
class CustomerFeedback:
    """Customer feedback on support resolution."""
    
    # Identifiers (required)
    feedback_id: str
    ticket_id: str
    rating: int  # 1-5 scale
    feedback_timestamp: datetime
    
    # Optional identifiers
    customer_id: Optional[str] = None
    
    # Feedback details
    comments: Optional[str] = None
    feedback_type: str = "resolution_quality"
    
    # Resolution evaluation
    resolution_helpful: bool = True
    resolution_accurate: bool = True
    would_recommend: Optional[bool] = None
    
    # AI assistance evaluation
    ai_suggestions_used: bool = False
    ai_suggestions_helpful: Optional[bool] = None
    ai_accuracy_rating: Optional[int] = None  # 1-5 scale
    
    # Metadata
    resolution_time_hours: Optional[float] = None
    agent_id: Optional[str] = None
    
    # Categorization (computed after init)
    satisfaction_level: str = field(init=False, default="")
    needs_followup: bool = field(init=False, default=False)
    
    def __post_init__(self):
        """Set derived fields after initialization."""
        # Determine satisfaction level
        if self.rating >= 4:
            self.satisfaction_level = "satisfied"
        elif self.rating == 3:
            self.satisfaction_level = "neutral"
        else:
            self.satisfaction_level = "unsatisfied"
        
        # Determine if followup is needed
        self.needs_followup = (
            self.rating <= 2 or 
            not self.resolution_helpful or 
            not self.resolution_accurate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result["feedback_timestamp"] = self.feedback_timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomerFeedback":
        """Create from dictionary."""
        # Handle datetime field
        if "feedback_timestamp" in data and isinstance(data["feedback_timestamp"], str):
            data["feedback_timestamp"] = datetime.fromisoformat(data["feedback_timestamp"])
        
        # Remove computed fields that can't be passed to constructor
        computed_fields = {"satisfaction_level", "needs_followup"}
        data_copy = {k: v for k, v in data.items() if k not in computed_fields}
        
        return cls(**data_copy)


@dataclass
class FeedbackEvent:
    """Generic feedback event for logging."""
    
    # Core identification (required)
    event_id: str
    event_type: FeedbackType
    ticket_id: str
    event_data: Dict[str, Any]
    timestamp: datetime
    
    # Optional fields
    severity: FeedbackSeverity = FeedbackSeverity.MEDIUM
    
    # Processing status
    processed: bool = False
    processing_notes: Optional[str] = None
    
    # Associated objects
    correction: Optional[AgentCorrection] = None
    customer_feedback: Optional[CustomerFeedback] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "ticket_id": self.ticket_id,
            "event_data": self.event_data,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "processing_notes": self.processing_notes
        }
        
        if self.correction:
            result["correction"] = self.correction.to_dict()
        
        if self.customer_feedback:
            result["customer_feedback"] = self.customer_feedback.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEvent":
        """Create from dictionary."""
        # Handle enum fields
        if "event_type" in data:
            data["event_type"] = FeedbackType(data["event_type"])
        if "severity" in data:
            data["severity"] = FeedbackSeverity(data["severity"])
        
        # Handle datetime field
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Handle nested objects
        if "correction" in data and data["correction"]:
            data["correction"] = AgentCorrection.from_dict(data["correction"])
        
        if "customer_feedback" in data and data["customer_feedback"]:
            data["customer_feedback"] = CustomerFeedback.from_dict(data["customer_feedback"])
        
        return cls(**data)


@dataclass
class FeedbackSummary:
    """Summary statistics for feedback analysis."""
    
    # Identifiers and time period
    summary_id: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    
    # Agent corrections
    total_corrections: int = 0
    corrections_by_model: Dict[str, int] = field(default_factory=dict)
    corrections_by_severity: Dict[str, int] = field(default_factory=dict)
    avg_original_confidence: float = 0.0
    
    # Customer feedback
    total_customer_feedback: int = 0
    avg_customer_rating: float = 0.0
    satisfaction_breakdown: Dict[str, int] = field(default_factory=dict)
    ai_usage_rate: float = 0.0
    
    # Events
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Retraining recommendations
    retraining_recommendations: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result["period_start"] = self.period_start.isoformat()
        result["period_end"] = self.period_end.isoformat()
        result["generated_at"] = self.generated_at.isoformat()
        return result


# Pydantic models for API requests/responses

class AgentCorrectionRequest(BaseModel):
    """Request model for agent corrections."""
    ticket_id: str = Field(..., description="Ticket ID being corrected")
    agent_id: str = Field(..., description="ID of the correcting agent")
    original_prediction: str = Field(..., description="Original AI prediction")
    original_confidence: float = Field(..., ge=0.0, le=1.0, description="Original prediction confidence")
    model_type: str = Field(..., description="Model that made the prediction")
    corrected_label: str = Field(..., description="Correct label according to agent")
    correction_reason: str = Field(..., description="Reason for the correction")
    correction_notes: Optional[str] = Field(None, description="Additional correction notes")
    prediction_quality: str = Field("poor", description="Quality assessment of original prediction")
    severity: str = Field("medium", description="Severity of the correction")
    ticket_data: Optional[Dict[str, Any]] = Field(None, description="Original ticket data")
    
    @validator('prediction_quality')
    def validate_quality(cls, v):
        valid_qualities = [q.value for q in PredictionQuality]
        if v not in valid_qualities:
            raise ValueError(f"Invalid prediction quality. Must be one of: {valid_qualities}")
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = [s.value for s in FeedbackSeverity]
        if v not in valid_severities:
            raise ValueError(f"Invalid severity. Must be one of: {valid_severities}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "ticket_id": "TK-2025-001",
                "agent_id": "agent_001",
                "original_prediction": "bug",
                "original_confidence": 0.85,
                "model_type": "xgboost",
                "corrected_label": "feature_request",
                "correction_reason": "Customer is asking for new functionality, not reporting a bug",
                "correction_notes": "Need to improve feature detection in text analysis",
                "prediction_quality": "poor",
                "severity": "medium"
            }
        }


class CustomerFeedbackRequest(BaseModel):
    """Request model for customer feedback."""
    ticket_id: str = Field(..., description="Ticket ID being rated")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    rating: int = Field(..., ge=1, le=5, description="Overall satisfaction rating (1-5)")
    comments: Optional[str] = Field(None, description="Additional feedback comments")
    feedback_type: str = Field("resolution_quality", description="Type of feedback")
    resolution_helpful: bool = Field(True, description="Was the resolution helpful?")
    resolution_accurate: bool = Field(True, description="Was the resolution accurate?")
    would_recommend: Optional[bool] = Field(None, description="Would recommend this support?")
    ai_suggestions_used: bool = Field(False, description="Were AI suggestions used in resolution?")
    ai_suggestions_helpful: Optional[bool] = Field(None, description="Were AI suggestions helpful?")
    ai_accuracy_rating: Optional[int] = Field(None, ge=1, le=5, description="AI accuracy rating (1-5)")
    resolution_time_hours: Optional[float] = Field(None, ge=0.0, description="Time to resolution in hours")
    agent_id: Optional[str] = Field(None, description="Agent who handled the ticket")
    
    class Config:
        schema_extra = {
            "example": {
                "ticket_id": "TK-2025-001",
                "customer_id": "customer_123",
                "rating": 4,
                "comments": "Resolution was helpful but took longer than expected",
                "resolution_helpful": True,
                "resolution_accurate": True,
                "would_recommend": True,
                "ai_suggestions_used": True,
                "ai_suggestions_helpful": True,
                "ai_accuracy_rating": 4,
                "resolution_time_hours": 2.5,
                "agent_id": "agent_001"
            }
        }


class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics."""
    summary_id: str
    period_start: datetime
    period_end: datetime
    total_corrections: int
    corrections_by_model: Dict[str, int]
    corrections_by_severity: Dict[str, int]
    avg_original_confidence: float
    total_customer_feedback: int
    avg_customer_rating: float
    satisfaction_breakdown: Dict[str, int]
    ai_usage_rate: float
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    quality_metrics: Dict[str, float]
    retraining_recommendations: Dict[str, Any]


# Utility functions

def generate_feedback_id(prefix: str = "FB") -> str:
    """Generate unique feedback ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"


def calculate_prediction_accuracy(corrections: List[AgentCorrection]) -> float:
    """Calculate prediction accuracy from corrections."""
    if not corrections:
        return 1.0
    
    # Accuracy = 1 - (corrections / total_predictions)
    # This is simplified - in reality you'd need total predictions count
    total_corrections = len(corrections)
    
    # Estimate total predictions (this would come from actual system metrics)
    estimated_total = total_corrections * 5  # Assume 1 correction per 5 predictions
    
    accuracy = max(0.0, 1.0 - (total_corrections / estimated_total))
    return round(accuracy, 3)


def categorize_correction_severity(
    original_confidence: float,
    prediction_quality: PredictionQuality
) -> FeedbackSeverity:
    """Determine correction severity based on confidence and quality."""
    if prediction_quality == PredictionQuality.INCORRECT:
        if original_confidence > 0.8:
            return FeedbackSeverity.CRITICAL
        elif original_confidence > 0.6:
            return FeedbackSeverity.HIGH
        else:
            return FeedbackSeverity.MEDIUM
    elif prediction_quality == PredictionQuality.POOR:
        return FeedbackSeverity.MEDIUM
    else:
        return FeedbackSeverity.LOW


def calculate_accuracy(predictions: List[str], corrected_labels: List[str]) -> float:
    """
    Calculate accuracy between predictions and corrected labels.
    
    Args:
        predictions: List of original predictions
        corrected_labels: List of corrected labels
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if not predictions or not corrected_labels or len(predictions) != len(corrected_labels):
        return 0.0
    
    correct_count = sum(1 for pred, correct in zip(predictions, corrected_labels) if pred == correct)
    return correct_count / len(predictions)