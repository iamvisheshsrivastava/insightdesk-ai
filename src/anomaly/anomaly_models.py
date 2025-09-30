# src/anomaly/anomaly_models.py

"""
Data models and enums for anomaly detection system.
Defines anomaly types, records, and configuration structures.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import numpy as np


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    VOLUME_SPIKE = "volume_spike"
    SENTIMENT_SHIFT = "sentiment_shift"
    NEW_ISSUE = "new_issue"
    OUTLIER = "outlier"
    PATTERN_CHANGE = "pattern_change"


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyRecord(BaseModel):
    """Individual anomaly detection record."""
    
    id: str = Field(..., description="Unique anomaly identifier")
    type: AnomalyType = Field(..., description="Type of anomaly detected")
    severity: AnomalySeverity = Field(..., description="Severity level of the anomaly")
    
    # Core information
    timestamp: datetime = Field(..., description="When the anomaly was detected")
    category: Optional[str] = Field(None, description="Ticket category associated with anomaly")
    product: Optional[str] = Field(None, description="Product associated with anomaly")
    
    # Anomaly details
    description: str = Field(..., description="Human-readable description of the anomaly")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed anomaly information")
    
    # Metrics
    score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    baseline_value: Optional[float] = Field(None, description="Expected baseline value")
    actual_value: Optional[float] = Field(None, description="Actual observed value")
    threshold: Optional[float] = Field(None, description="Threshold that was exceeded")
    
    # Context
    affected_tickets: List[str] = Field(default_factory=list, description="List of ticket IDs affected")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in anomaly detection")
    
    # Metadata
    detection_method: str = Field(..., description="Method used to detect the anomaly")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When record was created")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnomalyDetectionResult(BaseModel):
    """Result of anomaly detection analysis."""
    
    # Summary
    total_anomalies: int = Field(..., description="Total number of anomalies detected")
    detection_period: Dict[str, datetime] = Field(..., description="Time period analyzed")
    
    # Anomalies by type
    anomalies: List[AnomalyRecord] = Field(default_factory=list, description="List of detected anomalies")
    
    # Statistics
    severity_breakdown: Dict[AnomalySeverity, int] = Field(
        default_factory=dict, 
        description="Count of anomalies by severity"
    )
    type_breakdown: Dict[AnomalyType, int] = Field(
        default_factory=dict,
        description="Count of anomalies by type"
    )
    
    # Performance metrics
    processing_time: float = Field(..., description="Time taken for detection in seconds")
    tickets_analyzed: int = Field(..., description="Number of tickets analyzed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnomalyThresholds(BaseModel):
    """Configuration thresholds for anomaly detection."""
    
    # Volume spike detection
    volume_spike_sigma: float = Field(default=3.0, description="Standard deviations for volume spike detection")
    volume_rolling_window: int = Field(default=7, description="Rolling window in days for volume analysis")
    
    # Sentiment shift detection
    sentiment_shift_threshold: float = Field(default=0.3, description="Threshold for sentiment shift detection")
    sentiment_window_days: int = Field(default=14, description="Window for sentiment baseline calculation")
    
    # New issue detection
    new_issue_similarity_threshold: float = Field(default=0.8, description="Similarity threshold for new issue detection")
    new_issue_min_frequency: int = Field(default=3, description="Minimum frequency for pattern recognition")
    
    # Outlier detection
    outlier_contamination: float = Field(default=0.1, description="Expected proportion of outliers")
    outlier_min_samples: int = Field(default=50, description="Minimum samples needed for outlier detection")
    
    # General settings
    min_tickets_for_analysis: int = Field(default=10, description="Minimum tickets needed for anomaly analysis")
    lookback_days: int = Field(default=30, description="Default lookback period in days")


class VolumeSpike(BaseModel):
    """Specific model for volume spike anomalies."""
    
    category: str = Field(..., description="Category experiencing volume spike")
    date: datetime = Field(..., description="Date of the spike")
    actual_count: int = Field(..., description="Actual ticket count")
    expected_count: float = Field(..., description="Expected ticket count based on historical data")
    spike_ratio: float = Field(..., description="Ratio of actual to expected")
    historical_mean: float = Field(..., description="Historical mean count")
    historical_std: float = Field(..., description="Historical standard deviation")


class SentimentShift(BaseModel):
    """Specific model for sentiment shift anomalies."""
    
    category: str = Field(..., description="Category experiencing sentiment shift")
    current_sentiment: float = Field(..., description="Current average sentiment score")
    baseline_sentiment: float = Field(..., description="Baseline sentiment score")
    shift_magnitude: float = Field(..., description="Magnitude of the shift")
    affected_period: Dict[str, datetime] = Field(..., description="Time period of the shift")
    sample_size: int = Field(..., description="Number of tickets in sample")


class NewIssue(BaseModel):
    """Specific model for new/emerging issue anomalies."""
    
    pattern_description: str = Field(..., description="Description of the new pattern")
    first_occurrence: datetime = Field(..., description="When pattern first appeared")
    frequency: int = Field(..., description="Number of occurrences")
    similarity_to_known: float = Field(..., description="Similarity to known issues (0-1)")
    example_tickets: List[str] = Field(..., description="Example ticket IDs showing this pattern")
    keywords: List[str] = Field(default_factory=list, description="Key terms associated with pattern")


class OutlierInfo(BaseModel):
    """Specific model for outlier anomalies."""
    
    ticket_id: str = Field(..., description="ID of the outlier ticket")
    outlier_score: float = Field(..., description="Outlier score from IsolationForest")
    feature_values: Dict[str, float] = Field(..., description="Feature values that made it an outlier")
    feature_deviations: Dict[str, float] = Field(..., description="How much each feature deviates from normal")


class AnomalyContext(BaseModel):
    """Additional context information for anomalies."""
    
    related_events: List[str] = Field(default_factory=list, description="Related events or incidents")
    business_impact: Optional[str] = Field(None, description="Potential business impact description")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions to take")
    external_factors: List[str] = Field(default_factory=list, description="Potential external factors")


def create_anomaly_record(
    anomaly_type: AnomalyType,
    description: str,
    score: float,
    severity: AnomalySeverity = AnomalySeverity.MEDIUM,
    details: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
    product: Optional[str] = None,
    **kwargs
) -> AnomalyRecord:
    """
    Helper function to create anomaly records with consistent formatting.
    
    Args:
        anomaly_type: Type of anomaly
        description: Human-readable description
        score: Anomaly score
        severity: Severity level
        details: Additional details dictionary
        category: Associated category
        product: Associated product
        **kwargs: Additional fields for AnomalyRecord
        
    Returns:
        AnomalyRecord instance
    """
    import uuid
    
    record_id = f"ANOM-{anomaly_type.value}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
    
    return AnomalyRecord(
        id=record_id,
        type=anomaly_type,
        severity=severity,
        timestamp=datetime.utcnow(),
        category=category,
        product=product,
        description=description,
        details=details or {},
        score=score,
        detection_method=f"{anomaly_type.value}_detector",
        **kwargs
    )


def determine_severity(score: float, thresholds: Optional[Dict[str, float]] = None) -> AnomalySeverity:
    """
    Determine anomaly severity based on score.
    
    Args:
        score: Anomaly score (typically 0-1 or statistical measure)
        thresholds: Custom thresholds for severity levels
        
    Returns:
        AnomalySeverity enum value
    """
    if thresholds is None:
        thresholds = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.0
        }
    
    if score >= thresholds["critical"]:
        return AnomalySeverity.CRITICAL
    elif score >= thresholds["high"]:
        return AnomalySeverity.HIGH
    elif score >= thresholds["medium"]:
        return AnomalySeverity.MEDIUM
    else:
        return AnomalySeverity.LOW


# Utility functions for working with anomaly data

def aggregate_anomalies_by_type(anomalies: List[AnomalyRecord]) -> Dict[AnomalyType, List[AnomalyRecord]]:
    """Group anomalies by type."""
    result = {}
    for anomaly in anomalies:
        if anomaly.type not in result:
            result[anomaly.type] = []
        result[anomaly.type].append(anomaly)
    return result


def filter_anomalies_by_severity(
    anomalies: List[AnomalyRecord], 
    min_severity: AnomalySeverity
) -> List[AnomalyRecord]:
    """Filter anomalies by minimum severity level."""
    severity_order = {
        AnomalySeverity.LOW: 0,
        AnomalySeverity.MEDIUM: 1,
        AnomalySeverity.HIGH: 2,
        AnomalySeverity.CRITICAL: 3
    }
    
    min_level = severity_order[min_severity]
    return [
        anomaly for anomaly in anomalies
        if severity_order[anomaly.severity] >= min_level
    ]


def calculate_anomaly_statistics(anomalies: List[AnomalyRecord]) -> Dict[str, Any]:
    """Calculate summary statistics for a list of anomalies."""
    if not anomalies:
        return {"total": 0, "by_type": {}, "by_severity": {}, "avg_score": 0}
    
    by_type = {}
    by_severity = {}
    scores = []
    
    for anomaly in anomalies:
        # Count by type
        type_key = anomaly.type.value
        by_type[type_key] = by_type.get(type_key, 0) + 1
        
        # Count by severity
        severity_key = anomaly.severity.value
        by_severity[severity_key] = by_severity.get(severity_key, 0) + 1
        
        # Collect scores
        scores.append(anomaly.score)
    
    return {
        "total": len(anomalies),
        "by_type": by_type,
        "by_severity": by_severity,
        "avg_score": np.mean(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0
    }