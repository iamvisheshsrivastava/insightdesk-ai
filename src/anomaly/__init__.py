# src/anomaly/__init__.py

"""
Anomaly detection module for InsightDesk AI.
Detects volume spikes, sentiment shifts, new issues, and outliers in support tickets.
"""

from .anomaly_detector import (
    AnomalyDetector,
    VolumeAnomalyDetector,
    SentimentAnomalyDetector,
    NewIssueDetector,
    OutlierDetector
)

from .anomaly_models import (
    AnomalyType,
    AnomalyRecord,
    AnomalyDetectionResult,
    AnomalyThresholds
)

__all__ = [
    "AnomalyDetector",
    "VolumeAnomalyDetector", 
    "SentimentAnomalyDetector",
    "NewIssueDetector",
    "OutlierDetector",
    "AnomalyType",
    "AnomalyRecord",
    "AnomalyDetectionResult",
    "AnomalyThresholds"
]

__version__ = "1.0.0"
