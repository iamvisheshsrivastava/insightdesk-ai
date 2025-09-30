# src/monitoring/__init__.py

"""
Monitoring & Drift Detection module for the Intelligent Support System.

This module provides comprehensive monitoring capabilities including:
- Model performance tracking (accuracy, precision, recall, F1)
- Data drift detection for categorical and text features
- MLflow integration for metrics logging
- Real-time monitoring dashboards
- Alert systems for performance degradation and drift

Components:
- performance_monitor: Track model performance metrics
- drift_detector: Detect data distribution changes
- metrics_logger: Log metrics to MLflow and other systems
- monitoring_api: FastAPI endpoints for monitoring data
- alert_manager: Handle alerts for performance and drift issues
"""

from .performance_monitor import (
    ModelPerformanceMonitor,
    PerformanceMetrics,
    ModelMetrics,
    calculate_classification_metrics
)

from .drift_detector import (
    DataDriftDetector,
    DriftMetrics,
    DriftResult,
    FeatureDrift,
    detect_categorical_drift,
    detect_text_drift
)

from .metrics_logger import (
    MetricsLogger,
    LogEntry,
    PerformanceLogEntry,
    DriftLogEntry,
    SystemLogEntry,
    AlertLogEntry,
    NotificationConfig,
    create_metrics_logger,
    get_system_metrics
)

from .alert_manager import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertType
)

__all__ = [
    # Performance monitoring
    "ModelPerformanceMonitor",
    "PerformanceMetrics", 
    "ModelMetrics",
    "calculate_classification_metrics",
    
    # Drift detection
    "DataDriftDetector",
    "DriftMetrics",
    "DriftResult", 
    "FeatureDrift",
    "detect_categorical_drift",
    "detect_text_drift",
    
    # Metrics logging
    "MetricsLogger",
    "MLflowLogger",
    "PrometheusLogger",
    "MetricRecord",
    
    # Alert management
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertType"
]

__version__ = "1.0.0"