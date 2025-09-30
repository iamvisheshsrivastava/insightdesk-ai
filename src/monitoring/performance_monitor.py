# src/monitoring/performance_monitor.py

"""
Model Performance Monitoring for the Intelligent Support System.

This module tracks and analyzes model performance metrics including accuracy,
precision, recall, F1-score, and other classification metrics. It integrates
with MLflow for metric logging and provides real-time performance tracking.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import warnings

# ML metrics imports with fallbacks
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score,
        precision_recall_curve, roc_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Performance monitoring will be limited.")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Metrics logging will be limited.")

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Additional metrics
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_precision: Optional[float] = None
    weighted_recall: Optional[float] = None
    weighted_f1: Optional[float] = None
    
    # Class-specific metrics
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Confidence and prediction quality
    mean_prediction_confidence: Optional[float] = None
    prediction_uncertainty: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
            "timestamp": self.timestamp.isoformat()
        }
        
        # Add optional metrics if available
        optional_fields = [
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
            "mean_prediction_confidence", "prediction_uncertainty"
        ]
        
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        if self.per_class_metrics:
            result["per_class_metrics"] = self.per_class_metrics
            
        if self.confusion_matrix:
            result["confusion_matrix"] = self.confusion_matrix
        
        return result


@dataclass
class ModelMetrics:
    """Container for all metrics related to a specific model."""
    
    model_name: str
    model_version: Optional[str] = None
    recent_metrics: Optional[PerformanceMetrics] = None
    historical_metrics: List[PerformanceMetrics] = field(default_factory=list)
    
    # Performance trends
    accuracy_trend: List[float] = field(default_factory=list)
    f1_trend: List[float] = field(default_factory=list)
    
    # Inference statistics
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    # Timing metrics
    avg_inference_time: Optional[float] = None
    p95_inference_time: Optional[float] = None
    inference_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add new performance metrics."""
        self.recent_metrics = metrics
        self.historical_metrics.append(metrics)
        
        # Update trends (keep last 100 points)
        self.accuracy_trend.append(metrics.accuracy)
        self.f1_trend.append(metrics.f1_score)
        
        if len(self.accuracy_trend) > 100:
            self.accuracy_trend.pop(0)
        if len(self.f1_trend) > 100:
            self.f1_trend.pop(0)
    
    def add_inference_time(self, time_ms: float):
        """Add inference time measurement."""
        self.inference_times.append(time_ms)
        
        # Update statistics
        if self.inference_times:
            self.avg_inference_time = np.mean(self.inference_times)
            self.p95_inference_time = np.percentile(self.inference_times, 95)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        summary = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "success_rate": self.successful_predictions / max(self.total_predictions, 1)
        }
        
        if self.recent_metrics:
            summary.update({
                "latest_accuracy": self.recent_metrics.accuracy,
                "latest_f1": self.recent_metrics.f1_score,
                "latest_precision": self.recent_metrics.precision,
                "latest_recall": self.recent_metrics.recall,
                "metrics_timestamp": self.recent_metrics.timestamp.isoformat()
            })
        
        if self.avg_inference_time:
            summary.update({
                "avg_inference_time_ms": self.avg_inference_time,
                "p95_inference_time_ms": self.p95_inference_time
            })
        
        # Add trend information
        if len(self.accuracy_trend) > 1:
            summary["accuracy_trend"] = {
                "current": self.accuracy_trend[-1],
                "previous": self.accuracy_trend[-2],
                "change": self.accuracy_trend[-1] - self.accuracy_trend[-2]
            }
        
        return summary


class ModelPerformanceMonitor:
    """Monitor and track model performance metrics."""
    
    def __init__(self, 
                 window_size: int = 1000,
                 alert_threshold: float = 0.05,
                 mlflow_tracking: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent predictions to track
            alert_threshold: Performance drop threshold for alerts
            mlflow_tracking: Whether to log metrics to MLflow
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.mlflow_tracking = mlflow_tracking
        
        # Model metrics storage
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Prediction storage for batch evaluation
        self.prediction_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.true_labels_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.confidence_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Alerting
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info(f"Performance monitor initialized with window_size={window_size}")
    
    def register_model(self, model_name: str, model_version: Optional[str] = None):
        """Register a model for monitoring."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(
                model_name=model_name,
                model_version=model_version
            )
            logger.info(f"Registered model for monitoring: {model_name}")
    
    def log_prediction(self, 
                      model_name: str,
                      prediction: Union[str, int],
                      true_label: Optional[Union[str, int]] = None,
                      confidence: Optional[float] = None,
                      inference_time_ms: Optional[float] = None):
        """
        Log a single prediction for performance tracking.
        
        Args:
            model_name: Name of the model
            prediction: Model prediction
            true_label: Ground truth label (if available)
            confidence: Prediction confidence score
            inference_time_ms: Inference time in milliseconds
        """
        # Ensure model is registered
        self.register_model(model_name)
        
        # Update prediction counts
        self.model_metrics[model_name].total_predictions += 1
        
        if prediction is not None:
            self.model_metrics[model_name].successful_predictions += 1
            
            # Store prediction for batch evaluation
            self.prediction_buffer[model_name].append(prediction)
            
            if true_label is not None:
                self.true_labels_buffer[model_name].append(true_label)
            
            if confidence is not None:
                self.confidence_buffer[model_name].append(confidence)
        else:
            self.model_metrics[model_name].failed_predictions += 1
        
        # Log inference time
        if inference_time_ms is not None:
            self.model_metrics[model_name].add_inference_time(inference_time_ms)
        
        # Check if we should evaluate performance
        if (len(self.prediction_buffer[model_name]) >= 100 and 
            len(self.true_labels_buffer[model_name]) >= 100):
            self._evaluate_model_performance(model_name)
    
    def log_batch_predictions(self,
                             model_name: str,
                             predictions: List[Union[str, int]],
                             true_labels: List[Union[str, int]],
                             confidences: Optional[List[float]] = None,
                             inference_times_ms: Optional[List[float]] = None):
        """
        Log a batch of predictions for performance evaluation.
        
        Args:
            model_name: Name of the model
            predictions: List of model predictions
            true_labels: List of ground truth labels
            confidences: List of prediction confidence scores
            inference_times_ms: List of inference times in milliseconds
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and true_labels must have same length")
        
        # Ensure model is registered
        self.register_model(model_name)
        
        # Update prediction counts
        self.model_metrics[model_name].total_predictions += len(predictions)
        self.model_metrics[model_name].successful_predictions += len(predictions)
        
        # Extend buffers
        self.prediction_buffer[model_name].extend(predictions)
        self.true_labels_buffer[model_name].extend(true_labels)
        
        if confidences:
            self.confidence_buffer[model_name].extend(confidences)
        
        if inference_times_ms:
            for time_ms in inference_times_ms:
                self.model_metrics[model_name].add_inference_time(time_ms)
        
        # Evaluate performance immediately for batch
        self._evaluate_model_performance(model_name)
        
        logger.info(f"Logged batch of {len(predictions)} predictions for {model_name}")
    
    def _evaluate_model_performance(self, model_name: str):
        """Evaluate model performance using buffered predictions."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Skipping performance evaluation.")
            return
        
        predictions = list(self.prediction_buffer[model_name])
        true_labels = list(self.true_labels_buffer[model_name])
        
        if len(predictions) < 10 or len(true_labels) < 10:
            return
        
        # Ensure same length
        min_length = min(len(predictions), len(true_labels))
        predictions = predictions[-min_length:]
        true_labels = true_labels[-min_length:]
        
        try:
            # Calculate metrics
            metrics = calculate_classification_metrics(
                true_labels, predictions, 
                list(self.confidence_buffer[model_name])[-min_length:] if self.confidence_buffer[model_name] else None
            )
            
            # Store metrics
            self.model_metrics[model_name].add_metrics(metrics)
            
            # Check for performance degradation
            self._check_performance_alerts(model_name, metrics)
            
            # Log to MLflow if enabled
            if self.mlflow_tracking and MLFLOW_AVAILABLE:
                self._log_to_mlflow(model_name, metrics)
            
            logger.info(f"Evaluated performance for {model_name}: "
                       f"Accuracy={metrics.accuracy:.3f}, F1={metrics.f1_score:.3f}")
        
        except Exception as e:
            logger.error(f"Error evaluating performance for {model_name}: {e}")
    
    def _check_performance_alerts(self, model_name: str, metrics: PerformanceMetrics):
        """Check if performance has degraded and create alerts."""
        model_metrics = self.model_metrics[model_name]
        
        if len(model_metrics.accuracy_trend) >= 2:
            current_accuracy = metrics.accuracy
            previous_accuracy = model_metrics.accuracy_trend[-2]
            
            accuracy_drop = previous_accuracy - current_accuracy
            
            if accuracy_drop > self.alert_threshold:
                alert = {
                    "type": "performance_degradation",
                    "model": model_name,
                    "metric": "accuracy",
                    "current_value": current_accuracy,
                    "previous_value": previous_accuracy,
                    "drop": accuracy_drop,
                    "threshold": self.alert_threshold,
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "high" if accuracy_drop > self.alert_threshold * 2 else "medium"
                }
                
                self.alerts.append(alert)
                logger.warning(f"Performance alert: {model_name} accuracy dropped by {accuracy_drop:.3f}")
    
    def _log_to_mlflow(self, model_name: str, metrics: PerformanceMetrics):
        """Log metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_name}_performance_monitoring", nested=True):
                mlflow.log_metrics({
                    f"{model_name}_accuracy": metrics.accuracy,
                    f"{model_name}_precision": metrics.precision,
                    f"{model_name}_recall": metrics.recall,
                    f"{model_name}_f1_score": metrics.f1_score,
                    f"{model_name}_support": metrics.support
                })
                
                # Log additional metrics if available
                if metrics.macro_f1:
                    mlflow.log_metric(f"{model_name}_macro_f1", metrics.macro_f1)
                if metrics.weighted_f1:
                    mlflow.log_metric(f"{model_name}_weighted_f1", metrics.weighted_f1)
                if metrics.mean_prediction_confidence:
                    mlflow.log_metric(f"{model_name}_mean_confidence", metrics.mean_prediction_confidence)
                
                # Log confusion matrix if available
                if metrics.confusion_matrix:
                    mlflow.log_dict(
                        {"confusion_matrix": metrics.confusion_matrix},
                        f"{model_name}_confusion_matrix.json"
                    )
        
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
    
    def get_model_status(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get current status and metrics for a model."""
        if model_name not in self.model_metrics:
            return None
        
        return self.model_metrics[model_name].get_performance_summary()
    
    def get_all_models_status(self) -> Dict[str, Any]:
        """Get status for all monitored models."""
        return {
            "models": {
                name: metrics.get_performance_summary()
                for name, metrics in self.model_metrics.items()
            },
            "total_models": len(self.model_metrics),
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }
    
    def get_performance_trends(self, model_name: str, days: int = 7) -> Optional[Dict[str, Any]]:
        """Get performance trends for a model over specified days."""
        if model_name not in self.model_metrics:
            return None
        
        model_metrics = self.model_metrics[model_name]
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter historical metrics by date
        recent_metrics = [
            m for m in model_metrics.historical_metrics
            if m.timestamp >= cutoff_date
        ]
        
        if not recent_metrics:
            return None
        
        # Calculate trends
        timestamps = [m.timestamp.isoformat() for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics]
        f1_scores = [m.f1_score for m in recent_metrics]
        precisions = [m.precision for m in recent_metrics]
        recalls = [m.recall for m in recent_metrics]
        
        return {
            "model_name": model_name,
            "period_days": days,
            "data_points": len(recent_metrics),
            "trends": {
                "timestamps": timestamps,
                "accuracy": accuracies,
                "f1_score": f1_scores,
                "precision": precisions,
                "recall": recalls
            },
            "summary": {
                "avg_accuracy": np.mean(accuracies),
                "avg_f1": np.mean(f1_scores),
                "accuracy_std": np.std(accuracies),
                "f1_std": np.std(f1_scores),
                "trend_slope_accuracy": self._calculate_trend_slope(accuracies),
                "trend_slope_f1": self._calculate_trend_slope(f1_scores)
            }
        }
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def reset_model_metrics(self, model_name: str):
        """Reset metrics for a specific model."""
        if model_name in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(
                model_name=model_name,
                model_version=self.model_metrics[model_name].model_version
            )
            self.prediction_buffer[model_name].clear()
            self.true_labels_buffer[model_name].clear()
            self.confidence_buffer[model_name].clear()
            logger.info(f"Reset metrics for model: {model_name}")


def calculate_classification_metrics(
    true_labels: List[Union[str, int]],
    predictions: List[Union[str, int]],
    confidences: Optional[List[float]] = None
) -> PerformanceMetrics:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        confidences: Prediction confidence scores
        
    Returns:
        PerformanceMetrics object with calculated metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for metric calculation")
    
    if len(true_labels) != len(predictions):
        raise ValueError("true_labels and predictions must have same length")
    
    # Convert to numpy arrays
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate macro and weighted averages
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class_metrics = {
            str(label): {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": metrics["support"]
            }
            for label, metrics in report.items()
            if isinstance(metrics, dict) and "precision" in metrics
        }
    except Exception as e:
        logger.warning(f"Could not calculate per-class metrics: {e}")
        per_class_metrics = None
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrix_list = cm.tolist()
    except Exception as e:
        logger.warning(f"Could not calculate confusion matrix: {e}")
        confusion_matrix_list = None
    
    # Confidence metrics
    mean_confidence = None
    prediction_uncertainty = None
    
    if confidences:
        conf_array = np.array(confidences)
        mean_confidence = float(np.mean(conf_array))
        prediction_uncertainty = float(np.std(conf_array))
    
    return PerformanceMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        support=len(true_labels),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        per_class_metrics=per_class_metrics,
        confusion_matrix=confusion_matrix_list,
        mean_prediction_confidence=mean_confidence,
        prediction_uncertainty=prediction_uncertainty
    )


# Utility functions for performance analysis

def analyze_prediction_patterns(predictions: List[str], 
                               true_labels: List[str]) -> Dict[str, Any]:
    """Analyze patterns in model predictions vs true labels."""
    if not predictions or not true_labels:
        return {}
    
    # Convert to pandas for easier analysis
    df = pd.DataFrame({
        'prediction': predictions,
        'true_label': true_labels
    })
    
    # Calculate accuracy by class
    class_accuracy = {}
    for label in df['true_label'].unique():
        class_data = df[df['true_label'] == label]
        correct = (class_data['prediction'] == class_data['true_label']).sum()
        total = len(class_data)
        class_accuracy[label] = correct / total if total > 0 else 0
    
    # Find most confused pairs
    confusion_pairs = df[df['prediction'] != df['true_label']].groupby(
        ['true_label', 'prediction']
    ).size().sort_values(ascending=False).head(5)
    
    return {
        "class_accuracy": class_accuracy,
        "most_confused_pairs": confusion_pairs.to_dict(),
        "overall_accuracy": (df['prediction'] == df['true_label']).mean(),
        "total_samples": len(df)
    }


def calculate_model_stability(metrics_history: List[PerformanceMetrics]) -> Dict[str, float]:
    """Calculate stability metrics for a model over time."""
    if len(metrics_history) < 2:
        return {"stability_score": 1.0, "variance": 0.0}
    
    accuracies = [m.accuracy for m in metrics_history]
    f1_scores = [m.f1_score for m in metrics_history]
    
    accuracy_variance = np.var(accuracies)
    f1_variance = np.var(f1_scores)
    
    # Stability score (higher is more stable)
    stability_score = 1.0 / (1.0 + accuracy_variance + f1_variance)
    
    return {
        "stability_score": stability_score,
        "accuracy_variance": accuracy_variance,
        "f1_variance": f1_variance,
        "accuracy_trend": np.polyfit(range(len(accuracies)), accuracies, 1)[0],
        "f1_trend": np.polyfit(range(len(f1_scores)), f1_scores, 1)[0]
    }