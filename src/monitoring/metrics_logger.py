# src/monitoring/metrics_logger.py

"""
Metrics Logging for the Intelligent Support System.

This module handles centralized logging of performance metrics, drift detection results,
and system health metrics to various destinations including MLflow, file system, and databases.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque
import time
import warnings

# MLflow imports with fallback
try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. MLflow logging will be disabled.")

# Database imports with fallback
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logging.warning("SQLite not available. Database logging will be disabled.")

from .performance_monitor import PerformanceMetrics, ModelMetrics
from .drift_detector import DriftResult, FeatureDrift

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Base class for log entries."""
    
    timestamp: datetime
    log_type: str
    model_name: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "log_type": self.log_type,
            "model_name": self.model_name,
            "data": self.data
        }


@dataclass
class PerformanceLogEntry(LogEntry):
    """Log entry for performance metrics."""
    
    def __init__(self, model_name: str, metrics: PerformanceMetrics):
        super().__init__(
            timestamp=datetime.utcnow(),
            log_type="performance",
            model_name=model_name,
            data=asdict(metrics)
        )


@dataclass
class DriftLogEntry(LogEntry):
    """Log entry for drift detection results."""
    
    def __init__(self, model_name: str, drift_result: DriftResult):
        super().__init__(
            timestamp=datetime.utcnow(),
            log_type="drift",
            model_name=model_name,
            data=drift_result.to_dict()
        )


@dataclass
class SystemLogEntry(LogEntry):
    """Log entry for system health metrics."""
    
    def __init__(self, 
                 model_name: str, 
                 cpu_usage: float,
                 memory_usage: float,
                 prediction_latency: float,
                 throughput: float,
                 error_count: int = 0):
        system_data = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "prediction_latency": prediction_latency,
            "throughput": throughput,
            "error_count": error_count
        }
        
        super().__init__(
            timestamp=datetime.utcnow(),
            log_type="system",
            model_name=model_name,
            data=system_data
        )


@dataclass
class AlertLogEntry(LogEntry):
    """Log entry for alerts."""
    
    def __init__(self, 
                 model_name: str,
                 alert_type: str,
                 severity: str,
                 message: str,
                 metrics: Dict[str, Any] = None):
        alert_data = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metrics": metrics or {}
        }
        
        super().__init__(
            timestamp=datetime.utcnow(),
            log_type="alert",
            model_name=model_name,
            data=alert_data
        )


class MetricsLogger:
    """Centralized metrics logging system."""
    
    def __init__(self,
                 log_dir: str = "logs",
                 mlflow_tracking_uri: str = None,
                 mlflow_experiment_name: str = "model_monitoring",
                 db_path: str = None,
                 max_log_entries: int = 10000,
                 auto_flush_interval: int = 60):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            mlflow_tracking_uri: MLflow tracking server URI
            mlflow_experiment_name: MLflow experiment name
            db_path: SQLite database path for metrics storage
            max_log_entries: Maximum log entries to keep in memory
            auto_flush_interval: Auto-flush interval in seconds
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.db_path = db_path
        self.max_log_entries = max_log_entries
        self.auto_flush_interval = auto_flush_interval
        
        # In-memory log storage
        self.log_entries: deque = deque(maxlen=max_log_entries)
        self.model_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Threading for async operations
        self._lock = threading.Lock()
        self._stop_background = False
        self._background_thread = None
        
        # Initialize backends
        self._init_mlflow()
        self._init_database()
        
        # Start background thread
        self._start_background_thread()
        
        logger.info(f"Metrics logger initialized: {log_dir}")
    
    def _init_mlflow(self):
        """Initialize MLflow backend."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Set or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
                    logger.info(f"Created MLflow experiment: {self.mlflow_experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {self.mlflow_experiment_name}")
                
                mlflow.set_experiment(self.mlflow_experiment_name)
                
            except Exception as e:
                logger.warning(f"Could not set MLflow experiment: {e}")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
    
    def _init_database(self):
        """Initialize SQLite database backend."""
        if not SQLITE_AVAILABLE or not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    log_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics_log(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_name ON metrics_log(model_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_type ON metrics_log(log_type)
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    
    def _start_background_thread(self):
        """Start background thread for periodic operations."""
        self._background_thread = threading.Thread(
            target=self._background_worker,
            daemon=True
        )
        self._background_thread.start()
    
    def _background_worker(self):
        """Background worker for periodic tasks."""
        while not self._stop_background:
            try:
                time.sleep(self.auto_flush_interval)
                self._flush_logs()
            except Exception as e:
                logger.error(f"Background worker error: {e}")
    
    def log_performance_metrics(self, 
                               model_name: str, 
                               metrics: PerformanceMetrics,
                               run_id: str = None):
        """
        Log performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            run_id: Optional MLflow run ID
        """
        entry = PerformanceLogEntry(model_name, metrics)
        
        with self._lock:
            self.log_entries.append(entry)
            self.model_metrics[model_name]["performance"].append(entry)
        
        # Log to file
        self._log_to_file(entry, "performance")
        
        # Log to MLflow
        self._log_to_mlflow(entry, run_id)
        
        # Log to database
        self._log_to_database(entry)
        
        logger.debug(f"Logged performance metrics for {model_name}")
    
    def log_drift_metrics(self, 
                         model_name: str, 
                         drift_result: DriftResult,
                         run_id: str = None):
        """
        Log drift detection results.
        
        Args:
            model_name: Name of the model
            drift_result: Drift detection results
            run_id: Optional MLflow run ID
        """
        entry = DriftLogEntry(model_name, drift_result)
        
        with self._lock:
            self.log_entries.append(entry)
            self.model_metrics[model_name]["drift"].append(entry)
        
        # Log to file
        self._log_to_file(entry, "drift")
        
        # Log to MLflow
        self._log_to_mlflow(entry, run_id)
        
        # Log to database
        self._log_to_database(entry)
        
        logger.debug(f"Logged drift metrics for {model_name}")
    
    def log_system_metrics(self,
                          model_name: str,
                          cpu_usage: float,
                          memory_usage: float,
                          prediction_latency: float,
                          throughput: float,
                          error_count: int = 0,
                          run_id: str = None):
        """
        Log system health metrics.
        
        Args:
            model_name: Name of the model
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            prediction_latency: Average prediction latency (ms)
            throughput: Predictions per second
            error_count: Number of errors
            run_id: Optional MLflow run ID
        """
        entry = SystemLogEntry(
            model_name, cpu_usage, memory_usage, 
            prediction_latency, throughput, error_count
        )
        
        with self._lock:
            self.log_entries.append(entry)
            self.model_metrics[model_name]["system"].append(entry)
        
        # Log to file
        self._log_to_file(entry, "system")
        
        # Log to MLflow
        self._log_to_mlflow(entry, run_id)
        
        # Log to database
        self._log_to_database(entry)
        
        logger.debug(f"Logged system metrics for {model_name}")
    
    def log_alert(self,
                 model_name: str,
                 alert_type: str,
                 severity: str,
                 message: str,
                 metrics: Dict[str, Any] = None,
                 run_id: str = None):
        """
        Log alert.
        
        Args:
            model_name: Name of the model
            alert_type: Type of alert (performance, drift, system)
            severity: Alert severity (low, medium, high, critical)
            message: Alert message
            metrics: Additional metrics
            run_id: Optional MLflow run ID
        """
        entry = AlertLogEntry(model_name, alert_type, severity, message, metrics)
        
        with self._lock:
            self.log_entries.append(entry)
            self.model_metrics[model_name]["alerts"].append(entry)
        
        # Log to file
        self._log_to_file(entry, "alerts")
        
        # Log to MLflow
        self._log_to_mlflow(entry, run_id)
        
        # Log to database
        self._log_to_database(entry)
        
        logger.info(f"Alert logged for {model_name}: {severity} - {message}")
    
    def _log_to_file(self, entry: LogEntry, log_type: str):
        """Log entry to file."""
        try:
            log_file = self.log_dir / f"{log_type}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to log to file: {e}")
    
    def _log_to_mlflow(self, entry: LogEntry, run_id: str = None):
        """Log entry to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            with mlflow.start_run(run_id=run_id) as run:
                # Log metrics based on type
                if entry.log_type == "performance":
                    data = entry.data
                    mlflow.log_metrics({
                        f"{entry.model_name}_accuracy": data.get("accuracy", 0),
                        f"{entry.model_name}_precision": data.get("precision", 0),
                        f"{entry.model_name}_recall": data.get("recall", 0),
                        f"{entry.model_name}_f1_score": data.get("f1_score", 0)
                    })
                
                elif entry.log_type == "drift":
                    data = entry.data
                    mlflow.log_metrics({
                        f"{entry.model_name}_drift_score": data.get("drift_score", 0),
                        f"{entry.model_name}_drifted_features": data.get("num_drifted_features", 0)
                    })
                
                elif entry.log_type == "system":
                    data = entry.data
                    mlflow.log_metrics({
                        f"{entry.model_name}_cpu_usage": data.get("cpu_usage", 0),
                        f"{entry.model_name}_memory_usage": data.get("memory_usage", 0),
                        f"{entry.model_name}_prediction_latency": data.get("prediction_latency", 0),
                        f"{entry.model_name}_throughput": data.get("throughput", 0)
                    })
                
                # Log as artifact
                mlflow.log_dict(entry.to_dict(), f"metrics/{entry.log_type}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}.json")
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
    
    def _log_to_database(self, entry: LogEntry):
        """Log entry to database."""
        if not SQLITE_AVAILABLE or not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO metrics_log (timestamp, log_type, model_name, data)
                VALUES (?, ?, ?, ?)
            """, (
                entry.timestamp.isoformat(),
                entry.log_type,
                entry.model_name,
                json.dumps(entry.data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log to database: {e}")
    
    def _flush_logs(self):
        """Flush buffered logs."""
        # This is mainly for future extension if we add buffering
        pass
    
    def get_metrics_summary(self, 
                           model_name: str = None,
                           log_type: str = None,
                           hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Args:
            model_name: Filter by model name
            log_type: Filter by log type
            hours: Hours to look back
            
        Returns:
            Metrics summary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter entries
        filtered_entries = []
        with self._lock:
            for entry in self.log_entries:
                if entry.timestamp < cutoff_time:
                    continue
                
                if model_name and entry.model_name != model_name:
                    continue
                
                if log_type and entry.log_type != log_type:
                    continue
                
                filtered_entries.append(entry)
        
        # Summarize
        summary = {
            "total_entries": len(filtered_entries),
            "time_range_hours": hours,
            "models": set(),
            "log_types": set(),
            "latest_timestamp": None,
            "by_model": defaultdict(lambda: defaultdict(int)),
            "by_log_type": defaultdict(int)
        }
        
        for entry in filtered_entries:
            summary["models"].add(entry.model_name)
            summary["log_types"].add(entry.log_type)
            summary["by_model"][entry.model_name][entry.log_type] += 1
            summary["by_log_type"][entry.log_type] += 1
            
            if summary["latest_timestamp"] is None or entry.timestamp > summary["latest_timestamp"]:
                summary["latest_timestamp"] = entry.timestamp
        
        # Convert sets to lists for JSON serialization
        summary["models"] = list(summary["models"])
        summary["log_types"] = list(summary["log_types"])
        summary["by_model"] = dict(summary["by_model"])
        summary["by_log_type"] = dict(summary["by_log_type"])
        
        if summary["latest_timestamp"]:
            summary["latest_timestamp"] = summary["latest_timestamp"].isoformat()
        
        return summary
    
    def get_performance_history(self, 
                               model_name: str,
                               hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance metrics history for a model."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        history = []
        with self._lock:
            for entry in self.model_metrics[model_name]["performance"]:
                if entry.timestamp >= cutoff_time:
                    history.append(entry.to_dict())
        
        return sorted(history, key=lambda x: x["timestamp"])
    
    def get_drift_history(self, 
                         model_name: str,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get drift detection history for a model."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        history = []
        with self._lock:
            for entry in self.model_metrics[model_name]["drift"]:
                if entry.timestamp >= cutoff_time:
                    history.append(entry.to_dict())
        
        return sorted(history, key=lambda x: x["timestamp"])
    
    def get_alerts(self, 
                  model_name: str = None,
                  severity: str = None,
                  hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = []
        with self._lock:
            for entry in self.log_entries:
                if (entry.log_type == "alert" and 
                    entry.timestamp >= cutoff_time):
                    
                    if model_name and entry.model_name != model_name:
                        continue
                    
                    if severity and entry.data.get("severity") != severity:
                        continue
                    
                    alerts.append(entry.to_dict())
        
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up old log files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            for log_file in self.log_dir.glob("*.jsonl"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
        
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
    
    def export_metrics(self, 
                      output_path: str,
                      model_name: str = None,
                      start_date: datetime = None,
                      end_date: datetime = None) -> bool:
        """
        Export metrics to file.
        
        Args:
            output_path: Output file path
            model_name: Filter by model name
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            True if successful
        """
        try:
            # Filter entries
            filtered_entries = []
            with self._lock:
                for entry in self.log_entries:
                    if model_name and entry.model_name != model_name:
                        continue
                    
                    if start_date and entry.timestamp < start_date:
                        continue
                    
                    if end_date and entry.timestamp > end_date:
                        continue
                    
                    filtered_entries.append(entry.to_dict())
            
            # Export
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_entries, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(filtered_entries)} metrics to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def close(self):
        """Close the metrics logger."""
        self._stop_background = True
        if self._background_thread:
            self._background_thread.join(timeout=5)
        
        self._flush_logs()
        logger.info("Metrics logger closed")


# Utility functions

def create_metrics_logger(config: Dict[str, Any]) -> MetricsLogger:
    """
    Create metrics logger from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MetricsLogger instance
    """
    return MetricsLogger(
        log_dir=config.get("log_dir", "logs"),
        mlflow_tracking_uri=config.get("mlflow_tracking_uri"),
        mlflow_experiment_name=config.get("mlflow_experiment_name", "model_monitoring"),
        db_path=config.get("db_path"),
        max_log_entries=config.get("max_log_entries", 10000),
        auto_flush_interval=config.get("auto_flush_interval", 60)
    )


def get_system_metrics() -> Dict[str, float]:
    """Get current system metrics."""
    import psutil
    
    return {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }