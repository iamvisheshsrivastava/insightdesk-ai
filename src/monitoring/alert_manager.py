# src/monitoring/alert_manager.py

"""
Alert Management for the Intelligent Support System.

This module handles alert generation, management, and notification for
model performance degradation, data drift, and system health issues.
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import threading
import time
from collections import defaultdict, deque
import warnings

# Slack integration (optional)
try:
    import slack_sdk
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logging.warning("Slack SDK not available. Slack notifications will be disabled.")

# Webhook support
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Webhook notifications will be disabled.")

from .performance_monitor import PerformanceMetrics
from .drift_detector import DriftResult

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SYSTEM_HEALTH = "system_health"
    MODEL_ERROR = "model_error"
    CUSTOM = "custom"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # e.g., "accuracy < 0.8", "drift_score > 0.3"
    threshold_value: float
    comparison_operator: str  # "lt", "gt", "le", "ge", "eq", "ne"
    
    # Rule behavior
    enabled: bool = True
    consecutive_violations: int = 1  # Number of consecutive violations before alerting
    cooldown_minutes: int = 60  # Minimum time between alerts of same type
    
    # Notification settings
    notify_email: bool = True
    notify_slack: bool = False
    notify_webhook: bool = False
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if the rule condition is met."""
        operators = {
            "lt": lambda x, y: x < y,
            "gt": lambda x, y: x > y,
            "le": lambda x, y: x <= y,
            "ge": lambda x, y: x >= y,
            "eq": lambda x, y: x == y,
            "ne": lambda x, y: x != y
        }
        
        if self.comparison_operator not in operators:
            logger.error(f"Unknown operator: {self.comparison_operator}")
            return False
        
        return operators[self.comparison_operator](value, self.threshold_value)


@dataclass
class Alert:
    """Alert instance."""
    
    id: str
    rule_name: str
    alert_type: AlertType
    severity: AlertSeverity
    model_name: str
    
    # Alert details
    title: str
    message: str
    actual_value: float
    threshold_value: float
    
    # Status and timing
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Notification tracking
    notifications_sent: List[str] = field(default_factory=list)  # Channels where alert was sent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "model_name": self.model_name,
            "title": self.title,
            "message": self.message,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "notifications_sent": self.notifications_sent
        }
    
    def acknowledge(self, note: str = ""):
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        if note:
            self.metadata["acknowledgment_note"] = note
    
    def resolve(self, note: str = ""):
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        if note:
            self.metadata["resolution_note"] = note


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    
    # Email settings
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Slack settings
    slack_enabled: bool = False
    slack_token: str = ""
    slack_channel: str = ""
    
    # Webhook settings
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # Custom notification handlers
    custom_handlers: List[Callable] = field(default_factory=list)


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, notification_config: NotificationConfig = None):
        """
        Initialize alert manager.
        
        Args:
            notification_config: Notification configuration
        """
        self.notification_config = notification_config or NotificationConfig()
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Violation tracking for consecutive alerts
        self.violation_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_alert_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        
        # Background notification queue
        self.notification_queue: deque = deque()
        self._notification_lock = threading.Lock()
        self._stop_notifications = False
        self._notification_thread = None
        
        # Start notification worker
        self._start_notification_worker()
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def check_performance_alerts(self, 
                                model_name: str, 
                                metrics: PerformanceMetrics):
        """Check for performance-related alerts."""
        metric_values = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score
        }
        
        for rule_name, rule in self.alert_rules.items():
            if (rule.enabled and 
                rule.alert_type == AlertType.PERFORMANCE_DEGRADATION):
                
                # Extract metric name from condition
                metric_name = self._extract_metric_name(rule.condition)
                if metric_name in metric_values:
                    value = metric_values[metric_name]
                    
                    if rule.evaluate(value):
                        self._handle_rule_violation(
                            rule, model_name, metric_name, value
                        )
                    else:
                        self._reset_violation_count(rule_name, model_name)
    
    def check_drift_alerts(self, 
                          model_name: str, 
                          drift_result: DriftResult):
        """Check for data drift alerts."""
        drift_values = {
            "drift_score": drift_result.drift_score,
            "num_drifted_features": drift_result.num_drifted_features,
            "max_drift_score": drift_result.max_drift_score,
            "avg_drift_score": drift_result.avg_drift_score
        }
        
        for rule_name, rule in self.alert_rules.items():
            if (rule.enabled and 
                rule.alert_type == AlertType.DATA_DRIFT):
                
                # Extract metric name from condition
                metric_name = self._extract_metric_name(rule.condition)
                if metric_name in drift_values:
                    value = drift_values[metric_name]
                    
                    if rule.evaluate(value):
                        self._handle_rule_violation(
                            rule, model_name, metric_name, value
                        )
                    else:
                        self._reset_violation_count(rule_name, model_name)
    
    def check_system_alerts(self,
                           model_name: str,
                           cpu_usage: float,
                           memory_usage: float,
                           prediction_latency: float,
                           error_rate: float):
        """Check for system health alerts."""
        system_values = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "prediction_latency": prediction_latency,
            "error_rate": error_rate
        }
        
        for rule_name, rule in self.alert_rules.items():
            if (rule.enabled and 
                rule.alert_type == AlertType.SYSTEM_HEALTH):
                
                metric_name = self._extract_metric_name(rule.condition)
                if metric_name in system_values:
                    value = system_values[metric_name]
                    
                    if rule.evaluate(value):
                        self._handle_rule_violation(
                            rule, model_name, metric_name, value
                        )
                    else:
                        self._reset_violation_count(rule_name, model_name)
    
    def create_custom_alert(self,
                           model_name: str,
                           title: str,
                           message: str,
                           severity: AlertSeverity,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Create a custom alert.
        
        Args:
            model_name: Name of the model
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        alert_id = f"custom_{int(datetime.utcnow().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            rule_name="custom",
            alert_type=AlertType.CUSTOM,
            severity=severity,
            model_name=model_name,
            title=title,
            message=message,
            actual_value=0.0,
            threshold_value=0.0,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._queue_notification(alert)
        
        logger.info(f"Created custom alert: {alert_id}")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, note: str = "") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge(note)
            logger.info(f"Acknowledged alert: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, note: str = "") -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve(note)
            
            # Move to history and remove from active
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            logger.info(f"Resolved alert: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self, 
                         model_name: str = None,
                         severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts."""
        alerts = list(self.active_alerts.values())
        
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_history(self,
                         hours: int = 24,
                         model_name: str = None) -> List[Alert]:
        """Get alert history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff_time
        ]
        
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = list(self.active_alerts.values())
        
        summary = {
            "total_active_alerts": len(active_alerts),
            "by_severity": defaultdict(int),
            "by_type": defaultdict(int),
            "by_model": defaultdict(int),
            "recent_alerts_24h": 0,
            "total_rules": len(self.alert_rules),
            "enabled_rules": sum(1 for rule in self.alert_rules.values() if rule.enabled)
        }
        
        # Count by categories
        for alert in active_alerts:
            summary["by_severity"][alert.severity.value] += 1
            summary["by_type"][alert.alert_type.value] += 1
            summary["by_model"][alert.model_name] += 1
        
        # Count recent alerts
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        summary["recent_alerts_24h"] = sum(
            1 for alert in self.alert_history
            if alert.created_at >= cutoff_time
        )
        
        # Convert defaultdicts to regular dicts
        summary["by_severity"] = dict(summary["by_severity"])
        summary["by_type"] = dict(summary["by_type"])
        summary["by_model"] = dict(summary["by_model"])
        
        return summary
    
    def _extract_metric_name(self, condition: str) -> str:
        """Extract metric name from condition string."""
        # Simple extraction - assumes format like "accuracy < 0.8"
        parts = condition.split()
        if len(parts) >= 1:
            return parts[0].strip()
        return ""
    
    def _handle_rule_violation(self, 
                              rule: AlertRule, 
                              model_name: str, 
                              metric_name: str, 
                              value: float):
        """Handle a rule violation."""
        key = f"{rule.name}_{model_name}"
        
        # Increment violation count
        self.violation_counts[rule.name][model_name] += 1
        
        # Check if we should trigger an alert
        if self.violation_counts[rule.name][model_name] >= rule.consecutive_violations:
            # Check cooldown
            if self._is_in_cooldown(rule.name, model_name, rule.cooldown_minutes):
                return
            
            # Create alert
            alert_id = f"{rule.name}_{model_name}_{int(datetime.utcnow().timestamp())}"
            
            alert = Alert(
                id=alert_id,
                rule_name=rule.name,
                alert_type=rule.alert_type,
                severity=rule.severity,
                model_name=model_name,
                title=f"{rule.alert_type.value.replace('_', ' ').title()} Alert",
                message=f"{metric_name} value {value:.3f} violates threshold {rule.threshold_value}",
                actual_value=value,
                threshold_value=rule.threshold_value,
                tags=rule.tags,
                metadata={
                    "metric_name": metric_name,
                    "condition": rule.condition,
                    "consecutive_violations": self.violation_counts[rule.name][model_name]
                }
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update last alert time
            self.last_alert_times[rule.name][model_name] = datetime.utcnow()
            
            # Reset violation count
            self.violation_counts[rule.name][model_name] = 0
            
            # Queue notification
            self._queue_notification(alert)
            
            logger.warning(f"Alert triggered: {alert_id}")
    
    def _reset_violation_count(self, rule_name: str, model_name: str):
        """Reset violation count for a rule."""
        if rule_name in self.violation_counts:
            self.violation_counts[rule_name][model_name] = 0
    
    def _is_in_cooldown(self, rule_name: str, model_name: str, cooldown_minutes: int) -> bool:
        """Check if alert is in cooldown period."""
        if rule_name not in self.last_alert_times:
            return False
        
        if model_name not in self.last_alert_times[rule_name]:
            return False
        
        last_alert = self.last_alert_times[rule_name][model_name]
        cooldown_time = last_alert + timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() < cooldown_time
    
    def _queue_notification(self, alert: Alert):
        """Queue notification for an alert."""
        with self._notification_lock:
            self.notification_queue.append(alert)
    
    def _start_notification_worker(self):
        """Start background notification worker."""
        self._notification_thread = threading.Thread(
            target=self._notification_worker,
            daemon=True
        )
        self._notification_thread.start()
    
    def _notification_worker(self):
        """Background worker for sending notifications."""
        while not self._stop_notifications:
            try:
                # Check for notifications to send
                alert = None
                with self._notification_lock:
                    if self.notification_queue:
                        alert = self.notification_queue.popleft()
                
                if alert:
                    self._send_notifications(alert)
                else:
                    time.sleep(1)  # Wait before checking again
                    
            except Exception as e:
                logger.error(f"Notification worker error: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        rule = self.alert_rules.get(alert.rule_name)
        
        # Email notification
        if (self.notification_config.email_enabled and 
            (not rule or rule.notify_email)):
            if self._send_email_notification(alert):
                alert.notifications_sent.append("email")
        
        # Slack notification
        if (self.notification_config.slack_enabled and 
            (not rule or rule.notify_slack)):
            if self._send_slack_notification(alert):
                alert.notifications_sent.append("slack")
        
        # Webhook notification
        if (self.notification_config.webhook_enabled and 
            (not rule or rule.notify_webhook)):
            if self._send_webhook_notification(alert):
                alert.notifications_sent.append("webhook")
        
        # Custom handlers
        for handler in self.notification_config.custom_handlers:
            try:
                handler(alert)
                alert.notifications_sent.append("custom")
            except Exception as e:
                logger.error(f"Custom notification handler error: {e}")
    
    def _send_email_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            msg = MimeMultipart()
            msg['From'] = self.notification_config.email_from
            msg['To'] = ", ".join(self.notification_config.email_to)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
- Model: {alert.model_name}
- Type: {alert.alert_type.value}
- Severity: {alert.severity.value}
- Message: {alert.message}
- Actual Value: {alert.actual_value}
- Threshold: {alert.threshold_value}
- Time: {alert.created_at.isoformat()}

Alert ID: {alert.id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.notification_config.smtp_server, 
                                self.notification_config.smtp_port)
            server.starttls()
            server.login(self.notification_config.smtp_username, 
                        self.notification_config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.notification_config.email_from, 
                          self.notification_config.email_to, text)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_slack_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not SLACK_AVAILABLE:
            return False
        
        try:
            client = slack_sdk.WebClient(token=self.notification_config.slack_token)
            
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning", 
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")
            
            attachment = {
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Model", "value": alert.model_name, "short": True},
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Actual Value", "value": f"{alert.actual_value:.3f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold_value:.3f}", "short": True}
                ],
                "footer": f"Alert ID: {alert.id}",
                "ts": int(alert.created_at.timestamp())
            }
            
            response = client.chat_postMessage(
                channel=self.notification_config.slack_channel,
                text=f"Alert: {alert.title}",
                attachments=[attachment]
            )
            
            logger.info(f"Slack notification sent for alert: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _send_webhook_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not REQUESTS_AVAILABLE:
            return False
        
        try:
            payload = alert.to_dict()
            
            response = requests.post(
                self.notification_config.webhook_url,
                json=payload,
                headers=self.notification_config.webhook_headers,
                timeout=30
            )
            
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def close(self):
        """Close the alert manager."""
        self._stop_notifications = True
        if self._notification_thread:
            self._notification_thread.join(timeout=5)
        
        logger.info("Alert manager closed")


# Predefined alert rules

def create_default_performance_rules() -> List[AlertRule]:
    """Create default performance alert rules."""
    return [
        AlertRule(
            name="low_accuracy_warning",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.MEDIUM,
            condition="accuracy < 0.8",
            threshold_value=0.8,
            comparison_operator="lt",
            consecutive_violations=2,
            description="Alert when model accuracy drops below 80%"
        ),
        AlertRule(
            name="low_accuracy_critical",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.CRITICAL,
            condition="accuracy < 0.7",
            threshold_value=0.7,
            comparison_operator="lt",
            consecutive_violations=1,
            description="Critical alert when model accuracy drops below 70%"
        ),
        AlertRule(
            name="low_f1_score",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            condition="f1_score < 0.75",
            threshold_value=0.75,
            comparison_operator="lt",
            consecutive_violations=2,
            description="Alert when F1 score drops below 75%"
        )
    ]


def create_default_drift_rules() -> List[AlertRule]:
    """Create default drift alert rules."""
    return [
        AlertRule(
            name="data_drift_warning",
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.MEDIUM,
            condition="drift_score > 0.3",
            threshold_value=0.3,
            comparison_operator="gt",
            consecutive_violations=1,
            description="Alert when data drift score exceeds 0.3"
        ),
        AlertRule(
            name="data_drift_critical",
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.CRITICAL,
            condition="drift_score > 0.5",
            threshold_value=0.5,
            comparison_operator="gt",
            consecutive_violations=1,
            description="Critical alert when data drift score exceeds 0.5"
        ),
        AlertRule(
            name="multiple_features_drift",
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            condition="num_drifted_features > 3",
            threshold_value=3,
            comparison_operator="gt",
            consecutive_violations=1,
            description="Alert when more than 3 features show drift"
        )
    ]


def create_default_system_rules() -> List[AlertRule]:
    """Create default system health alert rules."""
    return [
        AlertRule(
            name="high_cpu_usage",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.MEDIUM,
            condition="cpu_usage > 80",
            threshold_value=80.0,
            comparison_operator="gt",
            consecutive_violations=3,
            description="Alert when CPU usage exceeds 80%"
        ),
        AlertRule(
            name="high_memory_usage",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition="memory_usage > 90",
            threshold_value=90.0,
            comparison_operator="gt",
            consecutive_violations=2,
            description="Alert when memory usage exceeds 90%"
        ),
        AlertRule(
            name="high_prediction_latency",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.MEDIUM,
            condition="prediction_latency > 1000",
            threshold_value=1000.0,
            comparison_operator="gt",
            consecutive_violations=5,
            description="Alert when prediction latency exceeds 1000ms"
        )
    ]