#!/usr/bin/env python3

"""
Monitoring & Drift Detection Demo

This script demonstrates the monitoring and drift detection capabilities
of the Intelligent Support System.
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_monitoring_system():
    """Initialize the monitoring system components."""
    logger.info("🔧 Setting up monitoring system...")
    
    try:
        from src.monitoring.performance_monitor import ModelPerformanceMonitor, PerformanceMetrics
        from src.monitoring.drift_detector import DataDriftDetector, DriftResult
        from src.monitoring.metrics_logger import MetricsLogger
        from src.monitoring.alert_manager import (
            AlertManager, AlertRule, AlertSeverity, AlertType,
            create_default_performance_rules, create_default_drift_rules
        )
        
        # Initialize components
        performance_monitor = ModelPerformanceMonitor(
            model_name="support_system_demo",
            target_accuracy=0.85,
            mlflow_experiment_name="monitoring_demo"
        )
        
        drift_detector = DataDriftDetector(
            drift_threshold=0.3,
            min_sample_size=50
        )
        
        metrics_logger = MetricsLogger(
            log_dir="logs/monitoring_demo",
            mlflow_experiment_name="monitoring_demo"
        )
        
        alert_manager = AlertManager()
        
        # Add default alert rules
        for rule in create_default_performance_rules():
            alert_manager.add_alert_rule(rule)
        
        for rule in create_default_drift_rules():
            alert_manager.add_alert_rule(rule)
        
        logger.info("✅ Monitoring system initialized successfully")
        
        return {
            "performance_monitor": performance_monitor,
            "drift_detector": drift_detector,
            "metrics_logger": metrics_logger,
            "alert_manager": alert_manager
        }
        
    except ImportError as e:
        logger.error(f"❌ Failed to import monitoring modules: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize monitoring system: {e}")
        return None

def generate_sample_data(n_samples=1000, introduce_drift=False):
    """Generate sample support ticket data."""
    logger.info(f"📊 Generating {n_samples} sample tickets...")
    
    np.random.seed(42)
    
    # Base distributions
    products = ["web_app", "mobile_app", "api", "database", "infrastructure"]
    priorities = ["low", "medium", "high", "critical"]
    categories = ["bug", "feature_request", "question", "incident"]
    channels = ["email", "chat", "phone", "api"]
    
    # Normal distribution weights
    product_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    priority_weights = [0.4, 0.3, 0.2, 0.1]
    category_weights = [0.4, 0.25, 0.2, 0.15]
    channel_weights = [0.5, 0.3, 0.15, 0.05]
    
    # Introduce drift if requested
    if introduce_drift:
        logger.info("🌊 Introducing data drift...")
        # Shift distributions to simulate drift
        product_weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # Reverse trend
        priority_weights = [0.1, 0.2, 0.3, 0.4]  # More critical tickets
        category_weights = [0.6, 0.15, 0.15, 0.1]  # More bugs
        channel_weights = [0.2, 0.4, 0.25, 0.15]  # More chat/phone
    
    data = []
    for i in range(n_samples):
        # Generate ticket data
        ticket = {
            "ticket_id": f"TICKET-{i+1:06d}",
            "subject": f"Sample issue {i+1}",
            "description": f"This is a sample support ticket description for issue {i+1}. " +
                          f"It contains various technical details and user reports.",
            "product": np.random.choice(products, p=product_weights),
            "priority": np.random.choice(priorities, p=priority_weights),
            "category": np.random.choice(categories, p=category_weights),
            "channel": np.random.choice(channels, p=channel_weights),
            "customer_tier": np.random.choice(["basic", "premium", "enterprise"], p=[0.6, 0.3, 0.1]),
            "region": np.random.choice(["us", "eu", "asia", "other"], p=[0.4, 0.3, 0.2, 0.1]),
            "account_age_days": max(1, int(np.random.exponential(365))),
            "previous_tickets": max(0, int(np.random.poisson(2)))
        }
        
        # Add some text variation for drift detection
        if introduce_drift:
            # Make descriptions shorter and more technical
            ticket["description"] = f"ERROR: System failure in {ticket['product']}. Code 500."
            ticket["subject"] = f"CRITICAL: {ticket['product']} down"
        
        data.append(ticket)
    
    return pd.DataFrame(data)

def simulate_model_predictions(df, introduce_performance_issues=False):
    """Simulate model predictions with optional performance degradation."""
    logger.info("🤖 Simulating model predictions...")
    
    n_samples = len(df)
    
    # Base performance metrics
    base_accuracy = 0.88
    base_precision = 0.86
    base_recall = 0.84
    base_f1 = 0.85
    
    if introduce_performance_issues:
        logger.info("📉 Introducing performance degradation...")
        # Degrade performance
        base_accuracy = 0.72
        base_precision = 0.70
        base_recall = 0.68
        base_f1 = 0.69
    
    # Add some noise to metrics
    noise_factor = 0.02
    accuracy = base_accuracy + np.random.normal(0, noise_factor)
    precision = base_precision + np.random.normal(0, noise_factor)
    recall = base_recall + np.random.normal(0, noise_factor)
    f1_score = base_f1 + np.random.normal(0, noise_factor)
    
    # Ensure metrics are in valid range
    accuracy = max(0, min(1, accuracy))
    precision = max(0, min(1, precision))
    recall = max(0, min(1, recall))
    f1_score = max(0, min(1, f1_score))
    
    # Generate predictions (simplified)
    y_true = np.random.choice(["bug", "feature_request", "question", "incident"], n_samples)
    y_pred = y_true.copy()
    
    # Introduce some errors based on accuracy
    n_errors = int(n_samples * (1 - accuracy))
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    
    categories = ["bug", "feature_request", "question", "incident"]
    for idx in error_indices:
        # Random wrong prediction
        y_pred[idx] = np.random.choice([c for c in categories if c != y_true[idx]])
    
    return y_true, y_pred, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def demo_performance_monitoring(monitoring_system, df):
    """Demonstrate performance monitoring capabilities."""
    logger.info("📈 Demonstrating performance monitoring...")
    
    performance_monitor = monitoring_system["performance_monitor"]
    metrics_logger = monitoring_system["metrics_logger"]
    alert_manager = monitoring_system["alert_manager"]
    
    # Simulate multiple model prediction cycles
    cycles = [
        {"name": "Normal Performance", "degrade": False},
        {"name": "Slight Degradation", "degrade": False},
        {"name": "Performance Issues", "degrade": True},
        {"name": "Recovery", "degrade": False}
    ]
    
    for i, cycle in enumerate(cycles):
        logger.info(f"🔄 Cycle {i+1}: {cycle['name']}")
        
        # Simulate predictions
        y_true, y_pred, metrics_dict = simulate_model_predictions(
            df, introduce_performance_issues=cycle["degrade"]
        )
        
        # Log predictions
        performance_monitor.log_batch_predictions(
            y_true=y_true,
            y_pred=y_pred,
            prediction_times=[0.05] * len(y_true),  # 50ms avg
            metadata={"cycle": cycle["name"]}
        )
        
        # Get current performance
        current_metrics = performance_monitor.get_current_performance()
        logger.info(f"   Accuracy: {current_metrics.accuracy:.3f}")
        logger.info(f"   Precision: {current_metrics.precision:.3f}")
        logger.info(f"   Recall: {current_metrics.recall:.3f}")
        logger.info(f"   F1 Score: {current_metrics.f1_score:.3f}")
        
        # Log metrics
        metrics_logger.log_performance_metrics(
            "support_system_demo", current_metrics
        )
        
        # Check for alerts
        alert_manager.check_performance_alerts(
            "support_system_demo", current_metrics
        )
        
        # Wait a bit between cycles
        time.sleep(1)
    
    # Show performance summary
    status = performance_monitor.get_current_status()
    logger.info(f"📊 Performance Summary:")
    logger.info(f"   Total predictions: {status.get('total_predictions', 0)}")
    logger.info(f"   Average accuracy: {status.get('avg_accuracy', 0):.3f}")
    logger.info(f"   Performance trend: {status.get('performance_trend', 'stable')}")
    
    return status

def demo_drift_detection(monitoring_system):
    """Demonstrate drift detection capabilities."""
    logger.info("🌊 Demonstrating drift detection...")
    
    drift_detector = monitoring_system["drift_detector"]
    metrics_logger = monitoring_system["metrics_logger"]
    alert_manager = monitoring_system["alert_manager"]
    
    # Generate reference (training) data
    reference_data = generate_sample_data(n_samples=2000, introduce_drift=False)
    logger.info(f"📊 Reference data: {len(reference_data)} samples")
    
    # Set reference data for drift detection
    drift_detector.set_reference_data(
        reference_data=reference_data,
        categorical_columns=["product", "priority", "category", "channel"],
        text_columns=["subject", "description"],
        model_name="support_system_demo"
    )
    
    # Test scenarios
    scenarios = [
        {"name": "Normal Data", "drift": False, "samples": 500},
        {"name": "Slight Changes", "drift": False, "samples": 300},
        {"name": "Data Drift Detected", "drift": True, "samples": 400}
    ]
    
    drift_results = []
    
    for scenario in scenarios:
        logger.info(f"🔄 Testing: {scenario['name']}")
        
        # Generate current data
        current_data = generate_sample_data(
            n_samples=scenario["samples"],
            introduce_drift=scenario["drift"]
        )
        
        # Detect drift
        drift_result = drift_detector.detect_drift(
            current_data=current_data,
            categorical_columns=["product", "priority", "category", "channel"],
            text_columns=["subject", "description"],
            model_name="support_system_demo"
        )
        
        drift_results.append(drift_result)
        
        logger.info(f"   Drift Score: {drift_result.drift_score:.3f}")
        logger.info(f"   Drift Detected: {drift_result.overall_drift_detected}")
        logger.info(f"   Drifted Features: {drift_result.num_drifted_features}/{drift_result.total_features}")
        
        if drift_result.most_drifted_feature:
            logger.info(f"   Most Drifted Feature: {drift_result.most_drifted_feature}")
        
        # Log drift results
        metrics_logger.log_drift_metrics(
            "support_system_demo", drift_result
        )
        
        # Check for drift alerts
        alert_manager.check_drift_alerts(
            "support_system_demo", drift_result
        )
        
        time.sleep(1)
    
    # Show drift summary
    drift_status = drift_detector.get_drift_status("support_system_demo")
    logger.info(f"🌊 Drift Detection Summary:")
    logger.info(f"   Current Status: {drift_status.get('current_status', 'unknown')}")
    logger.info(f"   Total Evaluations: {drift_status.get('total_evaluations', 0)}")
    logger.info(f"   Latest Drift Score: {drift_status.get('latest_drift_score', 0):.3f}")
    
    return drift_results

def demo_alert_management(monitoring_system):
    """Demonstrate alert management capabilities."""
    logger.info("🚨 Demonstrating alert management...")
    
    alert_manager = monitoring_system["alert_manager"]
    
    # Get current alerts
    active_alerts = alert_manager.get_active_alerts()
    logger.info(f"📋 Active Alerts: {len(active_alerts)}")
    
    for alert in active_alerts[:5]:  # Show first 5 alerts
        logger.info(f"   🚨 {alert.severity.value.upper()}: {alert.title}")
        logger.info(f"      Model: {alert.model_name}")
        logger.info(f"      Message: {alert.message}")
        logger.info(f"      Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create a custom alert
    alert_id = alert_manager.create_custom_alert(
        model_name="support_system_demo",
        title="Demo Custom Alert",
        message="This is a demonstration of custom alert creation",
        severity=AlertSeverity.MEDIUM,
        metadata={"demo": True, "source": "monitoring_demo"}
    )
    
    logger.info(f"✅ Created custom alert: {alert_id}")
    
    # Acknowledge the alert
    alert_manager.acknowledge_alert(alert_id, "Alert acknowledged during demo")
    logger.info(f"✅ Acknowledged alert: {alert_id}")
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    logger.info(f"📊 Alert Summary:")
    logger.info(f"   Total Active: {summary['total_active_alerts']}")
    logger.info(f"   By Severity: {summary['by_severity']}")
    logger.info(f"   By Type: {summary['by_type']}")
    
    return summary

def demo_metrics_logging(monitoring_system):
    """Demonstrate metrics logging capabilities."""
    logger.info("📝 Demonstrating metrics logging...")
    
    metrics_logger = monitoring_system["metrics_logger"]
    
    # Get metrics summary
    summary = metrics_logger.get_metrics_summary(
        model_name="support_system_demo",
        hours=1
    )
    
    logger.info(f"📊 Metrics Summary (last 1 hour):")
    logger.info(f"   Total Entries: {summary['total_entries']}")
    logger.info(f"   Models: {summary['models']}")
    logger.info(f"   Log Types: {summary['log_types']}")
    logger.info(f"   By Model: {summary['by_model']}")
    
    # Export metrics
    export_path = "logs/monitoring_demo/demo_metrics_export.json"
    success = metrics_logger.export_metrics(
        output_path=export_path,
        model_name="support_system_demo"
    )
    
    if success:
        logger.info(f"✅ Metrics exported to: {export_path}")
    else:
        logger.error("❌ Failed to export metrics")
    
    return summary

async def main():
    """Main demo function."""
    logger.info("🚀 Starting Monitoring & Drift Detection Demo")
    logger.info("=" * 60)
    
    # Setup monitoring system
    monitoring_system = setup_monitoring_system()
    if not monitoring_system:
        logger.error("❌ Failed to setup monitoring system. Exiting.")
        return
    
    try:
        # Generate sample data
        sample_data = generate_sample_data(n_samples=1000)
        logger.info(f"📊 Generated {len(sample_data)} sample tickets")
        
        # Demo 1: Performance Monitoring
        logger.info("\n" + "=" * 60)
        logger.info("📈 DEMO 1: Performance Monitoring")
        logger.info("=" * 60)
        performance_status = demo_performance_monitoring(monitoring_system, sample_data)
        
        # Demo 2: Drift Detection
        logger.info("\n" + "=" * 60)
        logger.info("🌊 DEMO 2: Drift Detection")
        logger.info("=" * 60)
        drift_results = demo_drift_detection(monitoring_system)
        
        # Demo 3: Alert Management
        logger.info("\n" + "=" * 60)
        logger.info("🚨 DEMO 3: Alert Management")
        logger.info("=" * 60)
        alert_summary = demo_alert_management(monitoring_system)
        
        # Demo 4: Metrics Logging
        logger.info("\n" + "=" * 60)
        logger.info("📝 DEMO 4: Metrics Logging")
        logger.info("=" * 60)
        metrics_summary = demo_metrics_logging(monitoring_system)
        
        # Final Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 DEMO SUMMARY")
        logger.info("=" * 60)
        logger.info("✅ Performance monitoring demonstrated")
        logger.info("✅ Drift detection demonstrated")
        logger.info("✅ Alert management demonstrated") 
        logger.info("✅ Metrics logging demonstrated")
        
        logger.info(f"\n📊 Key Results:")
        logger.info(f"   🔄 Performance cycles completed: 4")
        logger.info(f"   🌊 Drift scenarios tested: 3")
        logger.info(f"   🚨 Total alerts generated: {alert_summary.get('total_active_alerts', 0)}")
        logger.info(f"   📝 Metrics logged: {metrics_summary.get('total_entries', 0)}")
        
        # Save demo results
        demo_results = {
            "timestamp": datetime.now().isoformat(),
            "performance_status": performance_status,
            "drift_results": [dr.to_dict() for dr in drift_results],
            "alert_summary": alert_summary,
            "metrics_summary": metrics_summary
        }
        
        results_path = "monitoring_demo_results.json"
        with open(results_path, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Demo results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("\n🧹 Cleaning up...")
        if monitoring_system.get("metrics_logger"):
            monitoring_system["metrics_logger"].close()
        if monitoring_system.get("alert_manager"):
            monitoring_system["alert_manager"].close()
        
        logger.info("✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())