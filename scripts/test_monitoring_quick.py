#!/usr/bin/env python3

"""
Quick Monitoring Test

Simple test to verify monitoring components work.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance_monitor():
    """Test performance monitoring."""
    logger.info("🧪 Testing Performance Monitor...")
    
    try:
        from src.monitoring.performance_monitor import ModelPerformanceMonitor, PerformanceMetrics
        
        # Initialize monitor
        monitor = ModelPerformanceMonitor()
        
        # Generate test data
        n_samples = 100
        y_true = np.random.choice(['A', 'B', 'C'], n_samples)
        y_pred = y_true.copy()
        
        # Introduce some errors
        error_indices = np.random.choice(n_samples, 20, replace=False)
        for idx in error_indices:
            y_pred[idx] = np.random.choice([c for c in ['A', 'B', 'C'] if c != y_true[idx]])
        
        # Log predictions
        monitor.log_batch_predictions(
            y_true=y_true,
            y_pred=y_pred,
            prediction_times=[0.05] * n_samples
        )
        
        # Get performance
        performance = monitor.get_current_performance()
        
        logger.info(f"   ✅ Accuracy: {performance.accuracy:.3f}")
        logger.info(f"   ✅ Precision: {performance.precision:.3f}")
        logger.info(f"   ✅ F1 Score: {performance.f1_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Performance monitor test failed: {e}")
        return False

def test_drift_detector():
    """Test drift detection."""
    logger.info("🧪 Testing Drift Detector...")
    
    try:
        from src.monitoring.drift_detector import DataDriftDetector
        
        # Initialize detector
        detector = DataDriftDetector(drift_threshold=0.3)
        
        # Generate reference data
        np.random.seed(42)
        ref_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2]),
            'priority': np.random.choice(['low', 'medium', 'high'], 500),
            'description': [f"Sample text {i}" for i in range(500)]
        })
        
        # Set reference data
        detector.set_reference_data(
            reference_data=ref_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
        
        # Test no drift
        current_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 200, p=[0.5, 0.3, 0.2]),
            'priority': np.random.choice(['low', 'medium', 'high'], 200),
            'description': [f"Sample text {i}" for i in range(200)]
        })
        
        result = detector.detect_drift(
            current_data=current_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
        
        logger.info(f"   ✅ No drift test - Score: {result.drift_score:.3f}, Detected: {result.overall_drift_detected}")
        
        # Test with drift
        drift_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 200, p=[0.1, 0.2, 0.7]),  # Different distribution
            'priority': np.random.choice(['low', 'medium', 'high'], 200),
            'description': [f"Different text {i}" for i in range(200)]
        })
        
        drift_result = detector.detect_drift(
            current_data=drift_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
        
        logger.info(f"   ✅ Drift test - Score: {drift_result.drift_score:.3f}, Detected: {drift_result.overall_drift_detected}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Drift detector test failed: {e}")
        return False

def test_metrics_logger():
    """Test metrics logging."""
    logger.info("🧪 Testing Metrics Logger...")
    
    try:
        from src.monitoring.metrics_logger import MetricsLogger
        from src.monitoring.performance_monitor import PerformanceMetrics
        
        # Initialize logger
        metrics_logger = MetricsLogger(log_dir="logs/test_monitoring")
        
        # Create sample metrics
        sample_metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85
        )
        
        # Log metrics
        metrics_logger.log_performance_metrics("test_model", sample_metrics)
        
        # Get summary
        summary = metrics_logger.get_metrics_summary("test_model", hours=1)
        
        logger.info(f"   ✅ Logged entries: {summary['total_entries']}")
        logger.info(f"   ✅ Models tracked: {summary['models']}")
        
        metrics_logger.close()
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Metrics logger test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Quick Monitoring Test")
    logger.info("=" * 50)
    
    results = []
    
    # Test individual components
    results.append(test_performance_monitor())
    results.append(test_drift_detector())
    results.append(test_metrics_logger())
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS")
    logger.info("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"✅ Performance Monitor: {'PASSED' if results[0] else 'FAILED'}")
    logger.info(f"✅ Drift Detector: {'PASSED' if results[1] else 'FAILED'}")
    logger.info(f"✅ Metrics Logger: {'PASSED' if results[2] else 'FAILED'}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All monitoring components are working!")
    else:
        logger.warning("⚠️ Some components have issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)