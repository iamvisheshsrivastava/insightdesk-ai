#!/usr/bin/env python3

"""
Monitoring System Tests

This script tests all components of the monitoring and drift detection system.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

class TestPerformanceMonitor(unittest.TestCase):
    """Test the performance monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.monitoring.performance_monitor import ModelPerformanceMonitor
        
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = ModelPerformanceMonitor(
            model_name="test_model",
            target_accuracy=0.8,
            alert_threshold=0.75
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_calculation(self):
        """Test performance metrics calculation."""
        # Generate test data
        n_samples = 1000
        y_true = np.random.choice(['A', 'B', 'C'], n_samples)
        y_pred = y_true.copy()
        
        # Introduce some errors (20% error rate)
        error_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
        for idx in error_indices:
            # Wrong prediction
            y_pred[idx] = np.random.choice([c for c in ['A', 'B', 'C'] if c != y_true[idx]])
        
        # Log predictions
        self.monitor.log_batch_predictions(
            y_true=y_true,
            y_pred=y_pred,
            prediction_times=[0.05] * n_samples
        )
        
        # Get performance
        performance = self.monitor.get_current_performance()
        
        # Check results
        self.assertAlmostEqual(performance.accuracy, 0.8, places=1)
        self.assertGreater(performance.precision, 0.7)
        self.assertGreater(performance.recall, 0.7)
        self.assertGreater(performance.f1_score, 0.7)
        
        print(f"✅ Performance calculation: Accuracy={performance.accuracy:.3f}")
    
    def test_performance_trend_detection(self):
        """Test performance trend detection."""
        # Simulate degrading performance over time
        for i in range(5):
            accuracy = 0.9 - (i * 0.05)  # Decreasing accuracy
            n_samples = 100
            n_errors = int(n_samples * (1 - accuracy))
            
            y_true = np.random.choice(['A', 'B'], n_samples)
            y_pred = y_true.copy()
            
            # Introduce errors
            error_indices = np.random.choice(n_samples, n_errors, replace=False)
            for idx in error_indices:
                y_pred[idx] = 'B' if y_true[idx] == 'A' else 'A'
            
            self.monitor.log_batch_predictions(y_true, y_pred, [0.05] * n_samples)
        
        # Check trend detection
        status = self.monitor.get_current_status()
        trend = status.get('performance_trend', 'stable')
        
        print(f"✅ Trend detection: {trend}")
        self.assertIn(trend, ['declining', 'stable', 'improving'])
    
    def test_alert_generation(self):
        """Test alert generation for poor performance."""
        # Generate poor performance data
        n_samples = 100
        y_true = np.random.choice(['A', 'B'], n_samples)
        y_pred = np.random.choice(['A', 'B'], n_samples)  # Random predictions
        
        self.monitor.log_batch_predictions(y_true, y_pred, [0.05] * n_samples)
        
        performance = self.monitor.get_current_performance()
        needs_alert = self.monitor.should_alert()
        
        print(f"✅ Alert generation: accuracy={performance.accuracy:.3f}, needs_alert={needs_alert}")
        
        # Poor performance should trigger alert
        if performance.accuracy < 0.75:
            self.assertTrue(needs_alert)


class TestDriftDetector(unittest.TestCase):
    """Test the drift detection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.monitoring.drift_detector import DataDriftDetector
        
        self.drift_detector = DataDriftDetector(
            drift_threshold=0.3,
            min_sample_size=50
        )
        
        # Generate reference data
        np.random.seed(42)
        self.reference_data = self._generate_sample_data(1000, drift=False)
        
        self.drift_detector.set_reference_data(
            reference_data=self.reference_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
    
    def _generate_sample_data(self, n_samples, drift=False):
        """Generate sample data for testing."""
        np.random.seed(42 if not drift else 123)
        
        if drift:
            # Shifted distributions
            category_probs = [0.1, 0.3, 0.4, 0.2]  # Different from reference
            priority_probs = [0.2, 0.2, 0.3, 0.3]
        else:
            # Reference distributions
            category_probs = [0.4, 0.3, 0.2, 0.1]
            priority_probs = [0.5, 0.3, 0.15, 0.05]
        
        data = {
            'category': np.random.choice(['bug', 'feature', 'question', 'incident'], 
                                       n_samples, p=category_probs),
            'priority': np.random.choice(['low', 'medium', 'high', 'critical'], 
                                       n_samples, p=priority_probs),
            'description': [f"Sample description {i}" for i in range(n_samples)]
        }
        
        return pd.DataFrame(data)
    
    def test_no_drift_detection(self):
        """Test that similar data doesn't trigger drift."""
        # Generate similar data to reference
        current_data = self._generate_sample_data(200, drift=False)
        
        result = self.drift_detector.detect_drift(
            current_data=current_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
        
        print(f"✅ No drift test: drift_score={result.drift_score:.3f}, detected={result.overall_drift_detected}")
        
        # Should not detect drift
        self.assertFalse(result.overall_drift_detected)
        self.assertLess(result.drift_score, 0.3)
    
    def test_drift_detection(self):
        """Test that drifted data triggers drift detection."""
        # Generate drifted data
        current_data = self._generate_sample_data(200, drift=True)
        
        result = self.drift_detector.detect_drift(
            current_data=current_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
        
        print(f"✅ Drift test: drift_score={result.drift_score:.3f}, detected={result.overall_drift_detected}")
        
        # Should detect drift
        self.assertGreater(result.drift_score, 0.1)  # Some drift should be detected
        self.assertGreater(result.num_drifted_features, 0)
    
    def test_feature_level_drift(self):
        """Test feature-level drift detection."""
        current_data = self._generate_sample_data(200, drift=True)
        
        result = self.drift_detector.detect_drift(
            current_data=current_data,
            categorical_columns=['category', 'priority'],
            text_columns=['description'],
            model_name='test_model'
        )
        
        # Check feature-level results
        feature_names = [fd.feature_name for fd in result.feature_drifts]
        drift_scores = [fd.drift_score for fd in result.feature_drifts]
        
        print(f"✅ Feature drift: features={feature_names}, scores={[f'{s:.3f}' for s in drift_scores]}")
        
        self.assertGreater(len(result.feature_drifts), 0)
        self.assertTrue(any(fd.drift_detected for fd in result.feature_drifts))


class TestMetricsLogger(unittest.TestCase):
    """Test the metrics logging system."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.monitoring.metrics_logger import MetricsLogger
        from src.monitoring.performance_monitor import PerformanceMetrics
        from src.monitoring.drift_detector import DriftResult, FeatureDrift
        
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MetricsLogger(
            log_dir=self.temp_dir,
            max_log_entries=1000
        )
        
        # Create sample metrics
        self.sample_metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            prediction_count=100,
            avg_prediction_time=0.05
        )
        
        self.sample_drift = DriftResult(
            overall_drift_detected=True,
            drift_score=0.4,
            num_drifted_features=2,
            total_features=3,
            feature_drifts=[],
            max_drift_score=0.4,
            avg_drift_score=0.3
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.logger.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_logging(self):
        """Test performance metrics logging."""
        self.logger.log_performance_metrics("test_model", self.sample_metrics)
        
        # Check if logged
        summary = self.logger.get_metrics_summary("test_model", hours=1)
        
        print(f"✅ Performance logging: entries={summary['total_entries']}")
        self.assertGreater(summary['total_entries'], 0)
        self.assertIn("test_model", summary['models'])
    
    def test_drift_logging(self):
        """Test drift metrics logging."""
        self.logger.log_drift_metrics("test_model", self.sample_drift)
        
        # Check if logged
        summary = self.logger.get_metrics_summary("test_model", hours=1)
        
        print(f"✅ Drift logging: entries={summary['total_entries']}")
        self.assertGreater(summary['total_entries'], 0)
    
    def test_system_logging(self):
        """Test system metrics logging."""
        self.logger.log_system_metrics(
            model_name="test_model",
            cpu_usage=45.5,
            memory_usage=67.2,
            prediction_latency=50.0,
            throughput=200.0
        )
        
        summary = self.logger.get_metrics_summary("test_model", hours=1)
        
        print(f"✅ System logging: entries={summary['total_entries']}")
        self.assertGreater(summary['total_entries'], 0)
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        # Log some metrics
        self.logger.log_performance_metrics("test_model", self.sample_metrics)
        self.logger.log_drift_metrics("test_model", self.sample_drift)
        
        # Export metrics
        export_path = Path(self.temp_dir) / "test_export.json"
        success = self.logger.export_metrics(str(export_path), "test_model")
        
        print(f"✅ Metrics export: success={success}, file_exists={export_path.exists()}")
        self.assertTrue(success)
        self.assertTrue(export_path.exists())


class TestAlertManager(unittest.TestCase):
    """Test the alert management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.monitoring.alert_manager import (
            AlertManager, AlertRule, AlertSeverity, AlertType
        )
        from src.monitoring.performance_monitor import PerformanceMetrics
        
        self.alert_manager = AlertManager()
        
        # Add test alert rule
        rule = AlertRule(
            name="test_accuracy_rule",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            condition="accuracy < 0.8",
            threshold_value=0.8,
            comparison_operator="lt",
            consecutive_violations=1
        )
        self.alert_manager.add_alert_rule(rule)
        
        # Sample metrics
        self.good_metrics = PerformanceMetrics(
            accuracy=0.85, precision=0.83, recall=0.87, f1_score=0.85,
            prediction_count=100, avg_prediction_time=0.05
        )
        
        self.poor_metrics = PerformanceMetrics(
            accuracy=0.75, precision=0.73, recall=0.77, f1_score=0.75,
            prediction_count=100, avg_prediction_time=0.05
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.alert_manager.close()
    
    def test_alert_rule_creation(self):
        """Test alert rule creation and management."""
        rules = self.alert_manager.alert_rules
        
        print(f"✅ Alert rules: {len(rules)} rules created")
        self.assertGreater(len(rules), 0)
        self.assertIn("test_accuracy_rule", rules)
    
    def test_performance_alert_trigger(self):
        """Test performance alert triggering."""
        # This should not trigger an alert
        self.alert_manager.check_performance_alerts("test_model", self.good_metrics)
        active_alerts_good = self.alert_manager.get_active_alerts()
        
        # This should trigger an alert
        self.alert_manager.check_performance_alerts("test_model", self.poor_metrics)
        active_alerts_poor = self.alert_manager.get_active_alerts()
        
        print(f"✅ Performance alerts: good={len(active_alerts_good)}, poor={len(active_alerts_poor)}")
        self.assertLessEqual(len(active_alerts_good), len(active_alerts_poor))
    
    def test_custom_alert(self):
        """Test custom alert creation."""
        from src.monitoring.alert_manager import AlertSeverity
        
        alert_id = self.alert_manager.create_custom_alert(
            model_name="test_model",
            title="Test Custom Alert",
            message="This is a test alert",
            severity=AlertSeverity.MEDIUM
        )
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        print(f"✅ Custom alert: id={alert_id}, active_count={len(active_alerts)}")
        self.assertIsNotNone(alert_id)
        self.assertGreater(len(active_alerts), 0)
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment and resolution."""
        from src.monitoring.alert_manager import AlertSeverity
        
        # Create an alert
        alert_id = self.alert_manager.create_custom_alert(
            model_name="test_model",
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.LOW
        )
        
        # Acknowledge it
        ack_success = self.alert_manager.acknowledge_alert(alert_id, "Test acknowledgment")
        
        # Resolve it
        resolve_success = self.alert_manager.resolve_alert(alert_id, "Test resolution")
        
        print(f"✅ Alert lifecycle: ack={ack_success}, resolve={resolve_success}")
        self.assertTrue(ack_success)
        self.assertTrue(resolve_success)


def run_integration_test():
    """Run integration test with all components."""
    print("\n" + "="*60)
    print("🔗 INTEGRATION TEST")
    print("="*60)
    
    try:
        from src.monitoring.performance_monitor import ModelPerformanceMonitor, PerformanceMetrics
        from src.monitoring.drift_detector import DataDriftDetector
        from src.monitoring.metrics_logger import MetricsLogger
        from src.monitoring.alert_manager import (
            AlertManager, create_default_performance_rules, create_default_drift_rules
        )
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Initialize all components
        performance_monitor = ModelPerformanceMonitor("integration_test")
        drift_detector = DataDriftDetector()
        metrics_logger = MetricsLogger(log_dir=temp_dir)
        alert_manager = AlertManager()
        
        # Add default rules
        for rule in create_default_performance_rules():
            alert_manager.add_alert_rule(rule)
        
        for rule in create_default_drift_rules():
            alert_manager.add_alert_rule(rule)
        
        # Generate test data
        np.random.seed(42)
        n_samples = 500
        
        # Reference data
        ref_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'priority': np.random.choice(['low', 'medium', 'high'], n_samples),
            'description': [f"Sample text {i}" for i in range(n_samples)]
        })
        
        drift_detector.set_reference_data(
            ref_data, ['category', 'priority'], ['description'], 'integration_test'
        )
        
        # Simulate poor performance
        y_true = np.random.choice(['A', 'B', 'C'], 200)
        y_pred = np.random.choice(['A', 'B', 'C'], 200)  # Random predictions
        
        performance_monitor.log_batch_predictions(y_true, y_pred, [0.05] * 200)
        current_performance = performance_monitor.get_current_performance()
        
        # Log performance metrics
        metrics_logger.log_performance_metrics('integration_test', current_performance)
        
        # Check performance alerts
        alert_manager.check_performance_alerts('integration_test', current_performance)
        
        # Simulate drift
        drift_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 200, p=[0.7, 0.2, 0.1]),  # Different distribution
            'priority': np.random.choice(['low', 'medium', 'high'], 200),
            'description': [f"Different text {i}" for i in range(200)]
        })
        
        drift_result = drift_detector.detect_drift(
            drift_data, ['category', 'priority'], ['description'], 'integration_test'
        )
        
        # Log drift metrics
        metrics_logger.log_drift_metrics('integration_test', drift_result)
        
        # Check drift alerts
        alert_manager.check_drift_alerts('integration_test', drift_result)
        
        # Get summary
        alert_summary = alert_manager.get_alert_summary()
        metrics_summary = metrics_logger.get_metrics_summary(hours=1)
        
        print(f"✅ Integration test completed:")
        print(f"   Performance accuracy: {current_performance.accuracy:.3f}")
        print(f"   Drift detected: {drift_result.overall_drift_detected}")
        print(f"   Drift score: {drift_result.drift_score:.3f}")
        print(f"   Active alerts: {alert_summary['total_active_alerts']}")
        print(f"   Logged metrics: {metrics_summary['total_entries']}")
        
        # Cleanup
        metrics_logger.close()
        alert_manager.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Starting Monitoring System Tests")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPerformanceMonitor))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDriftDetector))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMetricsLogger))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAlertManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(test_suite)
    
    # Run integration test
    integration_success = run_integration_test()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Unit Tests: {successes}/{total_tests} passed")
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    if failures > 0:
        print(f"\n❌ Failures: {failures}")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if errors > 0:
        print(f"\n❌ Errors: {errors}")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    overall_success = (failures == 0 and errors == 0 and integration_success)
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)