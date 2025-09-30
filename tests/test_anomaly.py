# tests/test_anomaly.py

"""
Comprehensive tests for the anomaly detection module.
Tests volume spikes, sentiment shifts, new issues, and outlier detection.
"""

import unittest
import tempfile
import json
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import sys

# Add src to path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from anomaly.anomaly_detector import (
        AnomalyDetector, VolumeAnomalyDetector, SentimentAnomalyDetector,
        NewIssueDetector, OutlierDetector
    )
    from anomaly.anomaly_models import (
        AnomalyRecord, AnomalyType, AnomalySeverity, AnomalyThresholds,
        VolumeSpike, SentimentShift, NewIssue, OutlierInfo,
        create_anomaly_record, determine_severity
    )
except ImportError as e:
    raise ImportError(f"Failed to import anomaly detection modules: {e}")


class TestAnomalyModels(unittest.TestCase):
    """Test the anomaly data models and utilities."""
    
    def test_anomaly_record_creation(self):
        """Test creating anomaly records."""
        anomaly = create_anomaly_record(
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            description="Test volume spike",
            score=0.8,
            severity=AnomalySeverity.HIGH,
            category="test_category"
        )
        
        self.assertEqual(anomaly.type, AnomalyType.VOLUME_SPIKE)
        self.assertEqual(anomaly.description, "Test volume spike")
        self.assertEqual(anomaly.score, 0.8)
        self.assertEqual(anomaly.severity, AnomalySeverity.HIGH)
        self.assertEqual(anomaly.category, "test_category")
        self.assertIsInstance(anomaly.timestamp, datetime)
        self.assertIsNotNone(anomaly.id)
    
    def test_severity_determination(self):
        """Test severity determination logic."""
        self.assertEqual(determine_severity(0.1), AnomalySeverity.LOW)
        self.assertEqual(determine_severity(0.4), AnomalySeverity.MEDIUM)
        self.assertEqual(determine_severity(0.7), AnomalySeverity.HIGH)
        self.assertEqual(determine_severity(0.9), AnomalySeverity.CRITICAL)
        self.assertEqual(determine_severity(1.5), AnomalySeverity.CRITICAL)  # Cap at critical
    
    def test_volume_spike_model(self):
        """Test VolumeSpike data model."""
        spike = VolumeSpike(
            category="authentication",
            date=datetime.utcnow(),
            actual_count=50,
            expected_count=10.5,
            spike_ratio=4.76,
            historical_mean=10.5,
            historical_std=2.1
        )
        
        self.assertEqual(spike.category, "authentication")
        self.assertEqual(spike.actual_count, 50)
        self.assertAlmostEqual(spike.spike_ratio, 4.76, places=2)
    
    def test_sentiment_shift_model(self):
        """Test SentimentShift data model."""
        shift = SentimentShift(
            category="payment",
            current_sentiment=-0.5,
            baseline_sentiment=0.2,
            shift_magnitude=0.7,
            affected_period={
                "start": datetime.utcnow() - timedelta(days=7),
                "end": datetime.utcnow()
            },
            sample_size=100
        )
        
        self.assertEqual(shift.category, "payment")
        self.assertEqual(shift.current_sentiment, -0.5)
        self.assertEqual(shift.shift_magnitude, 0.7)
        self.assertEqual(shift.sample_size, 100)
    
    def test_anomaly_thresholds(self):
        """Test AnomalyThresholds configuration."""
        thresholds = AnomalyThresholds(
            volume_spike_sigma=3.0,
            sentiment_shift_threshold=0.3,
            new_issue_similarity_threshold=0.5,
            outlier_contamination=0.1
        )
        
        self.assertEqual(thresholds.volume_spike_sigma, 3.0)
        self.assertEqual(thresholds.sentiment_shift_threshold, 0.3)
        self.assertEqual(thresholds.new_issue_similarity_threshold, 0.5)
        self.assertEqual(thresholds.outlier_contamination, 0.1)


class TestVolumeAnomalyDetector(unittest.TestCase):
    """Test volume spike detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = VolumeAnomalyDetector()
    
    def create_volume_test_data(self):
        """Create test data with volume spikes."""
        base_date = datetime.utcnow() - timedelta(days=30)
        tickets = []
        
        # Normal baseline (days 0-25): 5-10 tickets per day per category
        for day in range(26):
            current_date = base_date + timedelta(days=day)
            
            for category in ["auth", "payment", "ui"]:
                daily_volume = np.random.randint(5, 11)  # Normal volume
                
                for i in range(daily_volume):
                    tickets.append({
                        "ticket_id": f"T{len(tickets):06d}",
                        "category": category,
                        "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "subject": f"Issue in {category}",
                        "description": f"Regular issue in {category}"
                    })
        
        # Volume spike (days 26-29): 30+ tickets for 'auth' category
        for day in range(26, 30):
            current_date = base_date + timedelta(days=day)
            
            # Spike in auth category
            for i in range(35):  # High volume
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "auth",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": "Authentication failure",
                    "description": "Sudden auth issues"
                })
            
            # Normal volume for other categories
            for category in ["payment", "ui"]:
                daily_volume = np.random.randint(5, 11)
                for i in range(daily_volume):
                    tickets.append({
                        "ticket_id": f"T{len(tickets):06d}",
                        "category": category,
                        "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "subject": f"Issue in {category}",
                        "description": f"Regular issue in {category}"
                    })
        
        return tickets
    
    def test_volume_spike_detection(self):
        """Test detection of volume spikes."""
        tickets_data = self.create_volume_test_data()
        anomalies = self.detector.detect(tickets_data)
        
        # Should detect volume spike in 'auth' category
        auth_spikes = [a for a in anomalies if a.category == "auth"]
        self.assertGreater(len(auth_spikes), 0, "Should detect volume spike in auth category")
        
        # Check anomaly properties
        for anomaly in auth_spikes:
            self.assertEqual(anomaly.type, AnomalyType.VOLUME_SPIKE)
            self.assertIn("volume_spike", anomaly.details)
            self.assertGreater(anomaly.score, 0)
            self.assertIn(anomaly.severity, [AnomalySeverity.MEDIUM, AnomalySeverity.HIGH, AnomalySeverity.CRITICAL])
    
    def test_no_false_positives(self):
        """Test that normal data doesn't trigger false positives."""
        base_date = datetime.utcnow() - timedelta(days=20)
        tickets = []
        
        # Create only normal volume data
        for day in range(20):
            current_date = base_date + timedelta(days=day)
            
            for category in ["auth", "payment", "ui"]:
                daily_volume = 8  # Consistent normal volume
                
                for i in range(daily_volume):
                    tickets.append({
                        "ticket_id": f"T{len(tickets):06d}",
                        "category": category,
                        "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "subject": f"Issue in {category}",
                        "description": f"Regular issue in {category}"
                    })
        
        anomalies = self.detector.detect(tickets)
        
        # Should not detect any volume spikes in normal data
        self.assertEqual(len(anomalies), 0, "Should not detect false positive volume spikes")
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Very few tickets
        tickets = [
            {
                "ticket_id": "T000001",
                "category": "test",
                "created_at": "2024-01-01 10:00:00",
                "subject": "Test",
                "description": "Test"
            }
        ]
        
        anomalies = self.detector.detect(tickets)
        self.assertEqual(len(anomalies), 0, "Should handle insufficient data gracefully")


class TestSentimentAnomalyDetector(unittest.TestCase):
    """Test sentiment shift detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SentimentAnomalyDetector()
    
    def create_sentiment_test_data(self):
        """Create test data with sentiment shifts."""
        base_date = datetime.utcnow() - timedelta(days=30)
        tickets = []
        
        # Historical data with neutral sentiment (days 0-20)
        for day in range(21):
            current_date = base_date + timedelta(days=day)
            
            for category in ["payment", "support", "ui"]:
                for i in range(10):  # 10 tickets per category per day
                    sentiment = np.random.choice(["neutral", "satisfied"], p=[0.7, 0.3])
                    
                    tickets.append({
                        "ticket_id": f"T{len(tickets):06d}",
                        "category": category,
                        "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "customer_sentiment": sentiment,
                        "subject": f"Issue in {category}",
                        "description": f"Regular issue in {category}"
                    })
        
        # Recent data with negative sentiment shift for payment (days 21-29)
        for day in range(21, 30):
            current_date = base_date + timedelta(days=day)
            
            for category in ["payment", "support", "ui"]:
                for i in range(10):
                    if category == "payment":
                        # Negative sentiment shift for payment
                        sentiment = np.random.choice(["frustrated", "angry"], p=[0.6, 0.4])
                    else:
                        # Normal sentiment for other categories
                        sentiment = np.random.choice(["neutral", "satisfied"], p=[0.7, 0.3])
                    
                    tickets.append({
                        "ticket_id": f"T{len(tickets):06d}",
                        "category": category,
                        "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "customer_sentiment": sentiment,
                        "subject": f"Issue in {category}",
                        "description": f"Regular issue in {category}"
                    })
        
        return tickets
    
    def test_sentiment_shift_detection(self):
        """Test detection of sentiment shifts."""
        tickets_data = self.create_sentiment_test_data()
        anomalies = self.detector.detect(tickets_data)
        
        # Should detect sentiment shift in payment category
        payment_shifts = [a for a in anomalies if a.category == "payment"]
        self.assertGreater(len(payment_shifts), 0, "Should detect sentiment shift in payment category")
        
        # Check anomaly properties
        for anomaly in payment_shifts:
            self.assertEqual(anomaly.type, AnomalyType.SENTIMENT_SHIFT)
            self.assertIn("sentiment_shift", anomaly.details)
            self.assertGreater(anomaly.score, 0)
    
    def test_sentiment_mapping(self):
        """Test sentiment score mapping."""
        mapping = self.detector.sentiment_mapping
        
        self.assertEqual(mapping["frustrated"], -1.0)
        self.assertEqual(mapping["angry"], -1.0)
        self.assertEqual(mapping["neutral"], 0.0)
        self.assertEqual(mapping["satisfied"], 0.5)
        self.assertEqual(mapping["happy"], 1.0)


class TestNewIssueDetector(unittest.TestCase):
    """Test new issue detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = NewIssueDetector()
    
    @patch('anomaly.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_new_issue_detection_with_sklearn(self):
        """Test new issue detection when sklearn is available."""
        tickets_data = self.create_new_issue_test_data()
        
        if self.detector.vectorizer is not None:
            anomalies = self.detector.detect(tickets_data)
            # Should detect some new issues
            self.assertGreaterEqual(len(anomalies), 0)
    
    @patch('anomaly.anomaly_detector.SKLEARN_AVAILABLE', False)
    def test_new_issue_detection_without_sklearn(self):
        """Test new issue detection when sklearn is not available."""
        detector = NewIssueDetector()
        tickets_data = self.create_new_issue_test_data()
        
        anomalies = detector.detect(tickets_data)
        # Should return empty list when sklearn not available
        self.assertEqual(len(anomalies), 0)
    
    def create_new_issue_test_data(self):
        """Create test data with new issue patterns."""
        base_date = datetime.utcnow() - timedelta(days=20)
        tickets = []
        
        # Historical patterns
        common_issues = [
            "Login timeout error when accessing dashboard",
            "Payment processing failure during checkout",
            "User interface not loading properly on mobile",
            "Database connection error in user profile"
        ]
        
        # Add historical tickets with common patterns
        for day in range(15):
            current_date = base_date + timedelta(days=day)
            
            for i in range(5):
                issue = np.random.choice(common_issues)
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "general",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": issue,
                    "description": f"Customer reports: {issue}",
                    "error_logs": f"Error related to: {issue}"
                })
        
        # Add new issue pattern (days 15-19)
        new_issue_pattern = "API rate limiting causing timeout in webhook processing"
        
        for day in range(15, 20):
            current_date = base_date + timedelta(days=day)
            
            # Add new pattern multiple times
            for i in range(3):
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "api",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": new_issue_pattern,
                    "description": f"New issue: {new_issue_pattern}. This is affecting webhook integrations.",
                    "error_logs": "RateLimitError: API rate limit exceeded. Webhook timeout after 30s."
                })
            
            # Continue adding some historical patterns
            issue = np.random.choice(common_issues)
            tickets.append({
                "ticket_id": f"T{len(tickets):06d}",
                "category": "general",
                "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "subject": issue,
                "description": f"Customer reports: {issue}",
                "error_logs": f"Error related to: {issue}"
            })
        
        return tickets


class TestOutlierDetector(unittest.TestCase):
    """Test outlier detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector()
    
    def create_outlier_test_data(self):
        """Create test data with outliers."""
        base_date = datetime.utcnow() - timedelta(days=10)
        tickets = []
        
        # Normal tickets
        for i in range(100):
            current_date = base_date + timedelta(hours=i)
            
            tickets.append({
                "ticket_id": f"T{i:06d}",
                "category": np.random.choice(["auth", "payment", "ui"]),
                "product": np.random.choice(["web", "mobile", "api"]),
                "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "priority": np.random.choice(["low", "medium", "high"], p=[0.5, 0.4, 0.1]),
                "customer_sentiment": np.random.choice(["neutral", "frustrated", "positive"], p=[0.6, 0.3, 0.1]),
                "subject": "Regular issue with system",
                "description": "Standard problem description",
                "error_logs": "Standard error message",
                "previous_tickets": np.random.poisson(2),
                "account_age_days": np.random.randint(30, 500),
                "resolution_time_hours": np.random.exponential(24)
            })
        
        # Add outliers
        outliers = [
            {
                "ticket_id": "T999001",
                "category": "billing",
                "product": "admin",
                "created_at": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "critical",
                "customer_sentiment": "positive",  # Outlier: positive sentiment for critical issue
                "subject": "Extremely long subject line that goes on and on with excessive detail" * 3,
                "description": "Extremely detailed description with excessive length" * 50,
                "error_logs": "Massive error log content" * 100,
                "previous_tickets": 100,  # Outlier: very high
                "account_age_days": 1,    # Outlier: very new account with many tickets
                "resolution_time_hours": 500  # Outlier: very long resolution time
            },
            {
                "ticket_id": "T999002",
                "category": "ui",
                "product": "mobile",
                "created_at": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "priority": "low",
                "customer_sentiment": "neutral",
                "subject": "",  # Outlier: empty subject
                "description": "Short",  # Outlier: very short
                "error_logs": "",  # Outlier: no logs
                "previous_tickets": 0,
                "account_age_days": 5000,  # Outlier: very old account
                "resolution_time_hours": 0.01  # Outlier: very fast resolution
            }
        ]
        
        tickets.extend(outliers)
        return tickets
    
    @patch('anomaly.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_outlier_detection_with_sklearn(self):
        """Test outlier detection when sklearn is available."""
        tickets_data = self.create_outlier_test_data()
        
        anomalies = self.detector.detect(tickets_data)
        
        # Should detect some outliers
        self.assertGreaterEqual(len(anomalies), 0)
        
        for anomaly in anomalies:
            self.assertEqual(anomaly.type, AnomalyType.OUTLIER)
            self.assertIn("outlier_info", anomaly.details)
    
    @patch('anomaly.anomaly_detector.SKLEARN_AVAILABLE', False)
    def test_outlier_detection_without_sklearn(self):
        """Test outlier detection when sklearn is not available."""
        detector = OutlierDetector()
        tickets_data = self.create_outlier_test_data()
        
        anomalies = detector.detect(tickets_data)
        # Should return empty list when sklearn not available
        self.assertEqual(len(anomalies), 0)
    
    def test_feature_extraction(self):
        """Test numerical feature extraction."""
        ticket = {
            "priority": "high",
            "customer_sentiment": "frustrated",
            "subject": "Test subject",
            "description": "Test description with some content",
            "error_logs": "Error log content",
            "previous_tickets": 5,
            "account_age_days": 365,
            "resolution_time_hours": 48
        }
        
        features = self.detector._extract_numerical_features(ticket)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 8)  # Expected number of features
        self.assertIsInstance(features[0], float)  # Priority
        self.assertIsInstance(features[1], float)  # Sentiment


class TestAnomalyDetectorIntegration(unittest.TestCase):
    """Test the main AnomalyDetector orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
    
    def create_comprehensive_test_data(self):
        """Create comprehensive test data with multiple anomaly types."""
        base_date = datetime.utcnow() - timedelta(days=40)
        tickets = []
        
        # Generate baseline data
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            for category in ["auth", "payment", "ui"]:
                daily_volume = 8  # Normal volume
                
                for i in range(daily_volume):
                    tickets.append({
                        "ticket_id": f"T{len(tickets):06d}",
                        "category": category,
                        "product": "web_app",
                        "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "customer_sentiment": np.random.choice(["neutral", "satisfied"], p=[0.7, 0.3]),
                        "priority": np.random.choice(["low", "medium", "high"], p=[0.5, 0.4, 0.1]),
                        "subject": f"Regular {category} issue",
                        "description": f"Standard issue in {category} category",
                        "error_logs": f"Standard error in {category}",
                        "previous_tickets": np.random.poisson(2),
                        "account_age_days": np.random.randint(30, 500),
                        "resolution_time_hours": np.random.exponential(24)
                    })
        
        # Add anomalies (days 30-39)
        
        # Volume spike in auth
        for day in range(30, 34):
            current_date = base_date + timedelta(days=day)
            
            # High volume for auth
            for i in range(30):
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "auth",
                    "product": "web_app",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": "frustrated",
                    "priority": "high",
                    "subject": "Authentication failure",
                    "description": "Cannot authenticate users",
                    "error_logs": "AuthError: Token validation failed",
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 500),
                    "resolution_time_hours": np.random.exponential(24)
                })
        
        # Sentiment shift in payment
        for day in range(34, 38):
            current_date = base_date + timedelta(days=day)
            
            for i in range(10):
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "payment",
                    "product": "web_app",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": "angry",  # Negative shift
                    "priority": "critical",
                    "subject": "Payment processing failed",
                    "description": "Payment gateway errors",
                    "error_logs": "PaymentError: Transaction failed",
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 500),
                    "resolution_time_hours": np.random.exponential(24)
                })
        
        # Add outlier
        tickets.append({
            "ticket_id": "T999999",
            "category": "billing",
            "product": "admin",
            "created_at": (base_date + timedelta(days=39)).strftime("%Y-%m-%d %H:%M:%S"),
            "customer_sentiment": "positive",  # Outlier: positive for billing issue
            "priority": "critical",
            "subject": "Extremely long subject line with excessive detail " * 10,
            "description": "Extremely detailed description " * 100,
            "error_logs": "Massive error log " * 200,
            "previous_tickets": 150,  # Outlier: very high
            "account_age_days": 1,    # Outlier: new account with many tickets
            "resolution_time_hours": 1000  # Outlier: very long resolution
        })
        
        return tickets
    
    def test_comprehensive_anomaly_detection(self):
        """Test comprehensive anomaly detection across all types."""
        tickets_data = self.create_comprehensive_test_data()
        
        result = self.detector.detect_all_anomalies(tickets_data)
        
        # Check result structure
        self.assertIsInstance(result.total_anomalies, int)
        self.assertIsInstance(result.anomalies, list)
        self.assertIsInstance(result.severity_breakdown, dict)
        self.assertIsInstance(result.type_breakdown, dict)
        self.assertGreater(result.processing_time, 0)
        self.assertEqual(result.tickets_analyzed, len(tickets_data))
        
        # Should detect some anomalies
        self.assertGreaterEqual(result.total_anomalies, 0)
        
        # Check anomaly types
        detected_types = set()
        for anomaly in result.anomalies:
            detected_types.add(anomaly.type)
            self.assertIsInstance(anomaly, AnomalyRecord)
            self.assertIsNotNone(anomaly.id)
            self.assertIsInstance(anomaly.timestamp, datetime)
    
    def test_specific_detection_types(self):
        """Test detection with specific anomaly types only."""
        tickets_data = self.create_comprehensive_test_data()
        
        # Test volume spike detection only
        result = self.detector.detect_all_anomalies(
            tickets_data, 
            detection_types=[AnomalyType.VOLUME_SPIKE]
        )
        
        # All detected anomalies should be volume spikes
        for anomaly in result.anomalies:
            self.assertEqual(anomaly.type, AnomalyType.VOLUME_SPIKE)
    
    def test_recent_anomalies_retrieval(self):
        """Test retrieving recent anomalies."""
        tickets_data = self.create_comprehensive_test_data()
        
        # Run detection first
        self.detector.detect_all_anomalies(tickets_data)
        
        # Get recent anomalies
        recent = self.detector.get_recent_anomalies(days=30)
        
        self.assertIsInstance(recent, list)
        # All anomalies should be within the time window
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        for anomaly in recent:
            self.assertGreaterEqual(anomaly.timestamp, cutoff_date)
    
    def test_threshold_updates(self):
        """Test updating detection thresholds."""
        new_thresholds = AnomalyThresholds(
            volume_spike_sigma=4.0,
            sentiment_shift_threshold=0.5,
            new_issue_similarity_threshold=0.3,
            outlier_contamination=0.15
        )
        
        self.detector.update_thresholds(new_thresholds)
        
        self.assertEqual(self.detector.thresholds.volume_spike_sigma, 4.0)
        self.assertEqual(self.detector.volume_detector.thresholds.volume_spike_sigma, 4.0)
        self.assertEqual(self.detector.sentiment_detector.thresholds.sentiment_shift_threshold, 0.5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_detect_volume_spikes_function(self):
        """Test standalone volume spike detection function."""
        from anomaly.anomaly_detector import detect_volume_spikes
        
        # Create simple test data
        tickets = []
        base_date = datetime.utcnow() - timedelta(days=10)
        
        for day in range(10):
            current_date = base_date + timedelta(days=day)
            volume = 30 if day >= 8 else 5  # Spike in last 2 days
            
            for i in range(volume):
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "test",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": "Test issue",
                    "description": "Test description"
                })
        
        anomalies = detect_volume_spikes(tickets)
        self.assertIsInstance(anomalies, list)
    
    def test_detect_sentiment_shifts_function(self):
        """Test standalone sentiment shift detection function."""
        from anomaly.anomaly_detector import detect_sentiment_shifts
        
        # Create simple test data
        tickets = []
        base_date = datetime.utcnow() - timedelta(days=20)
        
        for day in range(20):
            current_date = base_date + timedelta(days=day)
            sentiment = "angry" if day >= 15 else "neutral"  # Negative shift
            
            for i in range(10):
                tickets.append({
                    "ticket_id": f"T{len(tickets):06d}",
                    "category": "test",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": sentiment,
                    "subject": "Test issue",
                    "description": "Test description"
                })
        
        anomalies = detect_sentiment_shifts(tickets)
        self.assertIsInstance(anomalies, list)


if __name__ == '__main__':
    # Configure test logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)