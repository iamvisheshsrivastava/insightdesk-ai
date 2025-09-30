# scripts/test_anomaly_quick.py

"""
Quick test script for anomaly detection functionality.
Tests if the module works with current dependencies.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from anomaly.anomaly_models import (
            AnomalyRecord, AnomalyType, AnomalySeverity, AnomalyThresholds,
            create_anomaly_record
        )
        print("✅ Anomaly models imported successfully")
        
        from anomaly.anomaly_detector import AnomalyDetector
        print("✅ Anomaly detector imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic anomaly detection functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from anomaly.anomaly_detector import AnomalyDetector
        from anomaly.anomaly_models import AnomalyThresholds
        
        # Create test data
        test_tickets = [
            {
                "ticket_id": "T001",
                "category": "auth",
                "created_at": "2024-01-01 10:00:00",
                "customer_sentiment": "neutral",
                "priority": "medium",
                "subject": "Login issue",
                "description": "Cannot log into the system",
                "error_logs": "Auth error",
                "previous_tickets": 2,
                "account_age_days": 365,
                "resolution_time_hours": 24,
                "product": "web_app"
            },
            {
                "ticket_id": "T002",
                "category": "auth",
                "created_at": "2024-01-01 11:00:00",
                "customer_sentiment": "frustrated",
                "priority": "high",
                "subject": "Authentication failure",
                "description": "Auth system is down",
                "error_logs": "Auth service timeout",
                "previous_tickets": 1,
                "account_age_days": 180,
                "resolution_time_hours": 12,
                "product": "web_app"
            }
        ]
        
        # Initialize detector
        detector = AnomalyDetector()
        print("✅ Anomaly detector initialized")
        
        # Test individual detectors
        print("🔍 Testing volume detector...")
        volume_anomalies = detector.volume_detector.detect(test_tickets)
        print(f"   Found {len(volume_anomalies)} volume anomalies")
        
        print("🔍 Testing sentiment detector...")
        sentiment_anomalies = detector.sentiment_detector.detect(test_tickets)
        print(f"   Found {len(sentiment_anomalies)} sentiment anomalies")
        
        print("🔍 Testing new issue detector...")
        new_issue_anomalies = detector.new_issue_detector.detect(test_tickets)
        print(f"   Found {len(new_issue_anomalies)} new issue anomalies")
        
        print("🔍 Testing outlier detector...")
        outlier_anomalies = detector.outlier_detector.detect(test_tickets)
        print(f"   Found {len(outlier_anomalies)} outlier anomalies")
        
        # Test comprehensive detection
        print("🔍 Testing comprehensive detection...")
        result = detector.detect_all_anomalies(test_tickets)
        print(f"   Total anomalies: {result.total_anomalies}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Tickets analyzed: {result.tickets_analyzed}")
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_models():
    """Test anomaly data models."""
    print("\n📋 Testing data models...")
    
    try:
        from anomaly.anomaly_models import (
            create_anomaly_record, AnomalyType, AnomalySeverity,
            VolumeSpike, SentimentShift, determine_severity
        )
        
        # Test anomaly record creation
        anomaly = create_anomaly_record(
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            description="Test volume spike",
            score=0.8,
            severity=AnomalySeverity.HIGH,
            category="test"
        )
        print("✅ Anomaly record created successfully")
        print(f"   ID: {anomaly.id}")
        print(f"   Type: {anomaly.type.value}")
        print(f"   Severity: {anomaly.severity.value}")
        
        # Test severity determination
        test_scores = [0.1, 0.4, 0.7, 0.9]
        for score in test_scores:
            severity = determine_severity(score)
            print(f"   Score {score} -> Severity: {severity.value}")
        
        # Test VolumeSpike model
        spike = VolumeSpike(
            category="test",
            date=datetime.utcnow(),
            actual_count=50,
            expected_count=10.5,
            spike_ratio=4.76,
            historical_mean=10.5,
            historical_std=2.1
        )
        print("✅ VolumeSpike model created successfully")
        
        # Test SentimentShift model
        shift = SentimentShift(
            category="test",
            current_sentiment=-0.5,
            baseline_sentiment=0.2,
            shift_magnitude=0.7,
            affected_period={
                "start": datetime.utcnow() - timedelta(days=7),
                "end": datetime.utcnow()
            },
            sample_size=100
        )
        print("✅ SentimentShift model created successfully")
        
        print("✅ Data model tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Data model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if optional dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    # Check scikit-learn
    try:
        import sklearn
        print(f"✅ scikit-learn available (version: {sklearn.__version__})")
        sklearn_available = True
    except ImportError:
        print("⚠️  scikit-learn not available (outlier detection and new issue detection will be limited)")
        sklearn_available = False
    
    # Check pandas
    try:
        import pandas
        print(f"✅ pandas available (version: {pandas.__version__})")
        pandas_available = True
    except ImportError:
        print("❌ pandas not available (required for anomaly detection)")
        pandas_available = False
    
    # Check numpy
    try:
        import numpy
        print(f"✅ numpy available (version: {numpy.__version__})")
        numpy_available = True
    except ImportError:
        print("❌ numpy not available (required for anomaly detection)")
        numpy_available = False
    
    return {
        "sklearn": sklearn_available,
        "pandas": pandas_available,
        "numpy": numpy_available
    }

def main():
    """Run all tests."""
    print("🚀 Starting Anomaly Detection Quick Test\n")
    
    # Check dependencies
    deps = check_dependencies()
    
    if not deps["pandas"] or not deps["numpy"]:
        print("\n❌ Critical dependencies missing. Cannot proceed with tests.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot proceed.")
        return False
    
    # Test data models
    if not test_data_models():
        print("\n❌ Data model tests failed.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed.")
        return False
    
    print("\n🎉 All tests passed! Anomaly detection module is working correctly.")
    
    if not deps["sklearn"]:
        print("\n⚠️  Note: Some advanced features (outlier detection, new issue detection)")
        print("   may be limited without scikit-learn. Consider installing it for full functionality.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)