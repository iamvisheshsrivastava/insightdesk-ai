# scripts/test_feedback_quick.py

"""
Quick test script for the Feedback Loop system.

This script performs basic functionality tests to ensure
the feedback system is working correctly.
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.feedback.feedback_manager import FeedbackManager
from src.feedback.feedback_models import FeedbackType, FeedbackSeverity, PredictionQuality


def test_feedback_storage():
    """Test basic feedback storage functionality."""
    print("🧪 Testing Feedback Storage...")
    
    # Test JSON storage
    print("  📁 Testing JSON storage...")
    json_manager = FeedbackManager(
        storage_type="json",
        storage_config={"storage_dir": "test_feedback_json"}
    )
    
    # Test correction recording
    correction_id = json_manager.record_agent_correction(
        ticket_id="TEST-001",
        agent_id="test_agent",
        original_prediction="technical_issue",
        original_confidence=0.85,
        model_type="xgboost",
        corrected_label="billing_inquiry",
        correction_reason="Test correction",
        prediction_quality=PredictionQuality.POOR,
        severity=FeedbackSeverity.HIGH
    )
    
    if correction_id:
        print(f"    ✅ Correction recorded: {correction_id[:8]}...")
    else:
        print(f"    ❌ Failed to record correction")
        return False
    
    # Test feedback recording
    feedback_id = json_manager.record_customer_feedback(
        ticket_id="TEST-001",
        customer_id="test_customer",
        rating=4,
        comments="Good service",
        ai_suggestions_used=True,
        ai_suggestions_helpful=True
    )
    
    if feedback_id:
        print(f"    ✅ Customer feedback recorded: {feedback_id[:8]}...")
    else:
        print(f"    ❌ Failed to record customer feedback")
        return False
    
    # Test SQLite storage
    print("  💾 Testing SQLite storage...")
    sqlite_manager = FeedbackManager(
        storage_type="sqlite",
        storage_config={"db_path": "test_feedback.db"}
    )
    
    # Test correction recording
    correction_id = sqlite_manager.record_agent_correction(
        ticket_id="TEST-002",
        agent_id="test_agent",
        original_prediction="login_issue",
        original_confidence=0.75,
        model_type="tensorflow",
        corrected_label="password_reset",
        correction_reason="Test SQLite correction",
        prediction_quality=PredictionQuality.ACCEPTABLE,
        severity=FeedbackSeverity.MEDIUM
    )
    
    if correction_id:
        print(f"    ✅ SQLite correction recorded: {correction_id[:8]}...")
    else:
        print(f"    ❌ Failed to record SQLite correction")
        return False
    
    # Test feedback recording
    feedback_id = sqlite_manager.record_customer_feedback(
        ticket_id="TEST-002",
        customer_id="test_customer_2",
        rating=3,
        comments="Average service",
        ai_suggestions_used=False
    )
    
    if feedback_id:
        print(f"    ✅ SQLite customer feedback recorded: {feedback_id[:8]}...")
    else:
        print(f"    ❌ Failed to record SQLite customer feedback")
        return False
    
    print("  ✅ Storage tests passed")
    return True


def test_feedback_retrieval():
    """Test feedback data retrieval."""
    print("🔍 Testing Feedback Retrieval...")
    
    manager = FeedbackManager(
        storage_type="json",
        storage_config={"storage_dir": "test_feedback_json"}
    )
    
    # Test corrections retrieval
    corrections = manager.get_corrections(days_back=1)
    print(f"    📋 Retrieved {len(corrections)} corrections")
    
    if corrections:
        correction = corrections[0]
        print(f"    ✅ Sample correction: {correction.ticket_id} -> {correction.corrected_label}")
    
    # Test customer feedback retrieval
    feedback_list = manager.get_customer_feedback(days_back=1)
    print(f"    ⭐ Retrieved {len(feedback_list)} feedback entries")
    
    if feedback_list:
        feedback = feedback_list[0]
        print(f"    ✅ Sample feedback: {feedback.ticket_id} -> {feedback.rating}/5")
    
    # Test events retrieval
    events = manager.get_events(days_back=1)
    print(f"    📊 Retrieved {len(events)} events")
    
    print("  ✅ Retrieval tests passed")
    return True


def test_analytics():
    """Test analytics generation."""
    print("📈 Testing Analytics...")
    
    manager = FeedbackManager(
        storage_type="json",
        storage_config={"storage_dir": "test_feedback_json"}
    )
    
    # Generate summary
    summary = manager.generate_feedback_summary(days_back=1)
    
    print(f"    📊 Summary generated: {summary.summary_id[:8]}...")
    print(f"    📋 Total corrections: {summary.total_corrections}")
    print(f"    ⭐ Total feedback: {summary.total_customer_feedback}")
    print(f"    📈 Quality metrics: {len(summary.quality_metrics)} metrics")
    
    # Test retraining assessment
    retraining = summary.retraining_recommendations
    print(f"    🔄 Retraining needed: {retraining.get('needs_retraining', False)}")
    
    print("  ✅ Analytics tests passed")
    return True


def test_health_check():
    """Test system health monitoring."""
    print("🔧 Testing Health Check...")
    
    manager = FeedbackManager(
        storage_type="json",
        storage_config={"storage_dir": "test_feedback_json"}
    )
    
    health = manager.health_check()
    
    print(f"    📊 System status: {health.get('status', 'unknown')}")
    print(f"    💾 Storage operational: {health.get('storage_operational', False)}")
    
    if health.get('status') in ['healthy', 'degraded']:
        print("  ✅ Health check passed")
        return True
    else:
        print("  ❌ Health check failed")
        return False


def test_data_models():
    """Test data model serialization/deserialization."""
    print("📝 Testing Data Models...")
    
    from src.feedback.feedback_models import AgentCorrection, CustomerFeedback, FeedbackEvent
    
    # Test AgentCorrection
    correction = AgentCorrection(
        correction_id="test_correction",
        ticket_id="TEST-003",
        agent_id="test_agent",
        original_prediction="test_pred",
        original_confidence=0.8,
        model_type="test_model",
        corrected_label="test_correct",
        correction_reason="test reason",
        prediction_quality=PredictionQuality.GOOD,
        severity=FeedbackSeverity.LOW,
        correction_timestamp=datetime.now()
    )
    
    # Test serialization
    correction_dict = correction.to_dict()
    correction_restored = AgentCorrection.from_dict(correction_dict)
    
    if correction.correction_id == correction_restored.correction_id:
        print("    ✅ AgentCorrection serialization works")
    else:
        print("    ❌ AgentCorrection serialization failed")
        return False
    
    # Test CustomerFeedback
    feedback = CustomerFeedback(
        feedback_id="test_feedback",
        ticket_id="TEST-003",
        customer_id="test_customer",
        rating=5,
        feedback_timestamp=datetime.now()
    )
    
    feedback_dict = feedback.to_dict()
    feedback_restored = CustomerFeedback.from_dict(feedback_dict)
    
    if feedback.feedback_id == feedback_restored.feedback_id:
        print("    ✅ CustomerFeedback serialization works")
    else:
        print("    ❌ CustomerFeedback serialization failed")
        return False
    
    # Test FeedbackEvent
    event = FeedbackEvent(
        event_id="test_event",
        event_type=FeedbackType.AGENT_CORRECTION,
        ticket_id="TEST-003",
        event_data={"test": "data"},
        severity=FeedbackSeverity.LOW,
        timestamp=datetime.now()
    )
    
    event_dict = event.to_dict()
    event_restored = FeedbackEvent.from_dict(event_dict)
    
    if event.event_id == event_restored.event_id:
        print("    ✅ FeedbackEvent serialization works")
    else:
        print("    ❌ FeedbackEvent serialization failed")
        return False
    
    print("  ✅ Data model tests passed")
    return True


def main():
    """Run quick feedback system tests."""
    print("🚀 InsightDesk AI - Quick Feedback System Test")
    print("=" * 45)
    
    tests = [
        ("Data Models", test_data_models),
        ("Storage", test_feedback_storage),
        ("Retrieval", test_feedback_retrieval),
        ("Analytics", test_analytics),
        ("Health Check", test_health_check)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Feedback system is working correctly.")
        return True
    else:
        print("⚠️ Some tests FAILED. Please check the feedback system.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)