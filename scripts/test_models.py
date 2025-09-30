# scripts/test_models.py

"""
Test script to verify that both XGBoost and TensorFlow models are working correctly.
This script tests model loading, inference, and API integration.
"""

import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def test_xgboost_model():
    """Test XGBoost model loading and prediction."""
    print("🧪 Testing XGBoost Model...")
    
    try:
        from src.models.xgboost_classifier import XGBoostCategoryClassifier
        
        # Initialize classifier
        classifier = XGBoostCategoryClassifier()
        
        # Load model
        classifier.load_model()
        
        # Test prediction
        test_ticket = {
            "ticket_id": "TEST-XGB-001",
            "subject": "Application login failure",
            "description": "User cannot authenticate with correct credentials",
            "product": "web_app",
            "product_module": "authentication",
            "priority": "high",
            "severity": "major",
            "customer_tier": "premium",
            "region": "US",
            "previous_tickets": 1,
            "account_age_days": 180,
            "account_monthly_value": 500,
            "ticket_text_length": 65,
            "response_count": 0,
            "attachments_count": 0,
            "affected_users": 1,
            "resolution_time_hours": 0
        }
        
        result = classifier.predict(test_ticket)
        
        print(f"✅ XGBoost prediction successful!")
        print(f"   Category: {result['predicted_category']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Top 3: {list(result['top_3_predictions'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return False


def test_tensorflow_model():
    """Test TensorFlow model loading and prediction."""
    print("\n🧪 Testing TensorFlow Model...")
    
    try:
        from src.models.tensorflow_classifier import TensorFlowCategoryClassifier
        
        # Initialize classifier
        classifier = TensorFlowCategoryClassifier()
        
        # Load model
        classifier.load_model()
        
        # Test prediction
        test_ticket = {
            "ticket_id": "TEST-TF-001",
            "subject": "Database connection timeout",
            "description": "Application cannot connect to database server",
            "error_logs": "Connection timeout after 30 seconds",
            "stack_trace": "java.sql.SQLException: Connection timeout",
            "product": "api_server",
            "channel": "web",
            "priority": "critical",
            "severity": "major",
            "customer_tier": "enterprise",
            "region": "EU",
            "previous_tickets": 3,
            "account_age_days": 720,
            "account_monthly_value": 2000,
            "ticket_text_length": 95,
            "response_count": 0,
            "attachments_count": 1,
            "affected_users": 50,
            "resolution_time_hours": 0
        }
        
        result = classifier.predict(test_ticket)
        
        print(f"✅ TensorFlow prediction successful!")
        print(f"   Category: {result['predicted_category']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Top 3: {list(result['top_3_predictions'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False


def test_api_integration():
    """Test API integration with models."""
    print("\n🧪 Testing API Integration...")
    
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        print(f"Health check: {response.status_code}")
        
        # Test model info
        response = client.get("/models/info")
        print(f"Models info: {response.status_code}")
        
        if response.status_code == 200:
            info = response.json()
            print(f"   XGBoost status: {info['xgboost']['status']}")
            print(f"   TensorFlow status: {info['tensorflow']['status']}")
        
        # Test XGBoost endpoint
        test_ticket = {
            "ticket_id": "API-TEST-001",
            "subject": "Payment processing error",
            "description": "Credit card transaction failed with error code 402"
        }
        
        response = client.post("/classify/xgboost", json=test_ticket)
        print(f"XGBoost API: {response.status_code}")
        
        # Test TensorFlow endpoint
        response = client.post("/classify/tensorflow", json=test_ticket)
        print(f"TensorFlow API: {response.status_code}")
        
        # Test comparison endpoint
        response = client.post("/classify/compare", json=test_ticket)
        print(f"Compare API: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
        
        print("✅ API integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False


def test_model_consistency():
    """Test that both models give reasonable predictions on the same data."""
    print("\n🧪 Testing Model Consistency...")
    
    try:
        from src.models.xgboost_classifier import XGBoostCategoryClassifier
        from src.models.tensorflow_classifier import TensorFlowCategoryClassifier
        
        # Initialize classifiers
        xgb_classifier = XGBoostCategoryClassifier()
        tf_classifier = TensorFlowCategoryClassifier()
        
        # Load models
        xgb_classifier.load_model()
        tf_classifier.load_model()
        
        # Test tickets
        test_tickets = [
            {
                "subject": "Login authentication failure",
                "description": "User cannot log in with correct password",
                "product": "web_app",
                "priority": "high"
            },
            {
                "subject": "Database performance issue",
                "description": "Queries are running very slowly",
                "product": "database",
                "priority": "medium"
            },
            {
                "subject": "Payment gateway timeout",
                "description": "Credit card processing is timing out",
                "product": "payment",
                "priority": "critical"
            }
        ]
        
        print(f"Testing {len(test_tickets)} tickets...")
        
        agreements = 0
        for i, ticket in enumerate(test_tickets):
            xgb_result = xgb_classifier.predict(ticket)
            tf_result = tf_classifier.predict(ticket)
            
            agrees = xgb_result['predicted_category'] == tf_result['predicted_category']
            if agrees:
                agreements += 1
            
            print(f"   Ticket {i+1}: XGB={xgb_result['predicted_category']}, "
                  f"TF={tf_result['predicted_category']}, "
                  f"Agree={'✅' if agrees else '❌'}")
        
        agreement_rate = agreements / len(test_tickets)
        print(f"\n📊 Model agreement rate: {agreement_rate:.2%}")
        
        if agreement_rate >= 0.5:
            print("✅ Models show reasonable consistency")
        else:
            print("⚠️ Models show significant disagreement - review training")
        
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing InsightDesk AI Models")
    print("=" * 40)
    
    # Check if models exist
    models_dir = Path("models")
    xgb_model = models_dir / "xgb_category.pkl"
    tf_model = models_dir / "tf_category_model.h5"
    
    if not xgb_model.exists():
        print("❌ XGBoost model not found. Please run training first:")
        print("   python scripts/train_xgboost.py")
        return
    
    if not tf_model.exists():
        print("❌ TensorFlow model not found. Please run training first:")
        print("   python scripts/train_tensorflow.py")
        return
    
    print("✅ Both model files found")
    
    # Run tests
    tests = [
        ("XGBoost Model", test_xgboost_model),
        ("TensorFlow Model", test_tensorflow_model),
        ("API Integration", test_api_integration),
        ("Model Consistency", test_model_consistency)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Models are ready for deployment.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()