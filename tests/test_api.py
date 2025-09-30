import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app, get_model_manager


# Test client
client = TestClient(app)


# Mock data for testing
SAMPLE_TICKET = {
    "ticket_id": "TK-TEST-001",
    "subject": "Cannot login to application",
    "description": "User is unable to authenticate with correct credentials",
    "error_logs": "Authentication timeout after 30 seconds",
    "product": "web_application",
    "channel": "email",
    "priority": "high",
    "severity": "major",
    "customer_tier": "premium",
    "region": "US",
    "previous_tickets": 2,
    "account_age_days": 365,
    "account_monthly_value": 1000.0,
    "ticket_text_length": 65,
    "response_count": 0,
    "attachments_count": 1,
    "affected_users": 1,
    "resolution_time_hours": 0.0
}

MINIMAL_TICKET = {
    "ticket_id": "TK-MIN-001",
    "subject": "Test issue",
    "description": "This is a test issue"
}

INVALID_TICKET_MISSING_FIELDS = {
    "subject": "Missing required fields"
    # Missing ticket_id and description
}

SAMPLE_PREDICTION_RESPONSE = {
    "predicted_category": "authentication",
    "confidence": 0.95,
    "top_3_predictions": {
        "authentication": 0.95,
        "login": 0.03,
        "security": 0.02
    },
    "model_type": "xgboost",
    "inference_time_ms": 15.5
}


class TestBasicEndpoints:
    """Test basic API endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "InsightDesk AI" in data["message"]
        assert "version" in data
        assert "docs" in data
        assert "health" in data
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["status", "version", "models_loaded", "available_models", "mlflow_available", "timestamp"]
        for field in required_fields:
            assert field in data
        
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert isinstance(data["models_loaded"], bool)
        assert isinstance(data["available_models"], list)
        assert isinstance(data["mlflow_available"], bool)
    
    def test_models_info_endpoint(self):
        """Test the models info endpoint."""
        response = client.get("/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "models_loaded" in data
        assert "available_models" in data
        assert "model_details" in data


class TestInputValidation:
    """Test input validation for API endpoints."""
    
    def test_valid_minimal_ticket(self):
        """Test with minimal valid ticket data."""
        response = client.post("/predict/category", json=MINIMAL_TICKET)
        # Should succeed or fail gracefully with 503 if models not loaded
        assert response.status_code in [200, 503]
    
    def test_valid_complete_ticket(self):
        """Test with complete valid ticket data."""
        response = client.post("/predict/category", json=SAMPLE_TICKET)
        # Should succeed or fail gracefully with 503 if models not loaded
        assert response.status_code in [200, 503]
    
    def test_invalid_ticket_missing_required_fields(self):
        """Test with missing required fields."""
        response = client.post("/predict/category", json=INVALID_TICKET_MISSING_FIELDS)
        assert response.status_code == 422  # Validation error
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_invalid_model_type(self):
        """Test with invalid model_type parameter."""
        response = client.post(
            "/predict/category?model_type=invalid_model", 
            json=SAMPLE_TICKET
        )
        assert response.status_code == 400  # Bad request
        
        error_data = response.json()
        assert "detail" in error_data
        assert "Invalid model_type" in error_data["detail"]
    
    def test_valid_model_types(self):
        """Test with valid model_type parameters."""
        valid_model_types = ["xgboost", "tensorflow", "both"]
        
        for model_type in valid_model_types:
            response = client.post(
                f"/predict/category?model_type={model_type}",
                json=SAMPLE_TICKET
            )
            # Should succeed or fail gracefully with 503 if models not loaded
            assert response.status_code in [200, 503]
    
    def test_field_validation(self):
        """Test field validation (e.g., numeric constraints)."""
        invalid_ticket = SAMPLE_TICKET.copy()
        invalid_ticket["previous_tickets"] = -5  # Should be >= 0
        invalid_ticket["affected_users"] = 0  # Should be >= 1
        
        response = client.post("/predict/category", json=invalid_ticket)
        assert response.status_code == 422  # Validation error
    
    def test_field_defaults(self):
        """Test that optional fields get proper defaults."""
        ticket_with_defaults = {
            "ticket_id": "TK-DEFAULT-001",
            "subject": "Test subject",
            "description": "Test description"
        }
        
        response = client.post("/predict/category", json=ticket_with_defaults)
        # Should succeed or fail gracefully with 503 if models not loaded
        assert response.status_code in [200, 503]


class TestCategoryPrediction:
    """Test category prediction endpoints."""
    
    def test_predict_category_endpoint_structure(self):
        """Test the structure of category prediction response."""
        with patch('src.api.main.model_manager') as mock_manager:
            # Mock model manager
            mock_manager.get_available_models.return_value = ["xgboost"]
            mock_manager.predict_category.return_value = {
                "xgboost": SAMPLE_PREDICTION_RESPONSE,
                "total_inference_time_ms": 15.5,
                "timestamp": "2025-09-29T10:00:00"
            }
            
            response = client.post("/predict/category", json=SAMPLE_TICKET)
            assert response.status_code == 200
            
            data = response.json()
            required_fields = ["ticket_id", "predictions", "available_models", "total_inference_time_ms", "timestamp"]
            for field in required_fields:
                assert field in data
            
            assert data["ticket_id"] == SAMPLE_TICKET["ticket_id"]
            assert isinstance(data["predictions"], dict)
            assert isinstance(data["available_models"], list)
            assert isinstance(data["total_inference_time_ms"], (int, float))
    
    def test_predict_category_no_models_available(self):
        """Test category prediction when no models are available."""
        with patch('src.api.main.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = []
            
            response = client.post("/predict/category", json=SAMPLE_TICKET)
            assert response.status_code == 503
            
            error_data = response.json()
            assert "detail" in error_data
            assert "No ML models are currently available" in error_data["detail"]
    
    def test_predict_category_with_both_models(self):
        """Test category prediction with both models."""
        with patch('src.api.main.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = ["xgboost", "tensorflow"]
            mock_manager.predict_category.return_value = {
                "xgboost": SAMPLE_PREDICTION_RESPONSE,
                "tensorflow": {
                    **SAMPLE_PREDICTION_RESPONSE,
                    "model_type": "tensorflow",
                    "predicted_category": "login_issue"
                },
                "total_inference_time_ms": 25.0,
                "timestamp": "2025-09-29T10:00:00"
            }
            
            response = client.post("/predict/category?model_type=both", json=SAMPLE_TICKET)
            assert response.status_code == 200
            
            data = response.json()
            assert "xgboost" in data["predictions"]
            assert "tensorflow" in data["predictions"]
    
    def test_predict_category_single_model(self):
        """Test category prediction with single model."""
        with patch('src.api.main.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = ["xgboost"]
            mock_manager.predict_category.return_value = {
                "xgboost": SAMPLE_PREDICTION_RESPONSE,
                "total_inference_time_ms": 15.5,
                "timestamp": "2025-09-29T10:00:00"
            }
            
            response = client.post("/predict/category?model_type=xgboost", json=SAMPLE_TICKET)
            assert response.status_code == 200
            
            data = response.json()
            assert "xgboost" in data["predictions"]
            assert data["predictions"]["xgboost"]["model_type"] == "xgboost"
    
    def test_predict_category_error_handling(self):
        """Test error handling in category prediction."""
        with patch('src.api.main.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = ["xgboost"]
            mock_manager.predict_category.side_effect = Exception("Model prediction failed")
            
            response = client.post("/predict/category", json=SAMPLE_TICKET)
            assert response.status_code == 500
            
            error_data = response.json()
            assert "detail" in error_data
            assert "Prediction failed" in error_data["detail"]


class TestPriorityPrediction:
    """Test priority prediction endpoints."""
    
    def test_predict_priority_placeholder(self):
        """Test priority prediction placeholder endpoint."""
        response = client.post("/predict/priority", json=SAMPLE_TICKET)
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["ticket_id", "predicted_priority", "confidence", "message"]
        for field in required_fields:
            assert field in data
        
        assert data["ticket_id"] == SAMPLE_TICKET["ticket_id"]
        assert data["predicted_priority"] == "medium"  # Default placeholder
        assert data["confidence"] == 0.0
        assert "not yet implemented" in data["message"]


class TestMLflowIntegration:
    """Test MLflow integration endpoints."""
    
    def test_list_experiments_without_mlflow(self):
        """Test listing experiments when MLflow is not available."""
        with patch('src.api.main.MLFLOW_AVAILABLE', False):
            response = client.get("/experiments")
            assert response.status_code == 503
            
            error_data = response.json()
            assert "MLflow not available" in error_data["detail"]
    
    def test_list_experiments_with_mlflow(self):
        """Test listing experiments when MLflow is available."""
        with patch('src.api.main.MLFLOW_AVAILABLE', True):
            with patch('src.api.main.mlflow') as mock_mlflow:
                # Mock experiment data
                mock_experiment = Mock()
                mock_experiment.experiment_id = "1"
                mock_experiment.name = "test_experiment"
                mock_experiment.lifecycle_stage = "active"
                
                mock_mlflow.search_experiments.return_value = [mock_experiment]
                
                response = client.get("/experiments")
                assert response.status_code == 200
                
                data = response.json()
                assert "experiments" in data
                assert len(data["experiments"]) == 1
                assert data["experiments"][0]["name"] == "test_experiment"
    
    def test_list_runs_without_mlflow(self):
        """Test listing runs when MLflow is not available."""
        with patch('src.api.main.MLFLOW_AVAILABLE', False):
            response = client.get("/experiments/1/runs")
            assert response.status_code == 503
    
    def test_list_runs_with_mlflow(self):
        """Test listing runs when MLflow is available."""
        with patch('src.api.main.MLFLOW_AVAILABLE', True):
            with patch('src.api.main.mlflow') as mock_mlflow:
                import pandas as pd
                
                # Mock runs data
                mock_runs = pd.DataFrame({
                    "run_id": ["run_1", "run_2"],
                    "experiment_id": ["1", "1"],
                    "status": ["FINISHED", "FINISHED"]
                })
                
                mock_mlflow.search_runs.return_value = mock_runs
                
                response = client.get("/experiments/1/runs")
                assert response.status_code == 200
                
                data = response.json()
                assert "runs" in data
                assert len(data["runs"]) == 2


class TestResponseSchema:
    """Test response schema compliance."""
    
    def test_category_prediction_response_schema(self):
        """Test that category prediction response follows the expected schema."""
        with patch('src.api.main.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = ["xgboost"]
            mock_manager.predict_category.return_value = {
                "xgboost": SAMPLE_PREDICTION_RESPONSE,
                "total_inference_time_ms": 15.5,
                "timestamp": "2025-09-29T10:00:00.123456"
            }
            
            response = client.post("/predict/category", json=SAMPLE_TICKET)
            assert response.status_code == 200
            
            data = response.json()
            
            # Check all required fields exist
            assert isinstance(data["ticket_id"], str)
            assert isinstance(data["predictions"], dict)
            assert isinstance(data["available_models"], list)
            assert isinstance(data["total_inference_time_ms"], (int, float))
            assert isinstance(data["timestamp"], str)
            
            # Check nested prediction structure
            if "xgboost" in data["predictions"]:
                xgb_pred = data["predictions"]["xgboost"]
                assert isinstance(xgb_pred["predicted_category"], str)
                assert isinstance(xgb_pred["confidence"], (int, float))
                assert isinstance(xgb_pred["top_3_predictions"], dict)
                assert isinstance(xgb_pred["model_type"], str)
    
    def test_health_response_schema(self):
        """Test that health response follows the expected schema."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["models_loaded"], bool)
        assert isinstance(data["available_models"], list)
        assert isinstance(data["mlflow_available"], bool)
        assert isinstance(data["timestamp"], str)


class TestLogging:
    """Test logging functionality."""
    
    def test_prediction_logging(self):
        """Test that predictions are properly logged."""
        import logging
        
        with patch('src.api.main.model_manager') as mock_manager:
            with patch('src.api.main.logger') as mock_logger:
                mock_manager.get_available_models.return_value = ["xgboost"]
                mock_manager.predict_category.return_value = {
                    "xgboost": SAMPLE_PREDICTION_RESPONSE,
                    "total_inference_time_ms": 15.5,
                    "timestamp": "2025-09-29T10:00:00"
                }
                
                response = client.post("/predict/category", json=SAMPLE_TICKET)
                assert response.status_code == 200
                
                # Check that logging was called
                mock_logger.info.assert_called()
                
                # Check log content
                log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                
                # Should log prediction request
                request_logged = any("Category prediction request" in log for log in log_calls)
                assert request_logged
                
                # Should log successful prediction
                success_logged = any("Category prediction successful" in log for log in log_calls)
                assert success_logged


# Integration test (will only pass if models are actually loaded)
class TestRealModelIntegration:
    """Integration tests with real models (optional - requires trained models)."""
    
    def test_real_category_prediction(self):
        """Test category prediction with real models if available."""
        response = client.post("/predict/category", json=SAMPLE_TICKET)
        
        if response.status_code == 200:
            # Models are loaded and working
            data = response.json()
            assert data["ticket_id"] == SAMPLE_TICKET["ticket_id"]
            assert "predictions" in data
            
            # Check that at least one model made a prediction
            predictions = data["predictions"]
            has_prediction = False
            for model_name in ["xgboost", "tensorflow"]:
                if model_name in predictions:
                    pred = predictions[model_name]
                    assert "predicted_category" in pred
                    assert "confidence" in pred
                    assert pred["confidence"] >= 0 and pred["confidence"] <= 1
                    has_prediction = True
            
            assert has_prediction, "At least one model should make a prediction"
            
        elif response.status_code == 503:
            # Models not loaded - this is acceptable in test environment
            pytest.skip("Models not loaded in test environment")
        else:
            # Unexpected error
            pytest.fail(f"Unexpected response status: {response.status_code}")


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        "-v",
        "tests/test_api.py::TestBasicEndpoints",
        "tests/test_api.py::TestInputValidation", 
        "tests/test_api.py::TestCategoryPrediction"
    ])
