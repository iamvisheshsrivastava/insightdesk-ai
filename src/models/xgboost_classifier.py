# src/models/xgboost_classifier.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
from src.features.feature_builder import build_feature_pipeline, prepare_features


class XGBoostCategoryClassifier:
    """XGBoost-based ticket category classifier for inference."""
    
    def __init__(self, models_dir: Union[str, Path] = "models"):
        """Initialize the classifier with model artifacts."""
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_pipeline = None
        self.label_encoder = None
        self.is_loaded = False
    
    def load_model(self) -> None:
        """Load all model artifacts for inference."""
        try:
            # Load XGBoost model
            model_path = self.models_dir / "xgb_category.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"XGBoost model not found: {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"✅ Loaded XGBoost model from {model_path}")
            
            # Load feature pipeline
            pipeline_path = self.models_dir / "feature_pipeline.pkl"
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Feature pipeline not found: {pipeline_path}")
            
            self.feature_pipeline = joblib.load(pipeline_path)
            print(f"✅ Loaded feature pipeline from {pipeline_path}")
            
            # Load label encoder
            label_encoder_path = self.models_dir / "label_encoder_category.pkl"
            if not label_encoder_path.exists():
                raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
            
            self.label_encoder = joblib.load(label_encoder_path)
            print(f"✅ Loaded label encoder from {label_encoder_path}")
            
            self.is_loaded = True
            print("🚀 XGBoost classifier ready for inference!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model: {str(e)}")
    
    def preprocess_ticket(self, ticket_data: Dict) -> pd.DataFrame:
        """Preprocess a single ticket for inference."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create DataFrame from ticket data
        df = pd.DataFrame([ticket_data])
        
        # Prepare features (same as training)
        df = prepare_features(df)
        
        return df
    
    def predict(self, ticket_data: Dict) -> Dict[str, Union[str, float, Dict]]:
        """Predict category for a single ticket."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        df = self.preprocess_ticket(ticket_data)
        
        # Transform features
        X = self.feature_pipeline.transform(df)
        
        # Predict probabilities
        predictions = self.model.predict_proba(X)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_category = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = {
            self.label_encoder.inverse_transform([idx])[0]: float(predictions[0][idx])
            for idx in top_3_indices
        }
        
        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "model_type": "xgboost"
        }
    
    def predict_batch(self, tickets_data: List[Dict]) -> List[Dict]:
        """Predict categories for multiple tickets."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for ticket_data in tickets_data:
            try:
                result = self.predict(ticket_data)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "predicted_category": "unknown",
                    "confidence": 0.0,
                    "model_type": "xgboost"
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": "xgboost",
            "num_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "feature_count": self.model.n_features_in_
        }


# Singleton instance for FastAPI
xgb_classifier = XGBoostCategoryClassifier()


def load_xgboost_model():
    """Load the XGBoost model (call this at startup)."""
    global xgb_classifier
    xgb_classifier.load_model()


def predict_category_xgboost(ticket_data: Dict) -> Dict:
    """Predict ticket category using XGBoost model."""
    global xgb_classifier
    return xgb_classifier.predict(ticket_data)


def predict_categories_xgboost_batch(tickets_data: List[Dict]) -> List[Dict]:
    """Predict ticket categories for multiple tickets using XGBoost model."""
    global xgb_classifier
    return xgb_classifier.predict_batch(tickets_data)


if __name__ == "__main__":
    # Test the classifier
    classifier = XGBoostCategoryClassifier()
    
    try:
        classifier.load_model()
        
        # Test prediction
        test_ticket = {
            "subject": "Login issue with application",
            "description": "User cannot log into the system. Getting error message.",
            "product": "web_app",
            "product_module": "authentication",
            "priority": "high",
            "severity": "major",
            "customer_tier": "premium",
            "region": "US",
            "previous_tickets": 2,
            "account_age_days": 365,
            "account_monthly_value": 1000,
            "ticket_text_length": 50,
            "response_count": 0,
            "attachments_count": 0,
            "affected_users": 1,
            "resolution_time_hours": 0
        }
        
        result = classifier.predict(test_ticket)
        print("🧪 Test prediction:")
        print(f"Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Top 3: {result['top_3_predictions']}")
        
    except Exception as e:
        print(f"❌ Error testing classifier: {e}")
        print("💡 Make sure to train the XGBoost model first by running:")
        print("   python scripts/train_xgboost.py")