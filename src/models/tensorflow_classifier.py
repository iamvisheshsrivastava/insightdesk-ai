# src/models/tensorflow_classifier.py

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple
import tensorflow as tf


class TensorFlowCategoryClassifier:
    """TensorFlow-based ticket category classifier for inference."""
    
    def __init__(self, models_dir: Union[str, Path] = "models"):
        """Initialize the classifier with model artifacts."""
        self.models_dir = Path(models_dir)
        self.model = None
        self.vectorizer = None
        self.categorical_encoders = None
        self.numeric_stats = None
        self.label_encoder = None
        self.is_loaded = False
        
        # Feature definitions (must match training)
        self.TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]
        self.CATEGORICAL_COLS = ["product", "channel", "priority", "severity", "customer_tier", "region"]
        self.NUMERIC_COLS = [
            "previous_tickets", "account_age_days", "account_monthly_value",
            "ticket_text_length", "response_count", "attachments_count",
            "affected_users", "resolution_time_hours"
        ]
    
    def load_model(self) -> None:
        """Load all model artifacts for inference."""
        try:
            # Load TensorFlow model
            model_path = self.models_dir / "tf_category_model.h5"
            if not model_path.exists():
                raise FileNotFoundError(f"TensorFlow model not found: {model_path}")
            
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded TensorFlow model from {model_path}")
            
            # Load text vectorizer configuration
            vectorizer_path = self.models_dir / "tf_text_vectorizer.json"
            with open(vectorizer_path, 'r') as f:
                vectorizer_config = json.load(f)
            
            # Recreate vectorizer
            self.vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=vectorizer_config["max_tokens"],
                sequence_length=vectorizer_config["sequence_length"],
                output_mode="int"
            )
            self.vectorizer.set_vocabulary(vectorizer_config["vocabulary"])
            print(f"✅ Loaded text vectorizer from {vectorizer_path}")
            
            # Load categorical encoders
            encoders_path = self.models_dir / "tf_categorical_encoders.pkl"
            self.categorical_encoders = joblib.load(encoders_path)
            print(f"✅ Loaded categorical encoders from {encoders_path}")
            
            # Load numeric stats
            numeric_stats_path = self.models_dir / "tf_numeric_stats.json"
            with open(numeric_stats_path, 'r') as f:
                self.numeric_stats = json.load(f)
            print(f"✅ Loaded numeric stats from {numeric_stats_path}")
            
            # Load label encoder
            label_encoder_path = self.models_dir / "tf_label_encoder_category.pkl"
            self.label_encoder = joblib.load(label_encoder_path)
            print(f"✅ Loaded label encoder from {label_encoder_path}")
            
            self.is_loaded = True
            print("🚀 TensorFlow classifier ready for inference!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model: {str(e)}")
    
    def preprocess_ticket(self, ticket_data: Dict) -> List[np.ndarray]:
        """Preprocess a single ticket for inference."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Combine text fields
        text_parts = []
        for col in self.TEXT_COLS:
            value = ticket_data.get(col, "")
            if value is None:
                value = ""
            text_parts.append(str(value))
        
        combined_text = " ".join(text_parts)
        
        # Process text
        text_data = self.vectorizer([combined_text])
        
        # Process categorical features
        categorical_data = []
        for col in self.CATEGORICAL_COLS:
            value = ticket_data.get(col, "unknown")
            if value is None:
                value = "unknown"
            
            # Handle unknown categories
            try:
                encoded = self.categorical_encoders[col].transform([str(value)])[0]
            except ValueError:
                # Unknown category, use first class (or you could add an "unknown" class)
                encoded = 0
            
            categorical_data.append(np.array([encoded]))
        
        # Process numeric features
        numeric_data = np.zeros((1, len(self.NUMERIC_COLS)))
        for i, col in enumerate(self.NUMERIC_COLS):
            value = ticket_data.get(col, 0)
            if value is None:
                value = 0
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
            
            # Normalize using training stats
            stats = self.numeric_stats[col]
            if stats["std"] > 0:
                normalized = (value - stats["mean"]) / stats["std"]
            else:
                normalized = value - stats["mean"]
            
            numeric_data[0, i] = normalized
        
        # Return in the same order as model inputs
        return [text_data] + categorical_data + [numeric_data]
    
    def predict(self, ticket_data: Dict) -> Dict[str, Union[str, float, Dict]]:
        """Predict category for a single ticket."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        X = self.preprocess_ticket(ticket_data)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
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
            "model_type": "tensorflow"
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
                    "model_type": "tensorflow"
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": "tensorflow",
            "num_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "text_vocab_size": len(self.vectorizer.get_vocabulary()),
            "categorical_features": list(self.categorical_encoders.keys()),
            "numeric_features": self.NUMERIC_COLS
        }


# Singleton instance for FastAPI
tf_classifier = TensorFlowCategoryClassifier()


def load_tensorflow_model():
    """Load the TensorFlow model (call this at startup)."""
    global tf_classifier
    tf_classifier.load_model()


def predict_category_tensorflow(ticket_data: Dict) -> Dict:
    """Predict ticket category using TensorFlow model."""
    global tf_classifier
    return tf_classifier.predict(ticket_data)


def predict_categories_tensorflow_batch(tickets_data: List[Dict]) -> List[Dict]:
    """Predict ticket categories for multiple tickets using TensorFlow model."""
    global tf_classifier
    return tf_classifier.predict_batch(tickets_data)


if __name__ == "__main__":
    # Test the classifier
    classifier = TensorFlowCategoryClassifier()
    
    try:
        classifier.load_model()
        
        # Test prediction
        test_ticket = {
            "subject": "Login issue with application",
            "description": "User cannot log into the system. Getting error message.",
            "error_logs": "Authentication failed",
            "stack_trace": "",
            "product": "web_app",
            "channel": "email",
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
        print("💡 Make sure to train the TensorFlow model first by running:")
        print("   python scripts/train_tensorflow.py")