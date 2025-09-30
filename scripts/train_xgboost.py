# scripts/train_xgboost.py

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import time
import json

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
    print("✅ MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available. Skipping experiment tracking.")


def setup_mlflow():
    """Setup MLflow experiment tracking."""
    if MLFLOW_AVAILABLE:
        # Create mlruns directory
        mlflow_dir = Path("mlruns")
        mlflow_dir.mkdir(exist_ok=True)
        
        # Set tracking URI
        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        # Set or create experiment
        experiment_name = "ticket_classification"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"📊 Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                mlflow.set_experiment(experiment_name)
                print(f"📊 Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"⚠️ MLflow experiment setup warning: {e}")


def log_confusion_matrix(y_true, y_pred, labels, model_name="xgboost"):
    """Create and log confusion matrix plot to MLflow."""
    try:
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(plot_path), "plots")
        
        plt.close()
        print(f"📊 Confusion matrix saved: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create confusion matrix: {e}")


def main():
    """Main training pipeline with MLflow tracking."""
    start_time = time.time()
    
    # Setup MLflow
    setup_mlflow()
    
    # Start MLflow run
    if MLFLOW_AVAILABLE:
        mlflow.start_run(run_name="xgboost_category_classifier")
    
    try:
        # Paths
        data_path = Path("data/support_tickets.json")
        features_path = Path("data/features.joblib")
        pipeline_path = Path("models/feature_pipeline.pkl")

        # Check if features exist
        if not features_path.exists():
            print("❌ Features not found. Please run:")
            print("   python scripts/build_features.py")
            return

        # --- Load raw data (for labels) ---
        print("📊 Loading data...")
        from src.ingestion.data_loader import load_json_tickets
        df = load_json_tickets(data_path)

        # --- Load feature matrix & pipeline ---
        print("📊 Loading features...")
        X = joblib.load(features_path)
        pipeline = joblib.load(pipeline_path)

        # --- Target variable (category) ---
        y_raw = df["category"]
        print(f"📊 Target distribution:")
        print(y_raw.value_counts())

        # Encode category labels → integers
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Save the encoder for inference later
        Path("models").mkdir(exist_ok=True)
        joblib.dump(le, "models/label_encoder_category.pkl")
        print("💾 Saved label encoder → models/label_encoder_category.pkl")

        # --- Train/validation split ---
        print("📊 Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")

        # --- Define XGBoost model parameters ---
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        # Log parameters to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                "model_type": "xgboost",
                "feature_dim": X_train.shape[1],
                "n_classes": len(le.classes_),
                "train_samples": X_train.shape[0],
                "val_samples": X_val.shape[0],
                **xgb_params
            })
            
            # Log feature pipeline info
            mlflow.log_param("vectorizer_max_features", 5000)  # From feature_builder.py
            mlflow.log_param("text_features", "subject + description")
            mlflow.log_param("categorical_features", "priority,severity,customer_tier,product,product_module,region")

        # --- Train ---
        print("⚙️ Training XGBoost model...")
        model = XGBClassifier(**xgb_params)
        
        training_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - training_start

        # --- Evaluate ---
        print("📊 Evaluating model...")
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_micro = f1_score(y_val, y_pred, average='micro')
        
        # Calculate precision and recall
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

        # Print results
        print("\n📊 XGBoost Classification Results:")
        print("=" * 40)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Weighted F1 Score:  {f1_weighted:.4f}")
        print(f"Macro F1 Score:     {f1_macro:.4f}")
        print(f"Micro F1 Score:     {f1_micro:.4f}")
        print(f"Macro Precision:    {precision_macro:.4f}")
        print(f"Macro Recall:       {recall_macro:.4f}")
        print(f"Weighted Precision: {precision_weighted:.4f}")
        print(f"Weighted Recall:    {recall_weighted:.4f}")
        print(f"Training Time:      {training_time:.2f} seconds")

        # Log metrics to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "training_time_seconds": training_time,
                "total_time_seconds": time.time() - start_time
            })

        # --- Detailed Classification Report ---
        print("\n📊 Detailed Classification Report:")
        class_report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
        print(classification_report(y_val, y_pred, target_names=le.classes_))

        # Log classification report to MLflow
        if MLFLOW_AVAILABLE:
            # Save classification report as JSON
            report_path = Path("models/xgb_classification_report.json")
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=2)
            mlflow.log_artifact(str(report_path), "reports")

        # Create and log confusion matrix
        log_confusion_matrix(y_val, y_pred, le.classes_, "xgboost")

        # --- Save model ---
        model_path = "models/xgb_category.pkl"
        joblib.dump(model, model_path)
        print(f"💾 Saved XGBoost model → {model_path}")

        # Log model to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(model, "xgboost_model")
            mlflow.log_artifact(model_path, "models")
            mlflow.log_artifact("models/label_encoder_category.pkl", "models")
            mlflow.log_artifact("models/feature_pipeline.pkl", "models")

        # Feature importance analysis
        try:
            feature_importance = model.feature_importances_
            print(f"\n📊 Top 10 Most Important Features:")
            
            # Get feature names (this is simplified - in reality you'd need to map back to original features)
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"   {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
            
            # Log feature importance
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("top_feature_importance", feature_importance[top_indices[0]])
                
        except Exception as e:
            print(f"⚠️ Feature importance analysis failed: {e}")

        # Success summary
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed successfully!")
        print(f"✅ Total time: {total_time:.2f} seconds")
        print(f"✅ Model saved: {model_path}")
        
        # Check if target is met
        target_f1 = 0.85
        if f1_weighted >= target_f1:
            print(f"🎯 SUCCESS: Weighted F1 ({f1_weighted:.4f}) meets target (≥{target_f1})")
        else:
            print(f"⚠️  WARNING: Weighted F1 ({f1_weighted:.4f}) below target (≥{target_f1})")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
        raise
    
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


if __name__ == "__main__":
    main()
