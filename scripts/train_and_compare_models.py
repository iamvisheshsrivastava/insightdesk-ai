# scripts/train_and_compare_models.py

"""
Comprehensive training and evaluation script that trains both XGBoost and TensorFlow models
and provides detailed performance comparison for the technical assessment.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try to import MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
    print("✅ MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available. Skipping experiment tracking.")


def setup_mlflow():
    """Setup MLflow experiment tracking."""
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("model_comparison")
        print("📊 MLflow experiment 'model_comparison' initialized")


def load_and_prepare_data():
    """Load and prepare data for both models."""
    print("📊 Loading and preparing data...")
    
    from src.ingestion.data_loader import load_json_tickets
    data_path = Path("data/support_tickets.json")
    
    if not data_path.exists():
        print("❌ Data file not found. Please run:")
        print("   python scripts/unzip_and_load.py")
        return None
    
    df = load_json_tickets(data_path)
    
    # Prepare target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    
    print(f"✅ Data loaded: {df.shape[0]} tickets, {len(label_encoder.classes_)} categories")
    print(f"📊 Categories: {list(label_encoder.classes_)}")
    
    # Check class distribution
    class_counts = pd.Series(y).value_counts().sort_index()
    print(f"📊 Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"   {label_encoder.classes_[i]}: {count} ({count/len(y)*100:.1f}%)")
    
    return df, y, label_encoder


def train_xgboost_model(df, y, label_encoder):
    """Train XGBoost model."""
    print("\n🚀 Training XGBoost Model...")
    
    if MLFLOW_AVAILABLE:
        mlflow.start_run(run_name="xgboost_classifier")
    
    try:
        # Prepare features
        from src.features.feature_builder import build_feature_pipeline, prepare_features
        
        df_prepared = prepare_features(df)
        pipeline = build_feature_pipeline()
        
        print("⚙️ Building features...")
        X = pipeline.fit_transform(df_prepared)
        print(f"✅ Feature matrix shape: {X.shape}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train XGBoost
        from xgboost import XGBClassifier
        
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
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"✅ XGBoost Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Weighted F1: {f1_weighted:.4f}")
        print(f"   Macro F1: {f1_macro:.4f}")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_params(xgb_params)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro
            })
            mlflow.sklearn.log_model(model, "xgboost_model")
        
        # Save model artifacts
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(model, models_dir / "xgb_category.pkl")
        joblib.dump(pipeline, models_dir / "feature_pipeline.pkl")
        joblib.dump(label_encoder, models_dir / "label_encoder_category.pkl")
        
        return {
            "model": model,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "test_labels": y_test,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "feature_matrix": X_test
        }
        
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


def train_tensorflow_model(df, y, label_encoder):
    """Train TensorFlow model."""
    print("\n🚀 Training TensorFlow Model...")
    
    if MLFLOW_AVAILABLE:
        mlflow.start_run(run_name="tensorflow_classifier")
    
    try:
        # Import training script functions
        from scripts.train_tensorflow import (
            create_preprocessors, preprocess_data, build_model, 
            train_model, CONFIG, TEXT_COLS, CATEGORICAL_COLS, NUMERIC_COLS
        )
        import tensorflow as tf
        from sklearn.utils.class_weight import compute_class_weight
        
        # Prepare data for TensorFlow
        print("⚙️ Preparing data for TensorFlow...")
        
        # Create preprocessors
        vectorizer, categorical_encoders, numeric_stats = create_preprocessors(df)
        
        # Preprocess data
        text_data, categorical_data, numeric_data = preprocess_data(
            df, vectorizer, categorical_encoders, numeric_stats
        )
        
        # Prepare input data
        X = [text_data] + [categorical_data[col] for col in CATEGORICAL_COLS] + [numeric_data]
        
        # Train/test split
        train_size = int(0.7 * len(df))
        test_size = int(0.3 * len(df))
        
        indices = np.random.RandomState(42).permutation(len(df))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:train_size + test_size]
        
        X_train = [x[train_idx] for x in X]
        X_test = [x[test_idx] for x in X]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        print(f"✅ Data prepared - Train: {len(y_train)}, Test: {len(y_test)}")
        
        # Build model
        num_classes = len(label_encoder.classes_)
        model = build_model(num_classes, categorical_encoders, CONFIG)
        
        # Compute class weights
        class_weights_array = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        # Train model (use smaller config for faster training in comparison)
        quick_config = CONFIG.copy()
        quick_config.update({
            "epochs": 50,
            "batch_size": 64,
            "early_stopping_patience": 5
        })
        
        history = train_model(model, X_train, y_train, X_train, y_train, class_weights, quick_config)
        
        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"✅ TensorFlow Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Weighted F1: {f1_weighted:.4f}")
        print(f"   Macro F1: {f1_macro:.4f}")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_params(quick_config)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1]
            })
            mlflow.tensorflow.log_model(model, "tensorflow_model")
        
        # Save model artifacts
        models_dir = Path("models")
        model.save(models_dir / "tf_category_model.h5")
        
        # Save preprocessors
        vectorizer_config = {
            "vocabulary": vectorizer.get_vocabulary(),
            "max_tokens": CONFIG["max_vocab_size"],
            "sequence_length": CONFIG["max_sequence_length"]
        }
        
        with open(models_dir / "tf_text_vectorizer.json", "w") as f:
            json.dump(vectorizer_config, f)
        
        joblib.dump(categorical_encoders, models_dir / "tf_categorical_encoders.pkl")
        
        with open(models_dir / "tf_numeric_stats.json", "w") as f:
            json.dump(numeric_stats, f)
        
        joblib.dump(label_encoder, models_dir / "tf_label_encoder_category.pkl")
        
        return {
            "model": model,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "test_labels": y_test,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "history": history
        }
        
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


def compare_models(xgb_results, tf_results, label_encoder):
    """Compare the performance of both models."""
    print("\n📊 Model Comparison Report")
    print("=" * 50)
    
    # Performance comparison
    print("\n🏆 Performance Metrics:")
    print(f"{'Metric':<15} {'XGBoost':<10} {'TensorFlow':<12} {'Winner':<10}")
    print("-" * 50)
    
    metrics = ['accuracy', 'f1_weighted', 'f1_macro']
    for metric in metrics:
        xgb_val = xgb_results[metric]
        tf_val = tf_results[metric]
        winner = "XGBoost" if xgb_val > tf_val else "TensorFlow"
        print(f"{metric:<15} {xgb_val:<10.4f} {tf_val:<12.4f} {winner:<10}")
    
    # Statistical significance test
    print(f"\n📈 Detailed Analysis:")
    print(f"XGBoost F1 (weighted): {xgb_results['f1_weighted']:.4f}")
    print(f"TensorFlow F1 (weighted): {tf_results['f1_weighted']:.4f}")
    print(f"Difference: {abs(xgb_results['f1_weighted'] - tf_results['f1_weighted']):.4f}")
    
    # Determine overall winner
    xgb_score = (xgb_results['accuracy'] + xgb_results['f1_weighted'] + xgb_results['f1_macro']) / 3
    tf_score = (tf_results['accuracy'] + tf_results['f1_weighted'] + tf_results['f1_macro']) / 3
    
    print(f"\n🏆 Overall Winner: {'XGBoost' if xgb_score > tf_score else 'TensorFlow'}")
    print(f"XGBoost average score: {xgb_score:.4f}")
    print(f"TensorFlow average score: {tf_score:.4f}")
    
    # Assessment requirement check
    target_f1 = 0.85
    print(f"\n🎯 Assessment Target (≥{target_f1} weighted F1):")
    print(f"XGBoost: {'✅ PASSED' if xgb_results['f1_weighted'] >= target_f1 else '❌ FAILED'}")
    print(f"TensorFlow: {'✅ PASSED' if tf_results['f1_weighted'] >= target_f1 else '❌ FAILED'}")
    
    # Create comparison plots
    create_comparison_plots(xgb_results, tf_results, label_encoder)
    
    return {
        "xgboost_score": xgb_score,
        "tensorflow_score": tf_score,
        "winner": "XGBoost" if xgb_score > tf_score else "TensorFlow",
        "xgb_meets_target": xgb_results['f1_weighted'] >= target_f1,
        "tf_meets_target": tf_results['f1_weighted'] >= target_f1
    }


def create_comparison_plots(xgb_results, tf_results, label_encoder):
    """Create visualization plots comparing both models."""
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Confusion matrices comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # XGBoost confusion matrix
    cm_xgb = confusion_matrix(xgb_results['test_labels'], xgb_results['predictions'])
    sns.heatmap(cm_xgb, annot=True, fmt='d', ax=ax1, cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax1.set_title(f'XGBoost Confusion Matrix\nF1: {xgb_results["f1_weighted"]:.3f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # TensorFlow confusion matrix
    cm_tf = confusion_matrix(tf_results['test_labels'], tf_results['predictions'])
    sns.heatmap(cm_tf, annot=True, fmt='d', ax=ax2, cmap='Greens',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax2.set_title(f'TensorFlow Confusion Matrix\nF1: {tf_results["f1_weighted"]:.3f}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance comparison bar chart
    metrics = ['Accuracy', 'Weighted F1', 'Macro F1']
    xgb_values = [xgb_results['accuracy'], xgb_results['f1_weighted'], xgb_results['f1_macro']]
    tf_values = [tf_results['accuracy'], tf_results['f1_weighted'], tf_results['f1_macro']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, xgb_values, width, label='XGBoost', color='skyblue')
    bars2 = ax.bar(x + width/2, tf_values, width, label='TensorFlow', color='lightgreen')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to {plots_dir}/")


def generate_final_report(comparison_results, xgb_results, tf_results):
    """Generate a comprehensive final report."""
    report_path = Path("MODEL_COMPARISON_REPORT.md")
    
    with open(report_path, 'w') as f:
        f.write("# Model Comparison Report - InsightDesk AI\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"**Winner**: {comparison_results['winner']}\n\n")
        f.write(f"**Assessment Target (≥0.85 weighted F1)**:\n")
        f.write(f"- XGBoost: {'✅ PASSED' if comparison_results['xgb_meets_target'] else '❌ FAILED'}\n")
        f.write(f"- TensorFlow: {'✅ PASSED' if comparison_results['tf_meets_target'] else '❌ FAILED'}\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | XGBoost | TensorFlow | Winner |\n")
        f.write("|--------|---------|------------|--------|\n")
        f.write(f"| Accuracy | {xgb_results['accuracy']:.4f} | {tf_results['accuracy']:.4f} | {'XGBoost' if xgb_results['accuracy'] > tf_results['accuracy'] else 'TensorFlow'} |\n")
        f.write(f"| Weighted F1 | {xgb_results['f1_weighted']:.4f} | {tf_results['f1_weighted']:.4f} | {'XGBoost' if xgb_results['f1_weighted'] > tf_results['f1_weighted'] else 'TensorFlow'} |\n")
        f.write(f"| Macro F1 | {xgb_results['f1_macro']:.4f} | {tf_results['f1_macro']:.4f} | {'XGBoost' if xgb_results['f1_macro'] > tf_results['f1_macro'] else 'TensorFlow'} |\n\n")
        
        f.write("## Model Architectures\n\n")
        f.write("### XGBoost\n")
        f.write("- **Type**: Gradient boosting decision trees\n")
        f.write("- **Features**: TF-IDF text vectors + categorical encoding + numeric features\n")
        f.write("- **Hyperparameters**: 200 estimators, max_depth=8, learning_rate=0.1\n\n")
        
        f.write("### TensorFlow\n")
        f.write("- **Type**: Multi-input neural network\n")
        f.write("- **Architecture**: Text (Embedding + BiLSTM) + Categorical (Embeddings) + Numeric (Dense)\n")
        f.write("- **Features**: Combined text processing with deep embeddings\n\n")
        
        f.write("## Recommendations\n\n")
        if comparison_results['winner'] == 'XGBoost':
            f.write("- **Primary Model**: Deploy XGBoost for production use\n")
            f.write("- **Reasoning**: Better performance, faster inference, more interpretable\n")
        else:
            f.write("- **Primary Model**: Deploy TensorFlow for production use\n")
            f.write("- **Reasoning**: Better performance, handles complex text patterns\n")
        
        f.write("- **Ensemble Option**: Consider combining both models for maximum accuracy\n")
        f.write("- **Monitoring**: Track performance drift on both models\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `models/xgb_category.pkl` - XGBoost model\n")
        f.write("- `models/tf_category_model.h5` - TensorFlow model\n")
        f.write("- `plots/model_comparison_confusion_matrices.png` - Confusion matrix comparison\n")
        f.write("- `plots/model_performance_comparison.png` - Performance metrics comparison\n")
    
    print(f"📄 Final report saved to {report_path}")


def main():
    """Main comparison pipeline."""
    print("🚀 Starting Comprehensive Model Training & Comparison")
    print("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    data_result = load_and_prepare_data()
    if data_result is None:
        return
    
    df, y, label_encoder = data_result
    
    # Train both models
    print("\n🏗️ Training both models...")
    xgb_results = train_xgboost_model(df, y, label_encoder)
    tf_results = train_tensorflow_model(df, y, label_encoder)
    
    # Compare models
    comparison_results = compare_models(xgb_results, tf_results, label_encoder)
    
    # Generate final report
    generate_final_report(comparison_results, xgb_results, tf_results)
    
    print("\n🎉 Training and comparison complete!")
    print(f"🏆 Winner: {comparison_results['winner']}")
    print("📊 Check plots/ directory for visualizations")
    print("📄 Check MODEL_COMPARISON_REPORT.md for detailed analysis")


if __name__ == "__main__":
    main()