# scripts/train_tensorflow.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import MLflow, but don't fail if not available
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available. Skipping experiment tracking.")


# Configuration
CONFIG = {
    "max_vocab_size": 50000,
    "max_sequence_length": 256,
    "embedding_dim": 128,
    "lstm_units": 64,
    "dense_units": 128,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
}

# Feature definitions
TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]
CATEGORICAL_COLS = ["product", "channel", "priority", "severity", "customer_tier", "region"]
NUMERIC_COLS = [
    "previous_tickets", "account_age_days", "account_monthly_value",
    "ticket_text_length", "response_count", "attachments_count",
    "affected_users", "resolution_time_hours"
]


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare data with proper preprocessing."""
    from src.ingestion.data_loader import load_json_tickets
    
    print("📊 Loading data...")
    df = load_json_tickets(data_path)
    
    # Combine text fields
    text_fields = []
    for col in TEXT_COLS:
        if col in df.columns:
            text_fields.append(df[col].fillna(""))
        else:
            text_fields.append(pd.Series([""] * len(df)))
    
    df["combined_text"] = " ".join(text_fields).apply(lambda x: " ".join(x) if isinstance(x, (list, tuple)) else str(x))
    
    # Handle missing values in categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = "unknown"
    
    # Handle missing values in numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    
    print(f"✅ Data prepared: {df.shape[0]} samples")
    return df


def create_preprocessors(df: pd.DataFrame):
    """Create and fit preprocessors for different input types."""
    
    # Text vectorizer
    text_vectorizer = tf.keras.utils.text_dataset_from_tensor_slices(
        df["combined_text"].values
    )
    
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=CONFIG["max_vocab_size"],
        sequence_length=CONFIG["max_sequence_length"],
        output_mode="int"
    )
    
    vectorizer.adapt(text_vectorizer)
    
    # Categorical encoders
    categorical_encoders = {}
    for col in CATEGORICAL_COLS:
        encoder = LabelEncoder()
        encoder.fit(df[col])
        categorical_encoders[col] = encoder
    
    # Numeric normalizer (simple min-max scaling)
    numeric_stats = {}
    for col in NUMERIC_COLS:
        numeric_stats[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean(),
            "std": df[col].std()
        }
    
    return vectorizer, categorical_encoders, numeric_stats


def preprocess_data(df: pd.DataFrame, vectorizer, categorical_encoders, numeric_stats):
    """Preprocess data using fitted preprocessors."""
    
    # Text preprocessing
    text_data = vectorizer(df["combined_text"].values)
    
    # Categorical preprocessing
    categorical_data = {}
    for col in CATEGORICAL_COLS:
        encoded = categorical_encoders[col].transform(df[col])
        categorical_data[col] = encoded
    
    # Numeric preprocessing (standardization)
    numeric_data = np.zeros((len(df), len(NUMERIC_COLS)))
    for i, col in enumerate(NUMERIC_COLS):
        values = df[col].values
        std = numeric_stats[col]["std"]
        if std > 0:
            normalized = (values - numeric_stats[col]["mean"]) / std
        else:
            normalized = values - numeric_stats[col]["mean"]
        numeric_data[:, i] = normalized
    
    return text_data, categorical_data, numeric_data


def build_model(num_classes: int, categorical_encoders: dict, config: dict = CONFIG):
    """Build multi-input TensorFlow model."""
    
    # Text input branch
    text_input = layers.Input(shape=(config["max_sequence_length"],), name="text_input")
    text_embedding = layers.Embedding(
        input_dim=config["max_vocab_size"],
        output_dim=config["embedding_dim"],
        mask_zero=True
    )(text_input)
    
    # BiLSTM for text processing
    text_lstm = layers.Bidirectional(
        layers.LSTM(config["lstm_units"], dropout=config["dropout_rate"])
    )(text_embedding)
    text_dense = layers.Dense(config["dense_units"], activation="relu")(text_lstm)
    text_dropout = layers.Dropout(config["dropout_rate"])(text_dense)
    
    # Categorical inputs
    categorical_inputs = []
    categorical_embeddings = []
    
    for col in CATEGORICAL_COLS:
        vocab_size = len(categorical_encoders[col].classes_)
        embedding_dim = min(50, (vocab_size + 1) // 2)  # Rule of thumb for embedding size
        
        cat_input = layers.Input(shape=(), name=f"{col}_input")
        cat_embedding = layers.Embedding(vocab_size, embedding_dim)(cat_input)
        cat_flatten = layers.Flatten()(cat_embedding)
        
        categorical_inputs.append(cat_input)
        categorical_embeddings.append(cat_flatten)
    
    # Combine categorical embeddings
    if categorical_embeddings:
        categorical_concat = layers.Concatenate()(categorical_embeddings)
        categorical_dense = layers.Dense(config["dense_units"] // 2, activation="relu")(categorical_concat)
        categorical_dropout = layers.Dropout(config["dropout_rate"])(categorical_dense)
    else:
        categorical_dropout = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], config["dense_units"] // 2)))(text_input)
    
    # Numeric input
    numeric_input = layers.Input(shape=(len(NUMERIC_COLS),), name="numeric_input")
    numeric_dense = layers.Dense(config["dense_units"] // 2, activation="relu")(numeric_input)
    numeric_dropout = layers.Dropout(config["dropout_rate"])(numeric_dense)
    
    # Combine all branches
    combined = layers.Concatenate()([text_dropout, categorical_dropout, numeric_dropout])
    combined_dense = layers.Dense(config["dense_units"], activation="relu")(combined)
    combined_dropout = layers.Dropout(config["dropout_rate"])(combined_dense)
    
    # Output layer
    output = layers.Dense(num_classes, activation="softmax", name="category_output")(combined_dropout)
    
    # Create model
    inputs = [text_input] + categorical_inputs + [numeric_input]
    model = Model(inputs=inputs, outputs=output)
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, class_weights, config: dict = CONFIG):
    """Train the model with callbacks."""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config["reduce_lr_patience"],
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, label_encoder, save_plots: bool = True):
    """Evaluate model and generate reports."""
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\n📊 Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # F1 Score
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"✅ Weighted F1 Score: {f1_weighted:.4f}")
    print(f"✅ Macro F1 Score: {f1_macro:.4f}")
    
    # Confusion Matrix
    if save_plots:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap="Blues"
        )
        plt.title("Confusion Matrix - TensorFlow Category Classifier")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "tf_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Confusion matrix saved to {plots_dir / 'tf_confusion_matrix.png'}")
    
    return {
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "classification_report": report
    }


def save_model_artifacts(model, vectorizer, categorical_encoders, numeric_stats, label_encoder):
    """Save all model artifacts for inference."""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save TensorFlow model
    model.save(models_dir / "tf_category_model.h5")
    print(f"💾 Saved TensorFlow model → {models_dir / 'tf_category_model.h5'}")
    
    # Save vectorizer config
    vectorizer_config = {
        "vocabulary": vectorizer.get_vocabulary(),
        "max_tokens": CONFIG["max_vocab_size"],
        "sequence_length": CONFIG["max_sequence_length"]
    }
    
    with open(models_dir / "tf_text_vectorizer.json", "w") as f:
        json.dump(vectorizer_config, f)
    print(f"💾 Saved text vectorizer → {models_dir / 'tf_text_vectorizer.json'}")
    
    # Save categorical encoders
    joblib.dump(categorical_encoders, models_dir / "tf_categorical_encoders.pkl")
    print(f"💾 Saved categorical encoders → {models_dir / 'tf_categorical_encoders.pkl'}")
    
    # Save numeric stats
    with open(models_dir / "tf_numeric_stats.json", "w") as f:
        json.dump(numeric_stats, f)
    print(f"💾 Saved numeric stats → {models_dir / 'tf_numeric_stats.json'}")
    
    # Save label encoder (reuse existing if same)
    joblib.dump(label_encoder, models_dir / "tf_label_encoder_category.pkl")
    print(f"💾 Saved label encoder → {models_dir / 'tf_label_encoder_category.pkl'}")


def main():
    """Main training pipeline with comprehensive MLflow tracking."""
    import time
    start_time = time.time()
    
    # Setup MLflow
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
        
        # Start MLflow run
        mlflow.start_run(run_name="tensorflow_category_classifier")
        
        # Log configuration parameters
        mlflow.log_params({
            "model_type": "tensorflow",
            "architecture": "multi_input_bilstm",
            **CONFIG
        })
        
        # Log feature configuration
        mlflow.log_params({
            "text_features": "+".join(TEXT_COLS),
            "categorical_features": "+".join(CATEGORICAL_COLS),
            "numeric_features": "+".join(NUMERIC_COLS),
            "num_text_cols": len(TEXT_COLS),
            "num_categorical_cols": len(CATEGORICAL_COLS),
            "num_numeric_cols": len(NUMERIC_COLS)
        })
    
    try:
        data_path = Path("data/support_tickets.json")
        
        # Load and prepare data
        print("📊 Loading and preparing data...")
        df = load_and_prepare_data(data_path)
        
        # Prepare target variable
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["category"])
        num_classes = len(label_encoder.classes_)
        
        print(f"📊 Classes: {num_classes} ({list(label_encoder.classes_)})")
        
        # Log data information
        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                "num_samples": len(df),
                "num_classes": num_classes,
                "class_names": ",".join(label_encoder.classes_)
            })
            
            # Log class distribution
            class_counts = pd.Series(y).value_counts().sort_index()
            for i, count in enumerate(class_counts):
                mlflow.log_metric(f"class_{label_encoder.classes_[i]}_count", count)
                mlflow.log_metric(f"class_{label_encoder.classes_[i]}_percentage", count/len(y)*100)
        
        # Create preprocessors
        print("⚙️ Creating preprocessors...")
        vectorizer, categorical_encoders, numeric_stats = create_preprocessors(df)
        
        # Preprocess data
        print("⚙️ Preprocessing data...")
        text_data, categorical_data, numeric_data = preprocess_data(
            df, vectorizer, categorical_encoders, numeric_stats
        )
        
        # Prepare input data
        X = [text_data] + [categorical_data[col] for col in CATEGORICAL_COLS] + [numeric_data]
        
        # Proper data splitting with stratification
        print("📊 Splitting data...")
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        test_size = len(df) - train_size - val_size
        
        # Use stratified splitting
        X_temp, X_test, y_temp, y_test = train_test_split(
            list(range(len(df))), y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(train_size + val_size), 
            random_state=42, stratify=y_temp
        )
        
        # Extract data using indices
        X_train_data = [x[X_train] for x in X]
        X_val_data = [x[X_val] for x in X]
        X_test_data = [x[X_test] for x in X]
        
        print(f"📊 Data splits - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        
        # Log data splits
        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                "train_samples": len(y_train),
                "val_samples": len(y_val),
                "test_samples": len(y_test),
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15
            })
        
        # Compute class weights for imbalanced data
        class_weights_array = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        # Log class weights
        if MLFLOW_AVAILABLE:
            for i, weight in class_weights.items():
                mlflow.log_metric(f"class_weight_{label_encoder.classes_[i]}", weight)
        
        # Build model
        print("🏗️ Building model...")
        model = build_model(num_classes, categorical_encoders, CONFIG)
        
        # Log model architecture info
        if MLFLOW_AVAILABLE:
            mlflow.log_param("total_params", model.count_params())
            mlflow.log_param("trainable_params", sum([tf.reduce_prod(var.shape) for var in model.trainable_variables]))
        
        model.summary()
        
        # Train model
        print("⚙️ Training model...")
        training_start = time.time()
        history = train_model(model, X_train_data, y_train, X_val_data, y_val, class_weights, CONFIG)
        training_time = time.time() - training_start
        
        # Log training time
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate model
        print("📊 Evaluating model...")
        metrics = evaluate_model(model, X_test_data, y_test, label_encoder, save_plots=True)
        
        # Calculate additional metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        y_pred = np.argmax(model.predict(X_test_data, verbose=0), axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Log comprehensive metrics to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1],
                "final_train_accuracy": history.history["accuracy"][-1],
                "final_val_accuracy": history.history["val_accuracy"][-1],
                "min_val_loss": min(history.history["val_loss"]),
                "max_val_accuracy": max(history.history["val_accuracy"]),
                "epochs_trained": len(history.history["loss"]),
                "total_time_seconds": time.time() - start_time
            })
            
            # Log training history plots
            try:
                # Create training history plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Loss plot
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                # Accuracy plot
                ax2.plot(history.history['accuracy'], label='Training Accuracy')
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title('Model Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                
                plt.tight_layout()
                
                # Save and log plot
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                history_plot_path = plots_dir / "tf_training_history.png"
                plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(str(history_plot_path), "plots")
                plt.close()
                
            except Exception as e:
                print(f"⚠️ Failed to create training history plot: {e}")
            
            # Log model and artifacts
            mlflow.tensorflow.log_model(model, "tensorflow_model")
            
            # Log classification report
            try:
                report_path = Path("models/tf_classification_report.json")
                with open(report_path, 'w') as f:
                    json.dump(metrics["classification_report"], f, indent=2)
                mlflow.log_artifact(str(report_path), "reports")
            except Exception as e:
                print(f"⚠️ Failed to save classification report: {e}")
        
        # Save artifacts
        save_model_artifacts(model, vectorizer, categorical_encoders, numeric_stats, label_encoder)
        
        # Print final results
        total_time = time.time() - start_time
        print(f"\n� TensorFlow Training Complete!")
        print("=" * 50)
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ Weighted F1 Score: {metrics['f1_weighted']:.4f}")
        print(f"✅ Macro F1 Score: {metrics['f1_macro']:.4f}")
        print(f"✅ Training Time: {training_time:.2f} seconds")
        print(f"✅ Total Time: {total_time:.2f} seconds")
        
        # Check target achievement
        target_f1 = 0.85
        if metrics['f1_weighted'] >= target_f1:
            print(f"🎯 SUCCESS: Weighted F1 ({metrics['f1_weighted']:.4f}) meets target (≥{target_f1})")
        else:
            print(f"⚠️  WARNING: Weighted F1 ({metrics['f1_weighted']:.4f}) below target (≥{target_f1})")
        
        return {
            "accuracy": accuracy,
            "f1_weighted": metrics["f1_weighted"],
            "f1_macro": metrics["f1_macro"],
            "training_time": training_time,
            "total_time": total_time
        }
        
    except Exception as e:
        print(f"❌ TensorFlow training failed: {e}")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
        raise
        
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()


if __name__ == "__main__":
    main()