# scripts/train_xgboost.py

import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def main():
    data_path = Path("data/support_tickets.json")
    features_path = Path("data/features.joblib")
    pipeline_path = Path("models/feature_pipeline.pkl")

    # --- Load raw data (for labels) ---
    from src.ingestion.data_loader import load_json_tickets
    df = load_json_tickets(data_path)

    # --- Load feature matrix & pipeline ---
    X = joblib.load(features_path)
    pipeline = joblib.load(pipeline_path)

    # --- Target variable (category) ---
    y_raw = df["category"]

    # Encode category labels → integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Save the encoder for inference later
    Path("models").mkdir(exist_ok=True)
    joblib.dump(le, "models/label_encoder_category.pkl")
    print("💾 Saved label encoder → models/label_encoder_category.pkl")

    # --- Train/validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Define XGBoost model ---
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    # --- Train ---
    print("⚙️ Training XGBoost model...")
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_val)
    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    print(f"✅ Weighted F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")

    # --- Save model ---
    model_path = "models/xgb_category.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Saved XGBoost model → {model_path}")


if __name__ == "__main__":
    main()
