# scripts/build_features.py

import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm

from src.ingestion.data_loader import load_json_tickets
from src.features.feature_builder import build_feature_pipeline, prepare_features


def main():
    data_path = Path("data/support_tickets.json")

    # Load raw tickets
    df = load_json_tickets(data_path)

    # Prepare text column
    df = prepare_features(df)

    # Build pipeline
    pipeline = build_feature_pipeline()

    # Show progress message
    print("⚙️ Building features (this may take ~1–2 minutes)...")

    # Wrap fit_transform with tqdm so you know it's running
    with tqdm(total=1, desc="TF-IDF + Encoding") as pbar:
        X = pipeline.fit_transform(df)
        pbar.update(1)

    print(f"✅ Features built: shape = {X.shape}")

    # Save pipeline
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/feature_pipeline.pkl")
    print("💾 Saved feature pipeline → models/feature_pipeline.pkl")

    # Save features as joblib (sparse format preserved)
    joblib.dump(X, "data/features.joblib")
    print("💾 Saved feature matrix → data/features.joblib")


if __name__ == "__main__":
    main()
