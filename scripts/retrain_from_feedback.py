# scripts/retrain_from_feedback.py

"""
Reclassification feedback loop (issue #22).

Pulls AgentCorrection records already captured by the feedback system
(src/feedback/feedback_manager.py / feedback_storage.py) and uses the
(original ticket data, corrected category) pairs as additional labeled
training rows to re-fit the XGBoost classifier (same feature set used in
src/models/, via src/features/feature_builder.py).

Usage:
    python scripts/retrain_from_feedback.py
    python scripts/retrain_from_feedback.py --min-corrections 5 --force
    python scripts/retrain_from_feedback.py --data-sample-frac 0.05   # quick smoke test

Gating:
    By default the retrain only runs if there are at least `--min-corrections`
    corrections recorded in the last `--time-window-hours` hours (matching the
    weekly/periodic cadence suggested in the issue). Pass --force to bypass.

Output:
    - Overwrites models/xgb_category.pkl, models/feature_pipeline.pkl and
      models/label_encoder_category.pkl (after backing up the previous
      versions to models/backup_<timestamp>/).
    - Writes models/retrain_status.json with last retrain time, number of
      corrections used, and before/after validation accuracy, consumed by
      GET /feedback/retrain/status in src/api/main.py.
    - Logs a SYSTEM_EVENT feedback event so the retrain shows up alongside
      existing monitoring (src/monitoring/).
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feedback.feedback_manager import FeedbackManager  # noqa: E402
from src.feedback.feedback_models import FeedbackType, FeedbackSeverity  # noqa: E402
from src.features.feature_builder import build_feature_pipeline, prepare_features, NUMERIC_COLS, CATEGORICAL_COLS  # noqa: E402
from src.ingestion.data_loader import load_json_tickets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain_from_feedback")

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "support_tickets.json"
STATUS_PATH = MODELS_DIR / "retrain_status.json"

# Defaults applied to a correction's ticket_data for any column the model
# expects but the correction didn't capture - mirrors TicketInput's defaults
# in src/api/main.py so corrections stay compatible with the trained schema.
NUMERIC_DEFAULTS = {col: 0 for col in NUMERIC_COLS}
NUMERIC_DEFAULTS["affected_users"] = 1
CATEGORICAL_DEFAULTS = {
    "priority": "medium",
    "severity": "P2",
    "customer_tier": "standard",
    "product": "unknown",
    "product_module": "unknown",
    "region": "US",
}


def corrections_to_rows(corrections) -> pd.DataFrame:
    """Convert AgentCorrection records into training rows: (ticket text/fields, corrected category)."""
    rows = []
    for c in corrections:
        if not c.should_retrain or not c.corrected_label:
            continue
        ticket_data = dict(c.ticket_data or {})
        row = {
            "subject": ticket_data.get("subject", ""),
            "description": ticket_data.get("description", ""),
            "category": c.corrected_label,
        }
        for col, default in {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}.items():
            row[col] = ticket_data.get(col, default)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_existing_model(df: pd.DataFrame, y_encoded, label_encoder: LabelEncoder) -> float:
    """Accuracy of the currently-deployed model on a held-out split of the combined data, if present."""
    model_path = MODELS_DIR / "xgb_category.pkl"
    pipeline_path = MODELS_DIR / "feature_pipeline.pkl"
    existing_encoder_path = MODELS_DIR / "label_encoder_category.pkl"
    if not (model_path.exists() and pipeline_path.exists() and existing_encoder_path.exists()):
        return 0.0
    try:
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        existing_encoder = joblib.load(existing_encoder_path)

        df_prepared = prepare_features(df)
        X = pipeline.transform(df_prepared)

        # Map labels through the *existing* encoder; unseen categories can't be
        # scored against the old model, so drop those rows for this comparison.
        known_mask = df["category"].isin(existing_encoder.classes_)
        if known_mask.sum() == 0:
            return 0.0
        y_known = existing_encoder.transform(df.loc[known_mask, "category"])
        X_known = X[known_mask.values]

        y_pred = model.predict(X_known)
        return round(accuracy_score(y_known, y_pred), 4)
    except Exception as e:
        logger.warning(f"Could not evaluate existing model (treating as no baseline): {e}")
        return 0.0


def retrain(
    min_corrections: int,
    time_window_hours: int,
    force: bool,
    data_sample_frac: float,
    storage_dir: str = "feedback_data",
) -> dict:
    manager = FeedbackManager(storage_type="json", storage_config={"storage_dir": storage_dir})

    days_back = max(1, time_window_hours // 24 + 1)
    corrections = manager.get_corrections(days_back=days_back)

    if not force and len(corrections) < min_corrections:
        msg = (
            f"Only {len(corrections)} correction(s) in the last {time_window_hours}h "
            f"(need {min_corrections}). Skipping retrain. Use --force to override."
        )
        logger.info(msg)
        return {"status": "skipped", "reason": msg, "corrections_used": len(corrections)}

    correction_rows = corrections_to_rows(corrections)
    logger.info(f"Loaded {len(correction_rows)} usable correction row(s) out of {len(corrections)} total corrections.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Base training data not found: {DATA_PATH}")

    base_df = load_json_tickets(DATA_PATH)
    if data_sample_frac < 1.0:
        base_df = base_df.sample(frac=data_sample_frac, random_state=42)
        logger.info(f"Sampled base training data down to {len(base_df)} rows (data_sample_frac={data_sample_frac}).")

    needed_cols = ["subject", "description", "category"] + NUMERIC_COLS + CATEGORICAL_COLS
    for col in needed_cols:
        if col not in base_df.columns:
            base_df[col] = NUMERIC_DEFAULTS.get(col, CATEGORICAL_DEFAULTS.get(col, ""))
    base_df = base_df[needed_cols].copy()

    if not correction_rows.empty:
        combined_df = pd.concat([base_df, correction_rows[needed_cols]], ignore_index=True)
    else:
        combined_df = base_df

    combined_df = combined_df.dropna(subset=["category"]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined_df["category"])

    before_accuracy = evaluate_existing_model(combined_df, y, label_encoder)

    df_prepared = prepare_features(combined_df)
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df_prepared)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from xgboost import XGBClassifier

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
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    after_accuracy = round(accuracy_score(y_test, y_pred), 4)
    after_f1 = round(f1_score(y_test, y_pred, average="weighted"), 4)

    # Back up the previous model artifacts before overwriting.
    MODELS_DIR.mkdir(exist_ok=True)
    backup_dir = MODELS_DIR / f"backup_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    existing_artifacts = ["xgb_category.pkl", "feature_pipeline.pkl", "label_encoder_category.pkl"]
    if any((MODELS_DIR / f).exists() for f in existing_artifacts):
        backup_dir.mkdir(exist_ok=True)
        for f in existing_artifacts:
            src = MODELS_DIR / f
            if src.exists():
                shutil.copy2(src, backup_dir / f)
        logger.info(f"Backed up previous model artifacts to {backup_dir}")

    joblib.dump(model, MODELS_DIR / "xgb_category.pkl")
    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder_category.pkl")
    logger.info("Saved retrained model artifacts to models/")

    status = {
        "status": "retrained",
        "last_retrain_time": datetime.utcnow().isoformat() + "Z",
        "corrections_used": len(correction_rows),
        "total_corrections_considered": len(corrections),
        "training_rows": len(combined_df),
        "before_accuracy": before_accuracy,
        "after_accuracy": after_accuracy,
        "after_f1_weighted": after_f1,
        "accuracy_delta": round(after_accuracy - before_accuracy, 4) if before_accuracy else None,
        "backup_dir": str(backup_dir) if backup_dir.exists() else None,
    }

    STATUS_PATH.write_text(json.dumps(status, indent=2))

    manager.record_feedback_event(
        event_type=FeedbackType.SYSTEM_EVENT,
        ticket_id="retrain_from_feedback",
        event_data=status,
        severity=FeedbackSeverity.LOW,
    )

    logger.info(f"Retrain complete: {status}")
    return status


def main():
    parser = argparse.ArgumentParser(description="Retrain the XGBoost classifier from agent feedback corrections.")
    parser.add_argument("--min-corrections", type=int, default=10, help="Minimum corrections required to trigger a retrain.")
    parser.add_argument("--time-window-hours", type=int, default=24 * 7, help="Look-back window (hours) for counting corrections.")
    parser.add_argument("--force", action="store_true", help="Retrain even if the minimum correction threshold isn't met.")
    parser.add_argument("--data-sample-frac", type=float, default=1.0, help="Sample fraction of the base training data (for quick smoke tests).")
    parser.add_argument("--storage-dir", type=str, default="feedback_data", help="Feedback storage directory.")
    args = parser.parse_args()

    result = retrain(
        min_corrections=args.min_corrections,
        time_window_hours=args.time_window_hours,
        force=args.force,
        data_sample_frac=args.data_sample_frac,
        storage_dir=args.storage_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
