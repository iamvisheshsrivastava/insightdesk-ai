# src/features/feature_builder.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_COLS = [
    "resolution_time_hours", "previous_tickets", "account_age_days",
    "account_monthly_value", "affected_users", "ticket_text_length",
    "response_count", "attachments_count"
]

CATEGORICAL_COLS = [
    "priority", "severity", "customer_tier", "product", "product_module", "region"
]

TEXT_COLS = ["subject", "description"]


def build_feature_pipeline():
    """Return a sklearn pipeline for feature engineering."""

    # Text features (TF-IDF on subject + description concatenated)
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english"
    )

    # One-hot encoding for categorical variables
    one_hot = OneHotEncoder(handle_unknown="ignore")

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", one_hot, CATEGORICAL_COLS),
            ("text", tfidf, "text_concat"),
        ]
    )

    return preprocessor


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features from raw tickets DataFrame."""
    df = df.copy()
    df["text_concat"] = df["subject"].fillna("") + " " + df["description"].fillna("")
    return df
