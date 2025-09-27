# src/ingestion/data_loader.py

import json
import pandas as pd
from pathlib import Path
from typing import List, Union

# ✅ Full schema based on your dataset (~60 columns)
REQUIRED_FIELDS: List[str] = [
    "ticket_id", "created_at", "updated_at", "customer_id", "customer_tier",
    "organization_id", "product", "product_version", "product_module",
    "category", "subcategory", "priority", "severity", "channel",
    "subject", "description", "error_logs", "stack_trace",
    "customer_sentiment", "previous_tickets", "resolution",
    "resolution_code", "resolved_at", "resolution_time_hours",
    "resolution_attempts", "agent_id", "agent_experience_months",
    "agent_specialization", "agent_actions", "escalated",
    "escalation_reason", "transferred_count", "satisfaction_score",
    "feedback_text", "resolution_helpful", "tags", "related_tickets",
    "kb_articles_viewed", "kb_articles_helpful", "environment",
    "account_age_days", "account_monthly_value", "similar_issues_last_30_days",
    "product_version_age_days", "known_issue", "bug_report_filed",
    "resolution_template_used", "auto_suggested_solutions",
    "auto_suggestion_accepted", "ticket_text_length", "response_count",
    "attachments_count", "contains_error_code", "contains_stack_trace",
    "business_impact", "affected_users", "weekend_ticket", "after_hours",
    "language", "region"
]

DATETIME_FIELDS: List[str] = [
    "created_at", "updated_at", "resolved_at"
]


def load_json_tickets(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load support tickets ensuring 1 row = 1 ticket.
    Handles both JSON arrays and JSON Lines (JSONL).
    Converts list/dict fields into JSON strings.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    # --- Try JSONL first ---
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        # --- Fallback to standard JSON ---
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "tickets" in data:
            data = data["tickets"]
        elif not isinstance(data, list):
            raise ValueError("❌ Unexpected JSON structure")
        df = pd.DataFrame(data)

    # --- Ensure list/dict columns are stored as JSON strings ---
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )

    # --- Schema validation ---
    missing_cols = [col for col in REQUIRED_FIELDS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")

    # --- Datetime parsing ---
    for col in DATETIME_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"✅ Loaded {df.shape[0]} tickets with {df.shape[1]} columns")
    print(f"📌 Unique ticket_ids: {df['ticket_id'].nunique()}")
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"💾 Approx memory usage: {mem:.2f} MB")

    return df


def save_parquet(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Save DataFrame to Parquet format for fast re-use."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Data saved to {output_path}")
