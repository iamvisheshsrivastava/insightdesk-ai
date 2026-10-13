# tests/test_retrain_from_feedback.py

"""
Smoke test for scripts/retrain_from_feedback.py (issue #28).

Exercises the full retrain() path - including the actual XGBClassifier
model.fit() call - against a tiny synthetic dataset, so regressions in the
fit/eval/backup/status-writing logic are caught even without the full
production dataset.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("xgboost")

import scripts.retrain_from_feedback as retrain_mod  # noqa: E402
from src.feedback.feedback_manager import FeedbackManager  # noqa: E402
from src.feedback.feedback_models import FeedbackSeverity, PredictionQuality  # noqa: E402

CATEGORIES = ["billing", "technical issue", "account"]

# load_json_tickets() (src/ingestion/data_loader.py) validates the full ~60
# column production schema, so the synthetic base dataset needs every
# required column present (even if most are irrelevant to this smoke test).
_BASE_TICKET_DEFAULTS = {
    "ticket_id": "",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "customer_id": "cust-1",
    "organization_id": "org-1",
    "product_version": "1.0",
    "subcategory": "general",
    "channel": "email",
    "error_logs": "",
    "stack_trace": "",
    "customer_sentiment": "neutral",
    "resolution": "",
    "resolution_code": "",
    "resolved_at": "2024-01-02T00:00:00",
    "resolution_attempts": 1,
    "agent_id": "agent-1",
    "agent_experience_months": 12,
    "agent_specialization": "general",
    "agent_actions": "",
    "escalated": False,
    "escalation_reason": "",
    "transferred_count": 0,
    "satisfaction_score": 5,
    "feedback_text": "",
    "resolution_helpful": True,
    "tags": "",
    "related_tickets": "",
    "kb_articles_viewed": 0,
    "kb_articles_helpful": 0,
    "environment": "prod",
    "similar_issues_last_30_days": 0,
    "product_version_age_days": 30,
    "known_issue": False,
    "bug_report_filed": False,
    "resolution_template_used": False,
    "auto_suggested_solutions": "",
    "auto_suggestion_accepted": False,
    "contains_error_code": False,
    "contains_stack_trace": False,
    "business_impact": "low",
    "weekend_ticket": False,
    "after_hours": False,
    "language": "en",
}


def _make_ticket(i: int, category: str) -> dict:
    ticket = dict(_BASE_TICKET_DEFAULTS)
    ticket.update({
        "ticket_id": f"TK-{category[:3]}-{i}",
        "subject": f"{category} issue number {i}",
        "description": f"Customer reports a {category} related problem, details {i}.",
        "category": category,
        "priority": "medium",
        "severity": "P2",
        "customer_tier": "standard",
        "product": "core",
        "product_module": "unknown",
        "region": "US",
        "previous_tickets": 0,
        "account_age_days": 10,
        "account_monthly_value": 0.0,
        "response_count": 0,
        "attachments_count": 0,
        "affected_users": 1,
        "resolution_time_hours": 0.0,
        "ticket_text_length": 0,
    })
    return ticket


def test_retrain_smoke(tmp_path, monkeypatch):
    """retrain(force=True) should run end-to-end (including model.fit()) on a
    tiny synthetic dataset and write out fresh model artifacts + a status file."""
    # Small, class-balanced synthetic base dataset so the stratified
    # train/test split inside retrain() has enough rows per class.
    tickets = [
        _make_ticket(i, category)
        for category in CATEGORIES
        for i in range(12)
    ]

    data_path = tmp_path / "support_tickets.json"
    # Pretty-printed (multi-line) so pandas' JSONL auto-detection
    # (pd.read_json(..., lines=True), tried first in load_json_tickets)
    # correctly fails and falls back to parsing this as a single JSON array -
    # matching how the real data/support_tickets.json file is formatted.
    data_path.write_text(json.dumps(tickets, indent=2))

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr(retrain_mod, "DATA_PATH", data_path)
    monkeypatch.setattr(retrain_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(retrain_mod, "STATUS_PATH", models_dir / "retrain_status.json")

    storage_dir = tmp_path / "feedback_data"
    manager = FeedbackManager(storage_type="json", storage_config={"storage_dir": str(storage_dir)})
    manager.record_agent_correction(
        ticket_id="TK-SMOKE-1",
        agent_id="agent-1",
        original_prediction="billing",
        original_confidence=0.4,
        model_type="xgb_category",
        corrected_label="account",
        correction_reason="misclassified",
        prediction_quality=PredictionQuality.POOR,
        severity=FeedbackSeverity.MEDIUM,
        ticket_data=_make_ticket(99, "account"),
        should_retrain=True,
    )

    result = retrain_mod.retrain(
        min_corrections=1,
        time_window_hours=24,
        force=True,
        data_sample_frac=1.0,
        storage_dir=str(storage_dir),
    )

    assert result["status"] == "retrained"
    assert result["corrections_used"] == 1
    assert 0.0 <= result["after_accuracy"] <= 1.0
    assert (models_dir / "xgb_category.pkl").exists()
    assert (models_dir / "feature_pipeline.pkl").exists()
    assert (models_dir / "label_encoder_category.pkl").exists()
    assert retrain_mod.STATUS_PATH.exists()


def test_corrections_to_rows_derives_ticket_text_length():
    """corrections_to_rows() must compute ticket_text_length from the actual
    stored ticket text (issue #27), not leave it at the 0 default."""
    from dataclasses import dataclass
    from typing import Any, Dict, Optional

    @dataclass
    class FakeCorrection:
        should_retrain: bool
        corrected_label: Optional[str]
        ticket_data: Dict[str, Any]

    ticket_data = {
        "subject": "abc",          # len 3
        "description": "defgh",    # len 5
        "error_logs": "ij",        # len 2
        "stack_trace": "k",        # len 1
    }
    correction = FakeCorrection(
        should_retrain=True,
        corrected_label="billing",
        ticket_data=ticket_data,
    )

    rows = retrain_mod.corrections_to_rows([correction])

    assert len(rows) == 1
    assert rows.iloc[0]["ticket_text_length"] == 3 + 5 + 2 + 1
