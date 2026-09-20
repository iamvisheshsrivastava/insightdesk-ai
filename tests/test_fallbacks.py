"""Tests for classifier heuristics and demo fallbacks (issues #9, #16)."""
from src.api.main import (
    TicketInput, keyword_category_hint, generate_mock_classification,
    annotate_low_confidence, generate_demo_anomalies,
)


def test_keyword_hint_login_is_account_management():
    t = TicketInput(subject="Cannot log in", description="invalid password error on my account")
    assert keyword_category_hint(t) == "Account Management"
    assert generate_mock_classification(t) == "Account Management"


def test_keyword_hint_none_when_no_match():
    assert keyword_category_hint({"subject": "hello", "description": "zzz"}) is None


def test_text_length_derived():
    t = TicketInput(subject="abc", description="defg")
    assert t.ticket_text_length == 7


def test_low_confidence_annotation():
    preds = {"xgboost": {"predicted_category": "Security", "confidence": 0.2}}
    annotate_low_confidence(preds, {"subject": "password reset", "description": "login fails"})
    assert preds["xgboost"]["low_confidence"] is True
    assert preds["xgboost"]["keyword_suggestion"] == "Account Management"
    hi = {"xgboost": {"confidence": 0.9}}
    annotate_low_confidence(hi, {})
    assert "low_confidence" not in hi["xgboost"]


def test_demo_anomalies_shape():
    demo = generate_demo_anomalies(7)
    assert demo and all(a["demo"] and a["severity"] and a["type"] for a in demo)
