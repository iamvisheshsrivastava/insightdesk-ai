from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()

def test_create_ticket():
    r = client.post("/tickets", json={"ticket_id": "TK-0001", "description": "Sample issue"})
    assert r.status_code == 200
    assert r.json()["ticket_id"] == "TK-0001"
