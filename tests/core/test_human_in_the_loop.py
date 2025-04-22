import pytest
from fastapi.testclient import TestClient
from src.core.human_in_the_loop import app, approval_data

def test_human_approval_flow():
    client = TestClient(app)
    # Simulate pending action
    approval_data.clear()
    approval_data.update({
        "observation": 42,
        "proposed_action": 1,
        "approved": None,
        "modified_action": None
    })
    # Human fetches pending action
    resp = client.get("/human/pending_action")
    assert resp.status_code == 200
    data = resp.json()
    assert data["observation"] == 42
    # Human approves action
    resp2 = client.post("/human/approve_action", json={
        "observation": 42,
        "proposed_action": 1,
        "approved": True,
        "modified_action": None
    })
    assert resp2.status_code == 200
    assert approval_data["approved"] is True
