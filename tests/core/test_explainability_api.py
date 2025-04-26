from fastapi.testclient import TestClient

from neuro_fuzzy_multiagent.core.explainability_api import app


def test_agent_action_explanation():
    client = TestClient(app)
    resp = client.post("/explain/agent_action", json={"observation": 5, "action": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert "Agent took action" in data["explanation"]


def test_env_transition_explanation():
    client = TestClient(app)
    resp = client.post(
        "/explain/env_transition", json={"state": 0, "action": 1, "next_state": 1}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Env transitioned from" in data["explanation"]
