from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../api')))
from api.main import app

client = TestClient(app)

def test_list_plugins():
    r = client.get("/plugins/")
    assert r.status_code == 200
    data = r.json()
    assert "agent" in data
    assert "environment" in data

def test_list_agents():
    r = client.get("/agents/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_list_envs():
    r = client.get("/environments/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_submit_and_status():
    req = {"agent": "DummyAgent", "environment": "SimpleDiscreteEnv", "config": {}}
    r = client.post("/experiment/submit", json=req)
    assert r.status_code == 200
    exp_id = r.json()["id"]
    stat = client.get(f"/experiment/status/{exp_id}")
    assert stat.status_code == 200
    assert stat.json()["status"] == "running"
