from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid

from src.plugins.registry import get_registered_plugins

app = FastAPI(title="Neuro-Fuzzy Multiagent Backend API")

# In-memory experiment status store for demo
experiment_status = {}

class ExperimentRequest(BaseModel):
    agent: str
    environment: str
    config: Dict[str, Any] = {}

class ExperimentStatus(BaseModel):
    id: str
    status: str
    result: Any = None

@app.get("/plugins/", response_model=Dict[str, List[str]])
def list_plugins():
    """List all registered plugins by type."""
    return {ptype: list(get_registered_plugins(ptype).keys()) for ptype in ["agent", "environment", "sensor", "actuator"]}

@app.get("/agents/", response_model=List[str])
def list_agents():
    return list(get_registered_plugins("agent").keys())

@app.get("/environments/", response_model=List[str])
def list_envs():
    return list(get_registered_plugins("environment").keys())

@app.post("/experiment/submit", response_model=ExperimentStatus)
def submit_experiment(req: ExperimentRequest):
    exp_id = str(uuid.uuid4())
    # For now, just mock status
    experiment_status[exp_id] = {"status": "running", "result": None}
    # In real system, would launch async experiment
    return ExperimentStatus(id=exp_id, status="running")

@app.get("/experiment/status/{exp_id}", response_model=ExperimentStatus)
def get_experiment_status(exp_id: str):
    if exp_id not in experiment_status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    stat = experiment_status[exp_id]
    return ExperimentStatus(id=exp_id, status=stat["status"], result=stat["result"])
