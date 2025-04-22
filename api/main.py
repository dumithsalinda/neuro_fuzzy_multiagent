from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import hashlib

from src.plugins.registry import get_registered_plugins
from src.core.experiment.mlflow_tracker import ExperimentTracker
from src.core.experiment.result_analysis import ResultAnalyzer

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
    # Config versioning: hash config for reproducibility
    config_str = str(req.dict())
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    # MLflow tracking
    tracker = ExperimentTracker("neuro-fuzzy-experiments")
    run_id = tracker.start_run(run_name=f"exp_{exp_id}", params={"agent": req.agent, "environment": req.environment, **req.config, "config_version": config_hash}, tags={"exp_id": exp_id})
    # --- Automated Result Analysis ---
    analyzer = ResultAnalyzer(output_dir="/tmp")
    # For demo, use mock metrics. In real system, would use actual experiment results.
    metrics = {"accuracy": 0.95, "loss": 0.1}
    report_md = analyzer.generate_report(config={"agent": req.agent, "environment": req.environment, **req.config, "config_version": config_hash}, metrics=metrics, run_id=run_id)
    report_path = analyzer.save_report(report_md, filename=f"report_{exp_id}.md")
    tracker.log_artifact(report_path)
    tracker.end_run()
    experiment_status[exp_id] = {"status": "running", "result": None, "mlflow_run_id": run_id, "config_version": config_hash}
    return ExperimentStatus(id=exp_id, status="running")

@app.get("/experiment/status/{exp_id}", response_model=ExperimentStatus)
def get_experiment_status(exp_id: str):
    if exp_id not in experiment_status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    stat = experiment_status[exp_id]
    return ExperimentStatus(id=exp_id, status=stat["status"], result=stat["result"])

import asyncio
@app.websocket("/experiment/stream/{exp_id}")
async def experiment_status_stream(websocket: WebSocket, exp_id: str):
    await websocket.accept()
    try:
        for i in range(5):  # Demo: send 5 updates
            if exp_id not in experiment_status:
                await websocket.send_json({"error": "Experiment not found"})
                break
            stat = experiment_status[exp_id]
            await websocket.send_json({"id": exp_id, "status": stat["status"], "step": i})
            await asyncio.sleep(1)
        await websocket.close()
    except WebSocketDisconnect:
        pass
