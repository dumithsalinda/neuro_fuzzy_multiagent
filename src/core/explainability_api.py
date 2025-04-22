from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
from src.core.explainability import explain_agent_action, explain_env_transition

app = FastAPI()

class AgentExplainRequest(BaseModel):
    observation: Any
    action: Any

class EnvExplainRequest(BaseModel):
    state: Any
    action: Any
    next_state: Any

# Dummy references for demonstration (replace with actual agent/env in integration)
class DummyAgent:
    def explain(self, observation, action):
        return {"explanation": f"Agent took action {action} based on observation {observation}."}

class DummyEnv:
    def explain(self, state, action, next_state):
        return {"explanation": f"Env transitioned from {state} to {next_state} via action {action}."}

agent = DummyAgent()
env = DummyEnv()

@app.post("/explain/agent_action")
def api_explain_agent_action(req: AgentExplainRequest):
    return explain_agent_action(agent, req.observation, req.action)

@app.post("/explain/env_transition")
def api_explain_env_transition(req: EnvExplainRequest):
    return explain_env_transition(env, req.state, req.action, req.next_state)
