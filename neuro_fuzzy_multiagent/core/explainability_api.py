import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neuro_fuzzy_multiagent.core.explainability import (
    explain_agent_action,
    explain_env_transition,
)

app = FastAPI(
    title="Explainability API",
    description="APIs for explaining agent actions and environment transitions.",
)


class AgentExplainRequest(BaseModel):
    observation: Any = Field(..., description="Observation input to the agent.")
    action: Any = Field(..., description="Action taken by the agent.")


class EnvExplainRequest(BaseModel):
    state: Any = Field(..., description="Current environment state.")
    action: Any = Field(..., description="Action performed in the environment.")
    next_state: Any = Field(..., description="Next environment state after action.")


class ExplanationResponse(BaseModel):
    explanation: str


class DummyAgent:
    def explain(self, observation: Any, action: Any) -> dict:
        return {
            "explanation": f"Agent took action {action} based on observation {observation}."
        }


class DummyEnv:
    def explain(self, state: Any, action: Any, next_state: Any) -> dict:
        return {
            "explanation": f"Env transitioned from {state} to {next_state} via action {action}."
        }


agent = DummyAgent()
env = DummyEnv()


@app.post(
    "/explain/agent_action", response_model=ExplanationResponse, tags=["Explainability"]
)
def api_explain_agent_action(req: AgentExplainRequest) -> ExplanationResponse:
    """
    Explain why the agent took a particular action given an observation.
    """
    try:
        logging.info(f"Explaining agent action: {req}")
        result = explain_agent_action(agent, req.observation, req.action)
        return ExplanationResponse(**result)
    except Exception as e:
        logging.error(f"Error in agent action explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/explain/env_transition",
    response_model=ExplanationResponse,
    tags=["Explainability"],
)
def api_explain_env_transition(req: EnvExplainRequest) -> ExplanationResponse:
    """
    Explain how the environment transitioned from one state to another given an action.
    """
    try:
        logging.info(f"Explaining environment transition: {req}")
        result = explain_env_transition(env, req.state, req.action, req.next_state)
        return ExplanationResponse(**result)
    except Exception as e:
        logging.error(f"Error in environment transition explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
