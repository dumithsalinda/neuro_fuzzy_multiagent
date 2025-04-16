from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Any
import numpy as np
import logging
from src.core.agent import Agent
from src.core.tabular_q_agent import TabularQLearningAgent
from src.core.dqn_agent import DQNAgent
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.env.multiagent_gridworld import MultiAgentGridworldEnv

app = FastAPI()

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_api")

# --- Simple API key authentication ---
API_KEY = "mysecretkey"  # Change this in production!
api_key_header = APIKeyHeader(name="X-API-Key")
def verify_api_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Dynamic agent/env selection ---
AGENT_TYPES = ["Tabular Q-Learning", "DQN RL", "Neuro-Fuzzy"]
ENV_TYPES = ["MultiAgentGridworld"]

# --- State ---
state = {
    "env_type": "MultiAgentGridworld",
    "agent_types": ["Tabular Q-Learning"] * 3,
    "n_agents": 3,
    "n_obstacles": 2,
    "env": None,
    "agents": [],
    "obs": None,
    "done": False
}

def build_agents(agent_types, n_agents):
    agents = []
    n_states, n_actions = 5, 4
    for t in agent_types:
        if t == "Tabular Q-Learning":
            agents.append(TabularQLearningAgent(n_states=n_states, n_actions=n_actions))
        elif t == "DQN RL":
            agents.append(DQNAgent(state_dim=2, action_dim=4))
        elif t == "Neuro-Fuzzy":
            nn_config = {"input_dim": 2, "hidden_dim": 3, "output_dim": 1}
            fis_config = None
            agents.append(Agent(model=NeuroFuzzyHybrid(nn_config, fis_config)))
    return agents

def build_env(env_type, n_agents, n_obstacles):
    if env_type == "MultiAgentGridworld":
        return MultiAgentGridworldEnv(grid_size=5, n_agents=n_agents, n_obstacles=n_obstacles)
    raise ValueError("Unsupported env type")

def reset_all():
    state["agents"] = build_agents(state["agent_types"], state["n_agents"])
    state["env"] = build_env(state["env_type"], state["n_agents"], state["n_obstacles"])
    state["obs"] = state["env"].reset()
    state["done"] = False
    for agent in state["agents"]:
        agent.reset()

reset_all()

class ObservationRequest(BaseModel):
    obs: List[Any]

class ActionRequest(BaseModel):
    actions: List[int]

class AgentEnvConfig(BaseModel):
    env_type: str
    agent_types: List[str]
    n_agents: int
    n_obstacles: int

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/act", dependencies=[Depends(verify_api_key)])
def get_actions(request: ObservationRequest):
    actions = [int(agent.act(o)) for agent, o in zip(state["agents"], request.obs)]
    logger.info(f"/act: obs={request.obs} -> actions={actions}")
    return {"actions": actions}

@app.post("/step", dependencies=[Depends(verify_api_key)])
def step_env(request: ActionRequest):
    env = state["env"]
    obs, rewards, done = env.step(request.actions)
    for i, agent in enumerate(state["agents"]):
        agent.observe(rewards[i], obs[i], done)
    state["obs"] = obs
    state["done"] = done
    logger.info(f"/step: actions={request.actions} -> obs={obs}, rewards={rewards}, done={done}")
    return {"obs": obs, "rewards": rewards, "done": done}

@app.post("/reset", dependencies=[Depends(verify_api_key)])
def reset_env():
    reset_all()
    logger.info("/reset: Environment and agents reset.")
    return {"obs": state["obs"], "done": state["done"]}

@app.get("/state", dependencies=[Depends(verify_api_key)])
def get_state():
    return {
        "obs": state["obs"],
        "done": state["done"],
        "agent_knowledge": [getattr(agent, "online_knowledge", {}) for agent in state["agents"]],
        "law_violations": [getattr(agent, "law_violations", 0) for agent in state["agents"]],
        "agent_types": state["agent_types"],
        "env_type": state["env_type"]
    }

@app.post("/config", dependencies=[Depends(verify_api_key)])
def set_agent_env_config(cfg: AgentEnvConfig):
    if cfg.env_type not in ENV_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported env_type")
    if not all(t in AGENT_TYPES for t in cfg.agent_types):
        raise HTTPException(status_code=400, detail="Unsupported agent_type in agent_types")
    state["env_type"] = cfg.env_type
    state["agent_types"] = cfg.agent_types
    state["n_agents"] = cfg.n_agents
    state["n_obstacles"] = cfg.n_obstacles
    reset_all()
    logger.info(f"/config: Set env_type={cfg.env_type}, agent_types={cfg.agent_types}, n_agents={cfg.n_agents}, n_obstacles={cfg.n_obstacles}")
    return {"status": "ok", "obs": state["obs"]}

@app.get("/info", dependencies=[Depends(verify_api_key)])
def get_info():
    return {
        "supported_env_types": ENV_TYPES,
        "supported_agent_types": AGENT_TYPES,
        "current_config": {
            "env_type": state["env_type"],
            "agent_types": state["agent_types"],
            "n_agents": state["n_agents"],
            "n_obstacles": state["n_obstacles"]
        }
    }
