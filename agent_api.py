from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, Form
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Any, Optional
import numpy as np
import logging
from src.core.agent import Agent
from src.core.tabular_q_agent import TabularQLearningAgent
from src.core.dqn_agent import DQNAgent
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.env.multiagent_gridworld import MultiAgentGridworldEnv
import speech_recognition as sr
from PIL import Image
from src.core.multimodal_dqn_agent import MultiModalDQNAgent
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
import cv2

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

# --- Multi-modal agent (text+image+audio+video) ---
# Example: text (BERT 768) + image (ResNet18 512) + audio (BERT on Whisper transcript 768) + video (ResNet18 512)
MULTIMODAL_INPUT_DIMS = [768, 512, 768, 512]
MULTIMODAL_ACTION_DIM = 4
multimodal_agent = MultiModalDQNAgent(input_dims=MULTIMODAL_INPUT_DIMS, action_dim=MULTIMODAL_ACTION_DIM)

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

from src.env.environment_factory import EnvironmentFactory

def build_env(env_type, n_agents, n_obstacles):
    if env_type == "MultiAgentGridworld":
        return EnvironmentFactory.create("multiagent_gridworld_v2", grid_size=5, n_agents=n_agents, n_obstacles=n_obstacles)
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

@app.post("/observe/text", dependencies=[Depends(verify_api_key)])
def observe_text(text: str = Form(...)):
    # NLP: Use transformer to embed text
    inputs = nlp_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
        # Use [CLS] token embedding as feature
        feature = outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()
    action = int(state["agents"][0].act(feature))
    logger.info(f"/observe/text: text={text} -> action={action}")
    return {"action": action}

@app.post("/observe/audio", dependencies=[Depends(verify_api_key)])
def observe_audio(file: UploadFile = File(...)):
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.file.read())
        audio_path = tmp.name
    # Whisper speech-to-text
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]
    # NLP embedding of transcript
    inputs = nlp_tokenizer(transcript, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
        feature = outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()
    action = int(state["agents"][0].act(feature))
    logger.info(f"/observe/audio: transcript={transcript} -> action={action}")
    os.remove(audio_path)
    return {"action": action, "transcript": transcript}

@app.post("/observe/image", dependencies=[Depends(verify_api_key)])
def observe_image(file: UploadFile = File(...)):
    import io
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    img_tensor = vision_transform(image).unsqueeze(0)
    with torch.no_grad():
        feat = vision_model(img_tensor)
        feature = feat.squeeze().numpy().tolist()
    action = int(state["agents"][0].act(feature))
    logger.info(f"/observe/image: feature[0:5]={feature[:5]}... -> action={action}")
    return {"action": action, "feature_dim": len(feature)}

@app.post("/observe/video", dependencies=[Depends(verify_api_key)])
def observe_video(file: UploadFile = File(...)):
    import tempfile
    import os
    import cv2
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.file.read())
        video_path = tmp.name
    # Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        os.remove(video_path)
        raise HTTPException(status_code=400, detail="Could not read video file")
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = vision_transform(image).unsqueeze(0)
    with torch.no_grad():
        feat = vision_model(img_tensor)
        feature = feat.squeeze().numpy().tolist()
    action = int(state["agents"][0].act(feature))
    logger.info(f"/observe/video: feature[0:5]={feature[:5]}... -> action={action}")
    os.remove(video_path)
    return {"action": action, "feature_dim": len(feature)}

from src.integration.real_world_interface import RealWorldInterface
real_world = RealWorldInterface()

@app.post("/realworld/observe", dependencies=[Depends(verify_api_key)])
def realworld_observe(source_type: str = Form(...), config: str = Form(...)):
    """
    Fetch real-world observation from robot, API, or IoT sensor.
    config: JSON string with connection info
    """
    import json
    try:
        cfg = json.loads(config)
        obs = real_world.get_observation(source_type, cfg)
        return {"status": "ok", "observation": obs}
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Observation failed: {ex}")

@app.post("/realworld/act", dependencies=[Depends(verify_api_key)])
def realworld_act(target_type: str = Form(...), action: str = Form(...), config: str = Form(...)):
    """
    Send action/command to robot, API, or IoT sensor.
    action: JSON string
    config: JSON string with connection info
    """
    import json
    try:
        act = json.loads(action)
        cfg = json.loads(config)
        result = real_world.send_action(target_type, act, cfg)
        return {"status": "ok", "result": result}
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Action failed: {ex}")

@app.post("/explain", dependencies=[Depends(verify_api_key)])
def explain_action(agent_id: int = Form(...), observation: str = Form(...)):
    """
    Explain agent's action given observation/features.
    Accepts: agent_id, observation (comma-separated or JSON list)
    Returns: explanation dict
    """
    import json
    import numpy as np
    try:
        agent = state["agents"][agent_id]
    except Exception:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Parse observation/features
    def parse_vec(s):
        try:
            return np.array(json.loads(s))
        except Exception:
            return np.array([float(x) for x in s.split(",")])
    obs = parse_vec(observation)
    # For multimodal, expect a list of arrays
    if hasattr(agent, "explain_action"):
        try:
            # Try multimodal input if needed
            if hasattr(agent, "input_dims") and isinstance(obs, np.ndarray) and obs.dtype == object:
                obs = [np.array(x) for x in obs]
            result = agent.explain_action(obs)
            return {"status": "ok", "explanation": result}
        except Exception as ex:
            raise HTTPException(status_code=400, detail=f"Explain failed: {ex}")
    else:
        raise HTTPException(status_code=400, detail="Agent does not support explainability")

@app.post("/learn/online", dependencies=[Depends(verify_api_key)])
def learn_online(agent_id: int = Form(...), input: str = Form(...), target: str = Form(...)):
    """
    Online/continual learning endpoint.
    - agent_id: index of agent in state["agents"]
    - input: comma-separated floats or JSON-encoded list/array
    - target: comma-separated floats or JSON-encoded list/array
    """
    import json
    import numpy as np
    try:
        agent = state["agents"][agent_id]
    except Exception:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Parse input/target
    def parse_vec(s):
        try:
            return np.array(json.loads(s))
        except Exception:
            return np.array([float(x) for x in s.split(",")])
    x = parse_vec(input)
    y = parse_vec(target)
    # Route to appropriate update method
    if hasattr(agent, "online_update"):
        agent.online_update(x, y)
        logger.info(f"/learn/online: agent {agent_id} online_update called.")
        return {"status": "ok", "agent_id": agent_id}
    elif hasattr(agent, "observe"):
        # For DQN-style agents, treat y as (reward, next_state, done)
        reward = y[0]
        next_state = y[1:-1]
        done = bool(y[-1])
        agent.observe(reward, next_state, done)
        logger.info(f"/learn/online: agent {agent_id} observe called.")
        return {"status": "ok", "agent_id": agent_id}
    else:
        raise HTTPException(status_code=400, detail="Agent does not support online learning")

@app.post("/observe/multimodal", dependencies=[Depends(verify_api_key)])
def observe_multimodal(
    text: str = Form(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    video: UploadFile = File(...),
):
    import tempfile, os, cv2
    # Text feature (BERT)
    inputs = nlp_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
        text_feature = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    # Image feature (ResNet18)
    img = Image.open(image.file).convert("RGB")
    img_tensor = vision_transform(img).unsqueeze(0)
    with torch.no_grad():
        img_feature = vision_model(img_tensor).squeeze().numpy()
    # Audio feature: Whisper transcript -> BERT
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.file.read())
        audio_path = tmp.name
    whisper_result = whisper_model.transcribe(audio_path)
    transcript = whisper_result["text"]
    audio_inputs = nlp_tokenizer(transcript, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        audio_outputs = nlp_model(**audio_inputs)
        audio_feature = audio_outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    os.remove(audio_path)
    # Video feature: ResNet18 on first frame
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.file.read())
        video_path = tmp.name
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        os.remove(video_path)
        raise HTTPException(status_code=400, detail="Could not read video file")
    img_video = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_video_tensor = vision_transform(img_video).unsqueeze(0)
    with torch.no_grad():
        video_feature = vision_model(img_video_tensor).squeeze().numpy()
    os.remove(video_path)
    # Multi-modal agent expects [text_feature, img_feature, audio_feature, video_feature]
    action = multimodal_agent.act([
        text_feature, img_feature, audio_feature, video_feature
    ])
    logger.info(f"/observe/multimodal: action={action}")
    return {"action": action}

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
