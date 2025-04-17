import streamlit as st
from src.core.agent import Agent
from src.core.dqn_agent import DQNAgent
from src.core.multiagent import MultiAgentSystem
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.core.tabular_q_agent import TabularQLearningAgent
from src.core.anfis_agent import NeuroFuzzyANFISAgent
from src.env.simple_env import SimpleContinuousEnv, SimpleDiscreteEnv
import json
import requests
from collections import defaultdict, deque
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

reward_history = defaultdict(lambda: deque(maxlen=100))

def realworld_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Real-World Integration**")
    realworld_types = ["robot", "api", "iot_sensor"]
    realworld_mode = st.sidebar.selectbox("Mode", ["Observe", "Act"])
    realworld_type = st.sidebar.selectbox("Type", realworld_types)
    realworld_url = st.sidebar.text_input("Endpoint URL", value="http://localhost:9000/data")
    realworld_config = {"url": realworld_url}
    realworld_config_json = json.dumps(realworld_config)
    realworld_action = (
        st.sidebar.text_input("Action (JSON)", value='{"move": "forward"}')
        if realworld_mode == "Act" else None
    )
    realworld_result_placeholder = st.sidebar.empty()
    if st.sidebar.button(f"Real-World {realworld_mode}"):
        try:
            api_url = f"http://localhost:8000/realworld/{'observe' if realworld_mode == 'Observe' else 'act'}"
            headers = {"X-API-Key": "mysecretkey"}
            data = (
                {"config": realworld_config_json, "source_type": realworld_type}
                if realworld_mode == "Observe"
                else {"config": realworld_config_json, "target_type": realworld_type, "action": realworld_action}
            )
            r = requests.post(api_url, data=data, headers=headers, timeout=3)
            realworld_result_placeholder.success(f"Result: {r.text}")
        except Exception as ex:
            realworld_result_placeholder.error(f"Failed: {ex}")


def initialize_env_and_agents(agent_type, agent_count, n_obstacles):
    # This logic is moved from dashboard.py lines 159-241
    from src.env.environment_factory import EnvironmentFactory
    agents = []
    env_key = None
    env_kwargs = {}
    agent_types = None
    alpha = gamma = epsilon = n_states = n_actions = mm_img_dim = mm_txt_dim = mm_hidden_dim = mm_n_actions = mm_fusion_type = n_pursuers = n_evaders = 0
    env_type = agent_type  # This may need to be adjusted depending on your sidebar logic
    # Example mapping, adapt as needed:
    if env_type == "Multi-Agent Gridworld":
        env_key = "multiagent_gridworld_v2"
        env_kwargs = {
            "n_agents": agent_count,
            "n_resources": 3,
            "mode": "competitive",
        }
    elif env_key == "simple_discrete":
        env_kwargs = {"n_states": 5, "n_actions": 2}
    elif env_key == "simple_continuous":
        env_kwargs = {}
    env = EnvironmentFactory.create(env_key, **env_kwargs) if env_key else None

    # Agent creation logic (preserve previous logic)
    if env_key in ["multiagent_gridworld_v2", "multiagent_gridworld"]:
        for i in range(agent_count):
            ag_type = agent_types[i] if agent_types else "Tabular Q-Learning"
            if ag_type == "Neuro-Fuzzy":
                nn_config = {"input_dim": 2, "hidden_dim": 3, "output_dim": 1}
                fis_config = None
                agents.append(Agent(model=NeuroFuzzyHybrid(nn_config, fis_config)))
            elif ag_type == "DQN RL":
                agents.append(
                    DQNAgent(
                        state_dim=2,
                        action_dim=4,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon,
                    )
                )
            elif ag_type == "Multi-Modal Fusion Agent":
                from src.core.multimodal_fusion_agent import MultiModalFusionAgent
                agents.append(
                    MultiModalFusionAgent(
                        [mm_img_dim, mm_txt_dim],
                        mm_hidden_dim,
                        mm_n_actions,
                        fusion_type=mm_fusion_type,
                    )
                )
            else:
                agents.append(
                    TabularQLearningAgent(
                        n_states=n_states,
                        n_actions=n_actions,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon,
                    )
                )
    elif env_type == "Adversarial Gridworld":
        from src.env.adversarial_gridworld import AdversarialGridworldEnv
        n_states, n_actions = 5, 4
        agents = [
            TabularQLearningAgent(
                n_states=n_states,
                n_actions=n_actions,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
            )
            for _ in range(agent_count)
        ]
        env = AdversarialGridworldEnv(
            grid_size=5,
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            n_obstacles=n_obstacles,
        )
    elif env_type == "ANFIS Agent":
        # For demo: input_dim=2, n_rules=4, lr=0.05
        agents = [NeuroFuzzyANFISAgent(input_dim=2, n_rules=4, lr=0.05) for _ in range(agent_count)]
        env = SimpleContinuousEnv()  # Or another suitable env
    else:
        agents = [
            TabularQLearningAgent(
                n_states=5, n_actions=4, alpha=alpha, gamma=gamma, epsilon=epsilon
            )
            for _ in range(agent_count)
        ]
        env = None
    return env, agents
