import datetime
import os
import pickle

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
import json

import streamlit as st

st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
import time
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import requests

from dashboard_env import initialize_env_and_agents, realworld_sidebar
from dashboard_tables import render_group_decisions_log, render_knowledge_table
from dashboard_viz import plot_group_leader_spatial, plot_som_grid

# --- RL Reward History ---
reward_history = defaultdict(lambda: deque(maxlen=100))

# --- Environment and Agents Initialization ---
# Example usage (adapt as needed):
# env, agents = initialize_env_and_agents(agent_type, agent_count, n_obstacles)

import networkx as nx
import numpy as np

from src.core.agent import Agent
from src.core.dqn_agent import DQNAgent
from src.core.multiagent import MultiAgentSystem
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.core.tabular_q_agent import TabularQLearningAgent
from src.env.simple_env import SimpleContinuousEnv, SimpleDiscreteEnv

# --- Real-World Integration Sidebar ---
realworld_sidebar()

# --- Sidebar: Agent Type Selection & Parameter Tuning ---
from src.env.environment_factory import EnvironmentFactory

# List of registered environments for dynamic selection
env_choices = [
    ("Gridworld", "multiagent_gridworld_v2"),
    ("Gridworld (Simple)", "multiagent_gridworld"),
    ("Adversarial Gridworld", "adversarial_gridworld"),
    ("Resource Collection", "multiagent_resource"),
    ("Simple Discrete", "simple_discrete"),
    ("Simple Continuous", "simple_continuous"),
]

env_type, env_key = st.sidebar.selectbox(
    "Environment Type", env_choices, format_func=lambda x: x[0]
)

agent_type = None  # Always defined for safety

if env_type == "Gridworld":
    agent_type = None
    agent_count = st.sidebar.slider("Number of Agents", 1, 5, 3)
    n_obstacles = st.sidebar.slider("Number of Obstacles", 0, 10, 2)
    agent_type_choices = [
        "Neuro-Fuzzy",
        "Tabular Q-Learning",
        "DQN RL",
        "Multi-Modal Fusion Agent",
        "ANFIS Agent",
    ]
    agent_types = [
        st.sidebar.selectbox(
            f"Agent {i+1} Type", agent_type_choices, key=f"agent_type_{i}"
        )
        for i in range(agent_count)
    ]
    # Multi-modal settings
    # Multi-modal agent settings
    # If any agent is a Multi-Modal Fusion Agent, expose fusion settings in sidebar
    if "Multi-Modal Fusion Agent" in agent_types:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Multi-Modal Fusion Settings**")
        mm_img_dim = st.sidebar.number_input(
            "Image feature dim", 8, 128, 32, key="mm_img_dim"
        )
        mm_txt_dim = st.sidebar.number_input(
            "Text feature dim", 4, 64, 16, key="mm_txt_dim"
        )
        mm_hidden_dim = st.sidebar.number_input(
            "Fusion hidden dim", 8, 128, 32, key="mm_hidden_dim"
        )
        mm_n_actions = st.sidebar.number_input(
            "Number of actions", 2, 10, 4, key="mm_n_actions"
        )
        mm_fusion_type = st.sidebar.selectbox(
            "Fusion Method",
            ["concat", "attention", "gating"],
            index=0,
            key="mm_fusion_type",
        )
        # TODO: Add more modalities and fusion methods (e.g., attention, gating) here
elif env_type == "Adversarial Gridworld":
    agent_type = "Tabular Q-Learning"
    n_pursuers = st.sidebar.slider("Number of Pursuers", 1, 3, 1)
    n_evaders = st.sidebar.slider("Number of Evaders", 1, 3, 1)
    n_obstacles = st.sidebar.slider("Number of Obstacles", 0, 10, 2)
    agent_types = None
else:
    agent_type = "Tabular Q-Learning"
    agent_count = st.sidebar.slider("Number of Agents", 1, 5, 3)
    n_obstacles = 0
    agent_types = None

# --- Online Learning Toggle ---
online_learning_enabled = st.sidebar.checkbox("Enable Online Learning", value=True)

# Human-in-the-loop: RL parameter tuning
if (
    env_type == "Gridworld"
    and any(t in ["Tabular Q-Learning", "DQN RL"] for t in agent_types)
) or (env_type != "Gridworld" and agent_type in ["Tabular Q-Learning", "DQN RL"]):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Agent Parameters**")
    alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("Discount Factor (gamma)", 0.0, 1.0, 0.99, 0.01)
    epsilon = st.sidebar.slider("Exploration Rate (epsilon)", 0.0, 1.0, 0.1, 0.01)
else:
    alpha = gamma = epsilon = None


# --- Session State Setup ---
if "env" not in st.session_state or st.sidebar.button("Reset Environment"):
    st.session_state.env, st.session_state.agents = initialize_env_and_agents(
        agent_type, agent_count, n_obstacles
    )
    st.session_state.obs = (
        st.session_state.env.reset() if st.session_state.env else None
    )
    st.session_state.done = False
    st.session_state.rewards = [0 for _ in range(agent_count)]
    st.session_state.step = 0


# --- Simulation Step ---
def simulate_step():
    env = st.session_state.env
    agents = st.session_state.agents
    if env is None or st.session_state.done:
        return

    # Get actions for all agents
    def to_scalar_action(a):
        import numpy as np

        if isinstance(a, np.ndarray):
            return int(a.item()) if a.size == 1 else int(a.flat[0])
        return int(a) if isinstance(a, (np.integer, np.floating)) else a

    # --- Adversarial perturbation logic ---
    feedback = st.session_state.get("feedback", {})
    orig_obs = list(st.session_state.obs)
    perturbed_obs = []
    for i, obs in enumerate(orig_obs):
        is_dqn = agents[i].__class__.__name__ == "DQNAgent"
        if (
            adversarial_enabled
            and i in adversarial_agents
            and adversarial_type != "None"
        ):
            obs_arr = np.array(obs, dtype=np.float32)
            if adversarial_type == "Gaussian Noise":
                obs_arr = obs_arr + np.random.normal(
                    0, adversarial_strength, size=obs_arr.shape
                )
            elif adversarial_type == "Uniform Noise":
                obs_arr = obs_arr + np.random.uniform(
                    -adversarial_strength, adversarial_strength, size=obs_arr.shape
                )
            elif adversarial_type == "Zeros":
                obs_arr = np.zeros_like(obs_arr)
            elif adversarial_type == "Max Value":
                obs_arr = np.ones_like(obs_arr) * adversarial_strength
            elif adversarial_type == "FGSM (Targeted, DQN)" and is_dqn:
                try:
                    import torch

                    obs_tensor = (
                        torch.tensor(obs_arr, requires_grad=True).unsqueeze(0).float()
                    )
                    qvals = agents[i].model(obs_tensor)
                    action = torch.argmax(qvals, dim=1)
                    loss = -qvals[0, action]
                    loss.backward()
                    grad_sign = obs_tensor.grad.data.sign().squeeze(0).numpy()
                    obs_arr = obs_arr + adversarial_strength * grad_sign
                except Exception:
                    # If torch or model not available, fallback to Gaussian noise
                    obs_arr = obs_arr + np.random.normal(
                        0, adversarial_strength, size=obs_arr.shape
                    )
            perturbed_obs.append(obs_arr.tolist())
        else:
            perturbed_obs.append(obs)
    st.session_state.perturbed_obs = perturbed_obs
    actions = []
    import numpy as np

    from src.core.multimodal_fusion_agent import MultiModalFusionAgent

    for i, (agent, obs) in enumerate(zip(agents, perturbed_obs)):
        # If agent is MultiModalFusionAgent and obs is not a list, inject random multi-modal input
        # This allows dashboard simulation even if environment is not multi-modal aware
        if isinstance(agent, MultiModalFusionAgent) and not (
            isinstance(obs, list) and len(obs) == 2
        ):
            # Use agent.model.input_dims for feature sizes
            img_dim, txt_dim = agent.model.input_dims
            obs = [np.random.randn(img_dim), np.random.randn(txt_dim)]
            # TODO: Replace with real multi-modal features from environment when available
        fb = feedback.get(
            i, {"approve": "Approve", "override_action": None, "custom_reward": None}
        )
        if fb["approve"] == "Reject":
            action = getattr(agent, "last_action", 0)
        elif fb["approve"] == "Override" and fb["override_action"] is not None:
            try:
                action = (
                    type(obs[0])(fb["override_action"])
                    if hasattr(obs, "__getitem__")
                    else int(fb["override_action"])
                )
            except Exception:
                action = fb["override_action"]
        else:
            action = to_scalar_action(agent.act(obs))
        actions.append(action)
    next_obs, rewards, done = env.step(actions)
    # Apply custom rewards if provided
    for i, fb in feedback.items():
        if fb.get("custom_reward") is not None:
            try:
                rewards[i] = float(fb["custom_reward"])
            except Exception:
                pass
    # Update agents with new experience
    import random

    for i, agent in enumerate(st.session_state.agents):
        # Simulate knowledge update
        agent.integrate_online_knowledge({"step": st.session_state.step})
        agent.online_knowledge = {
            "step": st.session_state.step
        }  # Ensure dashboard always updates
        # Agent-to-agent communication: send message to a random other agent
        others = [a for a in st.session_state.agents if a is not agent]
        if others:
            recipient = random.choice(others)
            msg = {
                "from": i,
                "step": st.session_state.step,
                "knowledge": agent.online_knowledge,
            }
            agent.send_message(msg, recipient)
        # Track law violations
        try:
            agent.observe(rewards[i], next_obs[i], done)
        except Exception as e:
            # If a LawViolation, increment counter
            if e.__class__.__name__ == "LawViolation":
                if not hasattr(agent, "law_violations"):
                    agent.law_violations = 0
                agent.law_violations += 1
            else:
                raise
        # --- Online Learning: Automatic ---
        if online_learning_enabled:
            # For DQN-like: input=obs, target=[reward, next_obs, done]
            try:
                api_url = "http://localhost:8000/learn/online"
                headers = {"X-API-Key": "mysecretkey"}
                obs_str = ",".join(str(x) for x in st.session_state.obs[i])
                next_obs_str = ",".join(str(x) for x in next_obs[i])
                target_str = f"{rewards[i]},{next_obs_str},{int(done)}"
                data = {"agent_id": i, "input": obs_str, "target": target_str}
                requests.post(api_url, data=data, headers=headers, timeout=2)
            except Exception as ex:
                st.session_state["online_learning_log"] = (
                    f"Online update failed for agent {i}: {ex}"
                )
    st.session_state.obs = next_obs
    st.session_state.rewards = rewards
    st.session_state.done = done
    st.session_state.step += 1


# --- Main UI ---
st.title(f"🤖 Multi-Agent System Dashboard ({agent_type})")
tabs = st.tabs(
    [
        "Simulation",
        "Analytics",
        "Manual Feedback",
        "Batch Experiments",
        "Model Management",
        "Multi-Modal Demo",
    ]
)

# --- Manual Feedback Tab ---
with tabs[2]:
    st.header("Manual Feedback Review and Edit")
    feedback = st.session_state.get("feedback", {})
    for i, agent in enumerate(st.session_state.agents):
        with st.expander(f"Agent {i} Feedback History", expanded=False):
            approve = st.radio(
                f"Approve action for Agent {i}? (Manual)",
                ["Approve", "Reject", "Override"],
                index=["Approve", "Reject", "Override"].index(
                    feedback.get(i, {}).get("approve", "Approve")
                ),
                key=f"manual_approve_{i}",
            )
            override_action = st.text_input(
                f"Override Action (optional, Manual)",
                value=str(feedback.get(i, {}).get("override_action", "")),
                key=f"manual_override_{i}",
            )
            custom_reward = st.number_input(
                f"Custom Reward (optional, Manual)",
                value=float(
                    feedback.get(i, {}).get(
                        "custom_reward",
                        (
                            st.session_state.rewards[i]
                            if hasattr(st.session_state, "rewards")
                            else 0
                        ),
                    )
                ),
                key=f"manual_reward_{i}",
            )
            if st.button(f"Update Feedback for Agent {i}", key=f"manual_update_{i}"):
                st.session_state.feedback[i] = {
                    "approve": approve,
                    "override_action": (
                        override_action if approve == "Override" else None
                    ),
                    "custom_reward": custom_reward,
                }
                st.success(f"Feedback for Agent {i} updated.")

# --- Batch Experiments Tab ---
with tabs[3]:
    pass
# --- Model Management Tab ---

with tabs[4]:
    pass

# --- Multi-Modal Fusion Agent Demo Tab ---
with tabs[5]:
    import numpy as np

    from src.core.multimodal_fusion_agent import MultiModalFusionAgent

    st.header("Multi-Modal Fusion Agent Demo")
    st.write("Test the Multi-Modal Fusion Agent with random image and text features.")
    img_dim = st.number_input("Image feature dim", 8, 128, 32)
    txt_dim = st.number_input("Text feature dim", 4, 64, 16)
    n_actions = st.number_input("Number of actions", 2, 10, 4)
    hidden_dim = st.number_input("Fusion hidden dim", 8, 128, 32)
    # Generate random modalities
    img_feat = np.random.randn(img_dim)
    txt_feat = np.random.randn(txt_dim)
    st.write("#### Image Feature (random)")
    st.write(img_feat)
    st.write("#### Text Feature (random)")
    st.write(txt_feat)
    agent = MultiModalFusionAgent([img_dim, txt_dim], hidden_dim, n_actions)
    action = agent.act([img_feat, txt_feat])
    st.success(f"Agent selected action: {action}")
    st.header("Model Management & Versioning")
    save_name = st.text_input(
        "Save Model As",
        value=f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    if st.button("Save Current Agents", key="save_model"):
        save_path = os.path.join(MODEL_DIR, save_name + ".pkl")
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent_type": [type(a).__name__ for a in st.session_state.agents],
            "params": getattr(st.session_state, "env", None),
        }
        with open(save_path, "wb") as f:
            pickle.dump({"agents": st.session_state.agents, "meta": meta}, f)
        st.success(f"Saved to {save_path}")
    # List saved models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    st.write("### Saved Models")
    st.write("### Evaluate & Compare Saved Models")
    selected_models = st.multiselect("Select models to evaluate", model_files)
    n_eval_episodes = st.number_input(
        "Episodes per evaluation", min_value=1, max_value=50, value=5
    )
    if st.button("Run Evaluation", key="eval_models") and selected_models:
        eval_results = {}
        for mf in selected_models:
            with open(os.path.join(MODEL_DIR, mf), "rb") as f:
                loaded = pickle.load(f)
                agents = loaded["agents"]
                env, _ = initialize_env_and_agents(agent_count=len(agents))
                rewards_all = []
                law_violations = []
                for ep in range(n_eval_episodes):
                    obs = env.reset()
                    ep_rewards = [0 for _ in range(len(agents))]
                    for step in range(100):
                        acts = [agents[i].act(obs[i]) for i in range(len(agents))]
                        obs, rewards, done = env.step(acts)
                        for i in range(len(agents)):
                            ep_rewards[i] += rewards[i]
                        if done:
                            break
                    rewards_all.append(ep_rewards)
                    law_violations.append(
                        [getattr(a, "law_violations", 0) for a in agents]
                    )
                eval_results[mf] = {
                    "avg_reward": np.mean(rewards_all, axis=0).tolist(),
                    "law_violations": np.sum(law_violations, axis=0).tolist(),
                }
        st.session_state.eval_results = eval_results
        st.success("Evaluation complete!")
    if "eval_results" in st.session_state:
        st.write("### Evaluation Results")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for mf, res in st.session_state.eval_results.items():
            ax.plot(res["avg_reward"], label=f"{mf} (reward)")
        ax.set_xlabel("Agent Index")
        ax.set_ylabel("Avg Reward (eval)")
        ax.legend(fontsize=7)
        st.pyplot(fig)

# --- Group & Leader Spatial Visualization ---
if hasattr(st.session_state.agents[0], "position"):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    group_ids = list(set(agent.group for agent in st.session_state.agents))
    group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
    for i, agent in enumerate(st.session_state.agents):
        color = group_colors.get(agent.group, "gray")
        x, y = agent.position if hasattr(agent, "position") else (0, 0)
        marker = "*" if getattr(agent, "is_leader", False) else "o"
        size = 250 if getattr(agent, "is_leader", False) else 100
        ax2.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=size,
            edgecolor="black",
            label=f"{agent.group}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}",
        )
        ax2.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
    # Unique legend
    handles = []
    for gid in group_ids:
        is_leader = any(
            getattr(agent, "is_leader", False) and agent.group == gid
            for agent in st.session_state.agents
        )
        marker = "*" if is_leader else "o"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=f"{gid}{' (Leader)' if is_leader else ''}",
                markerfacecolor=group_colors[gid],
                markeredgecolor="black",
                markersize=12 if is_leader else 8,
            )
        )
    ax2.legend(handles=handles, loc="best")
    ax2.set_title("Agent Positions by Group & Leader")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    st.pyplot(fig2)

st.write("### Law Violations")
if "eval_results" not in st.session_state:
    st.session_state.eval_results = {}
for mf, res in st.session_state.eval_results.items():
    st.write(f"{mf}: {res['law_violations']}")
for mf in model_files:
    with open(os.path.join(MODEL_DIR, mf), "rb") as f:
        content = pickle.load(f)
        meta = content.get("meta", {})
    st.write(f"**{mf}** | {meta.get('timestamp', '')} | {meta.get('agent_type', '')}")
    if st.button(f"Load {mf}", key=f"load_{mf}"):
        with open(os.path.join(MODEL_DIR, mf), "rb") as f2:
            loaded = pickle.load(f2)
            st.session_state.agents = loaded["agents"]
        st.success(f"Loaded agents from {mf}")
    st.header("Batch Experimentation & Parameter Sweeps")
    st.write(
        "Configure and run multiple experiments with different parameters. Results will be aggregated for analysis."
    )
    sweep_agent_counts = st.text_input("Agent Counts (comma-separated)", value="2,3,4")
    sweep_learning_rates = st.text_input(
        "Learning Rates (comma-separated)", value="0.1,0.2,0.5"
    )
    sweep_gammas = st.text_input(
        "Gammas (Discount, comma-separated)", value="0.9,0.95,0.99"
    )
    sweep_epsilons = st.text_input(
        "Epsilons (Exploration, comma-separated)", value="0.05,0.1,0.2"
    )
    n_steps = st.number_input(
        "Steps per Experiment", min_value=10, max_value=500, value=50
    )
    run_batch = st.button("Run Batch Experiments")
    if run_batch:
        import itertools

        agent_counts = [int(x) for x in sweep_agent_counts.split(",") if x.strip()]
        learning_rates = [
            float(x) for x in sweep_learning_rates.split(",") if x.strip()
        ]
        gammas = [float(x) for x in sweep_gammas.split(",") if x.strip()]
        epsilons = [float(x) for x in sweep_epsilons.split(",") if x.strip()]
        results = []
        for ac, lr, gamma, eps in itertools.product(
            agent_counts, learning_rates, gammas, epsilons
        ):
            # Minimal env/agent re-init for each run
            env, agents = initialize_env_and_agents(
                agent_count=ac, alpha=lr, gamma=gamma, epsilon=eps
            )
            obs = env.reset()
            rewards_acc = [0 for _ in range(ac)]
            for step in range(n_steps):
                acts = [agents[i].act(obs[i]) for i in range(ac)]
                obs, rewards, done = env.step(acts)
                for i in range(ac):
                    rewards_acc[i] += rewards[i]
                if done:
                    break
            results.append(
                {
                    "agent_count": ac,
                    "learning_rate": lr,
                    "gamma": gamma,
                    "epsilon": eps,
                    "avg_reward": [r / n_steps for r in rewards_acc],
                }
            )
        st.session_state.batch_results = results
        import datetime

        import requests

        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user": "local_user",
            "params": {
                "agent_counts": agent_counts,
                "learning_rates": learning_rates,
                "gammas": gammas,
                "epsilons": epsilons,
                "steps": n_steps,
            },
            "results": results,
        }
        try:
            r = requests.post(
                "http://localhost:8000/api/experiments/log", json=meta, timeout=5
            )
            if r.status_code == 200:
                st.success("Batch experiments complete! Results logged to backend.")
            else:
                st.warning(
                    f"Batch complete, but backend logging failed: {r.status_code}"
                )
        except Exception as ex:
            st.warning(f"Batch complete, but backend logging error: {ex}")
    if "batch_results" in st.session_state:
        st.write("### Batch Results Table")
        st.dataframe(st.session_state.batch_results)
        # --- Export buttons ---
        import json

        import pandas as pd

        df = pd.DataFrame(st.session_state.batch_results)
        csv = df.to_csv(index=False).encode("utf-8")
        json_str = json.dumps(st.session_state.batch_results, indent=2)
        st.download_button("Download CSV", csv, "batch_results.csv", "text/csv")
        st.download_button(
            "Download JSON", json_str, "batch_results.json", "application/json"
        )
        st.write("### Aggregate Reward Plot")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for res in st.session_state.batch_results:
            label = f"Agents: {res['agent_count']}, LR: {res['learning_rate']}, γ: {res['gamma']}, ε: {res['epsilon']}"
            ax.plot(res["avg_reward"], label=label)
        ax.set_xlabel("Agent Index")
        ax.set_ylabel("Avg Reward (per step)")
        ax.legend(fontsize=7)
        st.pyplot(fig)

# --- Group & Leader Spatial Visualization ---
if hasattr(st.session_state.agents[0], "position"):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    group_ids = list(set(agent.group for agent in st.session_state.agents))
    group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
    for i, agent in enumerate(st.session_state.agents):
        color = group_colors.get(agent.group, "gray")
        x, y = agent.position if hasattr(agent, "position") else (0, 0)
        marker = "*" if getattr(agent, "is_leader", False) else "o"
        size = 250 if getattr(agent, "is_leader", False) else 100
        ax2.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=size,
            edgecolor="black",
            label=f"{agent.group}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}",
        )
        ax2.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
    # Unique legend
    handles = []
    for gid in group_ids:
        is_leader = any(
            getattr(agent, "is_leader", False) and agent.group == gid
            for agent in st.session_state.agents
        )
        marker = "*" if is_leader else "o"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=f"{gid}{' (Leader)' if is_leader else ''}",
                markerfacecolor=group_colors[gid],
                markeredgecolor="black",
                markersize=12 if is_leader else 8,
            )
        )
    ax2.legend(handles=handles, loc="best")
    ax2.set_title("Agent Positions by Group & Leader")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    st.pyplot(fig2)


# --- Manual Online Update UI ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Manual Online Update**")
manual_agent_id = st.sidebar.number_input("Agent ID", min_value=0, max_value=4, value=0)
manual_input = st.sidebar.text_input("Input (comma-separated)", value="1.0,2.0")
manual_target = st.sidebar.text_input("Target (comma-separated)", value="0.5")
if st.sidebar.button("Send Online Update"):
    try:
        api_url = "http://localhost:8000/learn/online"
        headers = {"X-API-Key": "mysecretkey"}
        data = {
            "agent_id": manual_agent_id,
            "input": manual_input,
            "target": manual_target,
        }
        r = requests.post(api_url, data=data, headers=headers, timeout=2)
        st.sidebar.success(f"Online update sent: {r.json()}")
    except Exception as ex:
        st.sidebar.error(f"Failed to send online update: {ex}")

# --- Adversarial Testing Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Adversarial Testing**")
adversarial_enabled = st.sidebar.checkbox("Enable Adversarial Testing", value=False)
perturb_types = [
    "None",
    "Gaussian Noise",
    "Uniform Noise",
    "Zeros",
    "Max Value",
    "FGSM (Targeted, DQN)",
]
adversarial_type = st.sidebar.selectbox("Perturbation Type", perturb_types)
adversarial_strength = st.sidebar.slider("Perturbation Strength", 0.0, 2.0, 0.2, 0.01)
agent_count = st.session_state.get("agents", [None] * 3)
if isinstance(agent_count, int):
    agent_indices = list(range(agent_count))
else:
    agent_indices = list(range(len(agent_count)))
adversarial_agents = st.sidebar.multiselect(
    "Agents to Attack", agent_indices, default=agent_indices
)

# --- Human-in-the-Loop Feedback ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Human-in-the-Loop Feedback**")
auto_approve = st.sidebar.checkbox("Auto-approve all agent actions", value=False)

# Store feedback in session state
if "feedback" not in st.session_state:
    st.session_state.feedback = {}


with tabs[0]:
    st.write(f"Step: {st.session_state.step}")
    st.write(f"Done: {st.session_state.done}")
    if st.button("Step Simulation", key="main_step_sim"):
        simulate_step()
    if st.session_state.obs is not None:
        st.write("### Agent Observations (Original)")
        st.write(st.session_state.obs)
        if adversarial_enabled and any(
            i in adversarial_agents for i in range(len(st.session_state.obs))
        ):
            st.write("### Agent Observations (Perturbed)")
            st.write(st.session_state.perturbed_obs)
        st.write("### Rewards")
        st.write(st.session_state.rewards)

        # --- Analytics: Plot rewards over time ---
        if "reward_history" not in st.session_state:
            st.session_state.reward_history = {
                i: [] for i in range(len(st.session_state.agents))
            }
        for i, r in enumerate(st.session_state.rewards):
            st.session_state.reward_history[i].append(r)
        st.write("### Reward Trajectories")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, rewards in st.session_state.reward_history.items():
            color = "red" if adversarial_enabled and i in adversarial_agents else None
            ax.plot(rewards, label=f"Agent {i}", color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.legend()
        st.pyplot(fig)

# --- Group & Leader Spatial Visualization ---
if hasattr(st.session_state.agents[0], "position"):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    group_ids = list(set(agent.group for agent in st.session_state.agents))
    group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
    for i, agent in enumerate(st.session_state.agents):
        color = group_colors.get(agent.group, "gray")
        x, y = agent.position if hasattr(agent, "position") else (0, 0)
        marker = "*" if getattr(agent, "is_leader", False) else "o"
        size = 250 if getattr(agent, "is_leader", False) else 100
        ax2.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=size,
            edgecolor="black",
            label=f"{agent.group}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}",
        )
        ax2.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
    # Unique legend
    handles = []
    for gid in group_ids:
        is_leader = any(
            getattr(agent, "is_leader", False) and agent.group == gid
            for agent in st.session_state.agents
        )
        marker = "*" if is_leader else "o"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=f"{gid}{' (Leader)' if is_leader else ''}",
                markerfacecolor=group_colors[gid],
                markeredgecolor="black",
                markersize=12 if is_leader else 8,
            )
        )
    ax2.legend(handles=handles, loc="best")
    ax2.set_title("Agent Positions by Group & Leader")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    st.pyplot(fig2)

# --- Law violations (if tracked) ---
if "eval_results" not in st.session_state:
    st.session_state.eval_results = {}
st.write("### Law Violations (if any)")
violations = [getattr(agent, "law_violations", 0) for agent in st.session_state.agents]
st.write({f"Agent {i}": v for i, v in enumerate(violations)})

# --- Inline Feedback UI ---
if not auto_approve:
    st.write("## Human-in-the-Loop Feedback (Approve, Override, or Reward)")
    for i, agent in enumerate(st.session_state.agents):
        with st.expander(f"Agent {i} Feedback", expanded=True):
            default_action = getattr(agent, "last_action", None)
            approve = st.radio(
                f"Approve action for Agent {i}?",
                ["Approve", "Reject", "Override"],
                key=f"approve_{i}",
            )
            override_action = st.text_input(
                f"Override Action (optional)",
                value=str(default_action) if default_action is not None else "",
                key=f"override_{i}",
            )
            custom_reward = st.number_input(
                f"Custom Reward (optional)",
                value=float(st.session_state.rewards[i]),
                key=f"reward_{i}",
            )
            st.session_state.feedback[i] = {
                "approve": approve,
                "override_action": override_action if approve == "Override" else None,
                "custom_reward": custom_reward,
            }
else:
    # Auto-approve: set feedback to approve for all
    for i, agent in enumerate(st.session_state.agents):
        st.session_state.feedback[i] = {
            "approve": "Approve",
            "override_action": None,
            "custom_reward": st.session_state.rewards[i],
        }

    # Table of agent knowledge/law violations (example)
    st.write("### Agent Knowledge Table")
    import requests

    for i, agent in enumerate(st.session_state.agents):
        ag_type = agent_types[i] if agent_types else agent.__class__.__name__
        last_msg = getattr(agent, "last_message", None)
        if (
            isinstance(last_msg, tuple)
            and len(last_msg) == 2
            and isinstance(last_msg[0], dict)
        ):
            last_msg_display = last_msg[0]
        else:
            last_msg_display = last_msg
        st.write(
            f"Agent {i} [{ag_type}] (Step {st.session_state.step}): Knowledge: {getattr(agent, 'online_knowledge', {})} Law Violations: {getattr(agent, 'law_violations', 0)} Last Msg: {last_msg_display}"
        )
        if hasattr(agent, "epsilon"):
            agent.epsilon = epsilon
        # Only display Q-table for Tabular Q-Learning
        if agent_type == "Tabular Q-Learning" and hasattr(agent, "q_table"):
            st.text(f"Q-table for Agent {i+1}:")
            st.write(agent.q_table)
        # --- Explainability Button ---
        explain_placeholder = st.empty()
        if st.button(f"Explain Agent {i}"):
            try:
                api_url = "http://localhost:8000/explain"
                headers = {"X-API-Key": "mysecretkey"}
                # Prepare observation string for API
                obs = st.session_state.obs[i]
                if isinstance(obs, (list, tuple)):
                    obs_str = ",".join(str(x) for x in obs)
                else:
                    obs_str = str(obs)
                data = {"agent_id": i, "observation": obs_str}
                r = requests.post(api_url, data=data, headers=headers, timeout=2)
                result = r.json()
                if result.get("status") == "ok":
                    explanation = result["explanation"]
                    # Pretty visualization
                    if "q_values" in explanation:
                        st.subheader(f"Q-values for Agent {i}")
                        st.table(
                            [[j, v] for j, v in enumerate(explanation["q_values"])]
                        )
                    if "rule_activations" in explanation:
                        st.subheader(f"Fuzzy Rule Activations for Agent {i}")
                        st.table(
                            [
                                [j, v]
                                for j, v in enumerate(explanation["rule_activations"])
                            ]
                        )
                    if "nn_output" in explanation:
                        st.subheader(f"Neural Net Output for Agent {i}")
                        st.write(explanation["nn_output"])
                    st.info(
                        f"Other Explanation Info: { {k: v for k, v in explanation.items() if k not in ['q_values', 'rule_activations', 'nn_output']} }"
                    )
                else:
                    explain_placeholder.warning(f"Failed to explain: {result}")
            except Exception as ex:
                explain_placeholder.error(f"Failed to get explanation: {ex}")


agent_count = len(st.session_state.agents)
for i, agent in enumerate(st.session_state.agents):
    agent.group = "G1" if i < (agent_count // 2 + 1) else "G2"

system = MultiAgentSystem(st.session_state.agents)

# --- State Tracking ---
knowledge_events = deque(maxlen=50)
group_decisions = deque(maxlen=20)

# --- Streamlit UI ---
st.title(f"🤖 Multi-Agent System Dashboard ({agent_type})")

tab_labels = ["Simulation", "Analytics"]
# Check if there are any MultiModalFusionAgents
fusion_agents = [
    a
    for a in st.session_state.agents
    if a.__class__.__name__ == "MultiModalFusionAgent"
]
if fusion_agents:
    tab_labels.append("Fusion Agent Explainability")
tabs = st.tabs(tab_labels)

with tabs[0]:
    run_sim = st.sidebar.button("Step Simulation", key="sidebar_step_sim")
    auto_run = st.sidebar.checkbox("Auto Step (every 2s)")
    online_learning_enabled = st.sidebar.checkbox("Enable Online Learning for Fusion Agents", value=True)
    # --- SOM Grid Visualization ---
    if hasattr(system, 'groups'):
        from dashboard_viz import plot_som_grid
        group_ids = list(system.groups.keys())
        group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
        plot_som_grid(group_ids, st.session_state.agents, group_colors)

if fusion_agents and len(tabs) > 2:
    with tabs[2]:
        st.header("Fusion Agent Explainability")
        agent_names = [
            f"Agent {st.session_state.agents.index(a)}" for a in fusion_agents
        ]
        selected_idx = st.selectbox(
            "Select Fusion Agent",
            range(len(fusion_agents)),
            format_func=lambda i: agent_names[i],
        )
        agent = fusion_agents[selected_idx]
        st.write(f"**Selected:** {agent}")
        # Use a sample observation (random or from session state)
        obs = None
        agent_idx = st.session_state.agents.index(agent)
        if "obs" in st.session_state and st.session_state.obs is not None:
            # If obs is a list of per-agent obs
            if (
                isinstance(st.session_state.obs, list)
                and len(st.session_state.obs) > agent_idx
            ):
                obs = st.session_state.obs[agent_idx]
            else:
                obs = st.session_state.obs
        if obs is None:
            st.warning("No observation available for this agent.")
        else:
            # If obs is a list, assume it's already split per modality
            if isinstance(obs, list) and all(
                isinstance(x, (list, tuple, np.ndarray)) for x in obs
            ):
                obs_list = obs
            else:
                obs_list = [obs]
            details = agent.get_fusion_details(obs_list)
            st.subheader("Raw Features (per modality)")
            st.json(details["raw_features"])
            if details["fusion_weights"] is not None:
                st.subheader("Fusion Weights")
                st.json(details["fusion_weights"])
            st.subheader("Fused Vector")
            st.json(details["fused_vector"])
            st.subheader("Q-values")
            st.json(details["q_values"])
            st.subheader("Loss Curve (Online Learning)")
            if hasattr(agent, "loss_history") and len(agent.loss_history) > 0:
                st.line_chart(list(agent.loss_history))
            else:
                st.warning("No loss history yet for this agent. Run simulation with online learning enabled.")

    if "env_states" not in st.session_state:
        st.session_state["env_states"] = [
            st.session_state.env.reset() if st.session_state.env else None
            for _ in st.session_state.agents
        ]
    if "step" not in st.session_state:
        st.session_state["step"] = 0

    # --- Simulation Step ---
    def simulate_step():
        obs = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        # RL agents interact with st.session_state.env, others get random obs
        for i, agent in enumerate(st.session_state.agents):
            state = st.session_state["env_states"][i] if st.session_state.env else None
            if agent_type == "Tabular Q-Learning":
                action = agent.act(state)
                next_state, reward, done = st.session_state.env.step(action)
                agent.observe(reward, next_state, done)
                st.session_state["env_states"][i] = (
                    next_state if not done else st.session_state.env.reset()
                )
                obs.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                reward_history[agent].append(reward)
            elif agent_type == "DQN RL":
                action = agent.act(state)
                next_state, reward, done = st.session_state.env.step(action)
                agent.observe(reward, next_state, done)
                st.session_state["env_states"][i] = (
                    next_state if not done else st.session_state.env.reset()
                )
                obs.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                reward_history[agent].append(reward)
            else:
                o = np.random.rand(2)
                action = agent.act(o)
                obs.append(o)
                actions.append(action)
                rewards.append(None)
                next_states.append(None)
                dones.append(False)
            # Law violation tracking
            # (Assume law_violations incremented in agent.act if exception)
        # --- SOM-based clustering and dynamic group formation ---
        # Extract features for all agents (for demo, use obs or obs vectors)
        try:
            feature_matrix = []
            for o in obs:
                # Flatten if needed
                if isinstance(o, (list, tuple, np.ndarray)):
                    arr = np.array(o).flatten()
                else:
                    arr = np.array([o])
                feature_matrix.append(arr)
            feature_matrix = np.array(feature_matrix)
            if feature_matrix.ndim == 1:
                feature_matrix = feature_matrix.reshape(-1, 1)
            # Run SOM clustering and update group assignments
            system.auto_group_by_som(feature_matrix)
        except Exception as ex:
            st.warning(f"SOM clustering failed: {ex}")
        # Knowledge sharing (simulate)
        src, dst = np.random.choice(agents, 2, replace=False)
        knowledge = {
            "foo": np.random.randint(0, 10),
            "privacy": np.random.choice(["public", "group-only", "private"]),
        }
        src.share_knowledge(knowledge, system=system, group=src.group)
        if knowledge["privacy"] != "private":
            knowledge_events.append((src, dst, knowledge, knowledge["privacy"]))
        # Group decision
        try:
            result = system.group_decision(
                obs,
                method=np.random.choice(["mean", "majority_vote", "weighted_mean"]),
                weights=np.ones(agent_count) / agent_count,
            )
            group_decisions.append((actions, result, True))
        except Exception:
            group_decisions.append((actions, None, False))

    st.session_state["step"] += 1
    if auto_run:
        time.sleep(2)

# --- Visualization ---
# Agent network
graph = nx.Graph()
for agent in st.session_state.agents:
    graph.add_node(
        agent.group + ":" + str(st.session_state.agents.index(agent)),
        knowledge=str(getattr(agent, "online_knowledge", {})),
        color="#1f78b4",
    )
for src, dst, knowledge, privacy in knowledge_events:
    graph.add_edge(
        src.group + ":" + str(agents.index(src)),
        dst.group + ":" + str(agents.index(dst)),
        label=str(knowledge),
        privacy=privacy,
    )

fig, ax = plt.subplots(figsize=(6, 4))
pos = nx.spring_layout(graph, seed=42)
colors = [graph.nodes[n]["color"] for n in graph.nodes]
nx.draw(graph, pos, with_labels=True, node_color=colors, ax=ax, node_size=900)
edge_labels = {(u, v): graph.edges[u, v]["privacy"] for u, v in graph.edges}
nx.draw_networkx_edge_labels(
    graph, pos, edge_labels=edge_labels, ax=ax, font_color="red"
)
st.pyplot(fig)

# --- Group & Leader Spatial Visualization ---
if hasattr(st.session_state.agents[0], "position"):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    group_ids = list(set(agent.group for agent in st.session_state.agents))
    group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
    for i, agent in enumerate(st.session_state.agents):
        color = group_colors.get(agent.group, "gray")
        x, y = agent.position if hasattr(agent, "position") else (0, 0)
        marker = "*" if getattr(agent, "is_leader", False) else "o"
        size = 250 if getattr(agent, "is_leader", False) else 100
        ax2.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=size,
            edgecolor="black",
            label=f"{agent.group}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}",
        )
        ax2.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
    # Unique legend
    handles = []
    for gid in group_ids:
        is_leader = any(
            getattr(agent, "is_leader", False) and agent.group == gid
            for agent in st.session_state.agents
        )
        marker = "*" if is_leader else "o"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=f"{gid}{' (Leader)' if is_leader else ''}",
                markerfacecolor=group_colors[gid],
                markeredgecolor="black",
                markersize=12 if is_leader else 8,
            )
        )
    ax2.legend(handles=handles, loc="best")
    ax2.set_title("Agent Positions by Group & Leader")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    st.pyplot(fig2)


# Per-Agent Knowledge Table
st.header("Agent Knowledge State")
data = [
    {
        "Agent": getattr(agent, 'group', 'Agent') + ":" + str(i),
        "Knowledge": str(getattr(agent, "online_knowledge", {})),
        "Law Violations": getattr(agent, "law_violations", 0),
    }
    for i, agent in enumerate(st.session_state.agents)
]
st.table(data)

# --- ANFIS Explainability Panel & Rule Controls ---
for i, agent in enumerate(st.session_state.agents):
    if hasattr(agent, 'model') and hasattr(agent.model, 'rule_weights') and hasattr(agent.model, 'centers'):
        with st.expander(f"ANFIS Agent {i+1} - Explainability & Rule Controls"):
            st.write("**Rule Weights:**", agent.model.rule_weights)
            st.write("**Centers:**", agent.model.centers)
            st.write("**Widths:**", agent.model.widths)
            st.markdown("---")
            st.subheader("Fuzzy Rule Management")
            # Experience Replay Settings Display
            if hasattr(agent, 'replay_enabled'):
                st.markdown(
                    f"**Experience Replay:** {'Enabled' if getattr(agent, 'replay_enabled', False) else 'Disabled'}  "+
                    f"Buffer Size: {getattr(agent, 'buffer_size', 'N/A')}, Batch Size: {getattr(agent, 'replay_batch', 'N/A')}"
                )
            # Add rule controls
            with st.form(f"add_rule_form_{i}", clear_on_submit=True):
                new_center = st.text_input("New Rule Center (comma-separated)", value=",")
                new_width = st.text_input("New Rule Width (comma-separated)", value="0.5,0.5")
                new_weight = st.number_input("New Rule Weight", value=0.0, step=0.1, format="%.2f")
                add_rule_btn = st.form_submit_button("Add Rule")
                if add_rule_btn:
                    try:
                        center = np.array([float(x) for x in new_center.split(",")]).reshape(1, -1)
                        width = np.array([float(x) for x in new_width.split(",")]).reshape(1, -1)
                        agent.model.add_rule(center, width, new_weight)
                        st.success("Rule added.")
                    except Exception as e:
                        st.error(f"Failed to add rule: {e}")
            # Remove rule controls
            remove_idx = st.number_input(f"Remove Rule Index (0 to {agent.model.n_rules-1})", min_value=0, max_value=max(agent.model.n_rules-1,0), value=0, step=1, key=f"remove_idx_{i}")
            if st.button(f"Remove Rule {remove_idx}", key=f"remove_btn_{i}"):
                agent.model.remove_rule(remove_idx)
                st.success(f"Rule {remove_idx} removed.")
            # Dynamic rule update
            if st.button("Dynamic Rule Update", key=f"dyn_rule_btn_{i}"):
                agent.model.dynamic_rule_update()
                st.info("Dynamic rule update triggered.")

# RL-specific: Reward plot and Q-table
if agent_type in ("Tabular Q-Learning", "DQN RL"):
    st.header("RL Agent Rewards")
    for i, agent in enumerate(st.session_state.agents):
        st.subheader(f"Agent {i+1} Reward History")
        r = list(reward_history.get(agent, []))
        st.line_chart(r if r else [0])
        if agent_type == "Tabular Q-Learning" and hasattr(agent, "q_table"):
            st.text(f"Q-table for Agent {i+1}:")
            st.write(agent.q_table)

st.info(
    f"This dashboard is running RL agents ({agent_type}) with live visualization. Use the sidebar to step or auto-run the simulation."
)
