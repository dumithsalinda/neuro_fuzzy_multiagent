import streamlit as st

st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
import time
from collections import deque

import matplotlib.pyplot as plt
from collections import defaultdict, deque
import requests

# --- RL Reward History ---
reward_history = defaultdict(lambda: deque(maxlen=100))

import networkx as nx
import numpy as np

from src.core.agent import Agent
from src.core.dqn_agent import DQNAgent
from src.core.multiagent import MultiAgentSystem
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.core.tabular_q_agent import TabularQLearningAgent
from src.env.simple_env import SimpleContinuousEnv, SimpleDiscreteEnv

# --- Sidebar: Agent Type Selection & Parameter Tuning ---
env_type = st.sidebar.selectbox(
    "Environment Type", ["Gridworld", "Adversarial Gridworld", "Resource Collection"]
)

agent_type = None  # Always defined for safety

if env_type == "Gridworld":
    agent_type = None
    agent_count = st.sidebar.slider("Number of Agents", 1, 5, 3)
    n_obstacles = st.sidebar.slider("Number of Obstacles", 0, 10, 2)
    agent_type_choices = ["Neuro-Fuzzy", "Tabular Q-Learning", "DQN RL"]
    agent_types = [
        st.sidebar.selectbox(
            f"Agent {i+1} Type", agent_type_choices, key=f"agent_type_{i}"
        )
        for i in range(agent_count)
    ]
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
if (env_type == "Gridworld" and any(t in ["Tabular Q-Learning", "DQN RL"] for t in agent_types)) or \
   (env_type != "Gridworld" and agent_type in ["Tabular Q-Learning", "DQN RL"]):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Agent Parameters**")
    alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("Discount Factor (gamma)", 0.0, 1.0, 0.99, 0.01)
    epsilon = st.sidebar.slider("Exploration Rate (epsilon)", 0.0, 1.0, 0.1, 0.01)
else:
    alpha = gamma = epsilon = None


# --- Session State Setup ---
def initialize_env_and_agents():
    # Recreate env and agents based on sidebar selections
    agents = []
    env = None
    if env_type == "Gridworld":
        from src.env.multiagent_gridworld import MultiAgentGridworldEnv

        n_states, n_actions = 5, 4
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
        env = MultiAgentGridworldEnv(
            grid_size=5, n_agents=agent_count, n_obstacles=n_obstacles
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
    else:
        agents = [
            TabularQLearningAgent(
                n_states=5, n_actions=4, alpha=alpha, gamma=gamma, epsilon=epsilon
            )
            for _ in range(agent_count)
        ]
        env = None
    return env, agents


if "env" not in st.session_state or st.sidebar.button("Reset Environment"):
    st.session_state.env, st.session_state.agents = initialize_env_and_agents()
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

    actions = [to_scalar_action(agent.act(obs)) for agent, obs in zip(agents, st.session_state.obs)]
    next_obs, rewards, done = env.step(actions)
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
                st.session_state["online_learning_log"] = f"Online update failed for agent {i}: {ex}"
    st.session_state.obs = next_obs
    st.session_state.rewards = rewards
    st.session_state.done = done
    st.session_state.step += 1


# --- Main UI ---
st.title(f"🤖 Multi-Agent System Dashboard ({agent_type})")
tabs = st.tabs(["Simulation", "Analytics"])

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
        data = {"agent_id": manual_agent_id, "input": manual_input, "target": manual_target}
        r = requests.post(api_url, data=data, headers=headers, timeout=2)
        st.sidebar.success(f"Online update sent: {r.json()}")
    except Exception as ex:
        st.sidebar.error(f"Failed to send online update: {ex}")


with tabs[0]:
    st.write(f"Step: {st.session_state.step}")
    st.write(f"Done: {st.session_state.done}")
    if st.button("Step Simulation", key="main_step_sim"):
        simulate_step()
    if st.session_state.obs is not None:
        st.write("### Agent Observations")
        st.write(st.session_state.obs)
        st.write("### Rewards")
        st.write(st.session_state.rewards)

    # Table of agent knowledge/law violations (example)
    st.write("### Agent Knowledge Table")
    for i, agent in enumerate(st.session_state.agents):
        ag_type = agent_types[i] if agent_types else agent.__class__.__name__
        last_msg = getattr(agent, 'last_message', None)
        if isinstance(last_msg, tuple) and len(last_msg) == 2 and isinstance(last_msg[0], dict):
            last_msg_display = last_msg[0]
        else:
            last_msg_display = last_msg
        st.write(
            f"Agent {i} [{ag_type}] (Step {st.session_state.step}): Knowledge: {getattr(agent, 'online_knowledge', {})} Law Violations: {getattr(agent, 'law_violations', 0)} Last Msg: {last_msg_display}"
        )
        if hasattr(agent, "epsilon"):
            agent.epsilon = epsilon

agent_count = len(st.session_state.agents)
for i, agent in enumerate(st.session_state.agents):
    agent.group = "G1" if i < (agent_count // 2 + 1) else "G2"

system = MultiAgentSystem(st.session_state.agents)

# --- State Tracking ---
knowledge_events = deque(maxlen=50)
group_decisions = deque(maxlen=20)

# --- Streamlit UI ---
st.title(f"🤖 Multi-Agent System Dashboard ({agent_type})")

tabs = st.tabs(["Simulation", "Analytics"])

with tabs[0]:
    run_sim = st.sidebar.button("Step Simulation", key="sidebar_step_sim")
    auto_run = st.sidebar.checkbox("Auto Step (every 2s)")

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

# Per-Agent Knowledge Table
st.header("Agent Knowledge State")
data = [
    {
        "Agent": agent.group + ":" + str(i),
        "Knowledge": str(getattr(agent, "online_knowledge", {})),
        "Law Violations": getattr(agent, "law_violations", 0),
    }
    for i, agent in enumerate(st.session_state.agents)
]
st.table(data)

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

# Knowledge Sharing Log
st.header("Knowledge Sharing Events")
for src, dst, knowledge, privacy in list(knowledge_events)[-10:][::-1]:
    st.write(
        f"{src.group}:{agents.index(src)} ➡️ {dst.group}:{agents.index(dst)} | {knowledge} | Privacy: {privacy}"
    )

# Group Decisions Log
st.header("Recent Group Decisions")
for actions, result, legal in list(group_decisions)[-10:][::-1]:
    color = "green" if legal else "red"
    st.markdown(
        f"<span style='color:{color}'>Actions: {actions} → Result: {result} | {'Legal' if legal else 'Violated Law'}</span>",
        unsafe_allow_html=True,
    )

st.info(
    f"This dashboard is running RL agents ({agent_type}) with live visualization. Use the sidebar to step or auto-run the simulation."
)
