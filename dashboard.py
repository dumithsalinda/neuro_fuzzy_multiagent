import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from src.core.agent import Agent
from src.core.multiagent import MultiAgentSystem
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.core.tabular_q_agent import TabularQLearningAgent
from src.core.dqn_agent import DQNAgent
from src.env.simple_env import SimpleDiscreteEnv, SimpleContinuousEnv

# --- Sidebar: Agent Type Selection ---
agent_type = st.sidebar.selectbox("Agent Type", ["Neuro-Fuzzy", "Tabular Q-Learning", "DQN RL"])
agent_count = st.sidebar.slider("Number of Agents", 1, 5, 3)

# --- System Setup ---
if agent_type == "Neuro-Fuzzy":
    nn_config = {'input_dim': 2, 'hidden_dim': 3, 'output_dim': 1}
    fis_config = None
    agents = [Agent(model=NeuroFuzzyHybrid(nn_config, fis_config)) for _ in range(agent_count)]
    env = None
elif agent_type == "Tabular Q-Learning":
    n_states, n_actions = 5, 2
    agents = [TabularQLearningAgent(n_states=n_states, n_actions=n_actions) for _ in range(agent_count)]
    env = SimpleDiscreteEnv(n_states=n_states, n_actions=n_actions)
elif agent_type == "DQN RL":
    state_dim, action_dim = 2, 4
    agents = [DQNAgent(state_dim=state_dim, action_dim=action_dim) for _ in range(agent_count)]
    env = SimpleContinuousEnv()
else:
    st.error("Unknown agent type.")
    st.stop()

for i, agent in enumerate(agents):
    agent.group = 'G1' if i < (agent_count // 2 + 1) else 'G2'

system = MultiAgentSystem(agents)

# --- State Tracking ---
knowledge_events = deque(maxlen=50)
group_decisions = deque(maxlen=20)
law_violations = {a: 0 for a in agents}
reward_history = {a: deque(maxlen=30) for a in agents}

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
st.title(f"🤖 Multi-Agent System Dashboard ({agent_type})")

run_sim = st.sidebar.button("Step Simulation")
auto_run = st.sidebar.checkbox("Auto Step (every 2s)")

if 'env_states' not in st.session_state:
    st.session_state['env_states'] = [env.reset() if env else None for _ in agents]
if 'step' not in st.session_state:
    st.session_state['step'] = 0

# --- Simulation Step ---
def simulate_step():
    obs = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    # RL agents interact with env, others get random obs
    for i, agent in enumerate(agents):
        state = st.session_state['env_states'][i] if env else None
        if agent_type == "Tabular Q-Learning":
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.observe(reward, next_state, done)
            st.session_state['env_states'][i] = next_state if not done else env.reset()
            obs.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            reward_history[agent].append(reward)
        elif agent_type == "DQN RL":
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.observe(reward, next_state, done)
            st.session_state['env_states'][i] = next_state if not done else env.reset()
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
    knowledge = {'foo': np.random.randint(0, 10), 'privacy': np.random.choice(['public', 'group-only', 'private'])}
    src.share_knowledge(knowledge, system=system, group=src.group)
    if knowledge['privacy'] != 'private':
        knowledge_events.append((src, dst, knowledge, knowledge['privacy']))
    # Group decision
    try:
        result = system.group_decision(obs, method=np.random.choice(['mean', 'majority_vote', 'weighted_mean']), weights=np.ones(agent_count)/agent_count)
        group_decisions.append((actions, result, True))
    except Exception:
        group_decisions.append((actions, None, False))

if run_sim or auto_run:
    simulate_step()
    st.session_state['step'] += 1
    if auto_run:
        time.sleep(2)

# --- Visualization ---
# Agent network
graph = nx.Graph()
for agent in agents:
    graph.add_node(agent.group + ':' + str(agents.index(agent)), knowledge=str(getattr(agent, 'online_knowledge', {})), color="#1f78b4")
for src, dst, knowledge, privacy in knowledge_events:
    graph.add_edge(src.group + ':' + str(agents.index(src)), dst.group + ':' + str(agents.index(dst)), label=str(knowledge), privacy=privacy)

fig, ax = plt.subplots(figsize=(6, 4))
pos = nx.spring_layout(graph, seed=42)
colors = [graph.nodes[n]["color"] for n in graph.nodes]
nx.draw(graph, pos, with_labels=True, node_color=colors, ax=ax, node_size=900)
edge_labels = {(u, v): graph.edges[u, v]["privacy"] for u, v in graph.edges}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_color="red")
st.pyplot(fig)

# Per-Agent Knowledge Table
st.header("Agent Knowledge State")
data = [{"Agent": agent.group + ':' + str(i), "Knowledge": str(getattr(agent, 'online_knowledge', {})), "Law Violations": law_violations[agent]} for i, agent in enumerate(agents)]
st.table(data)

# RL-specific: Reward plot and Q-table
if agent_type in ("Tabular Q-Learning", "DQN RL"):
    st.header("RL Agent Rewards")
    for i, agent in enumerate(agents):
        st.subheader(f"Agent {i+1} Reward History")
        r = list(reward_history[agent])
        st.line_chart(r if r else [0])
        if agent_type == "Tabular Q-Learning":
            st.text(f"Q-table for Agent {i+1}:")
            st.write(agent.q_table)

# Knowledge Sharing Log
st.header("Knowledge Sharing Events")
for src, dst, knowledge, privacy in list(knowledge_events)[-10:][::-1]:
    st.write(f"{src.group}:{agents.index(src)} ➡️ {dst.group}:{agents.index(dst)} | {knowledge} | Privacy: {privacy}")

# Group Decisions Log
st.header("Recent Group Decisions")
for actions, result, legal in list(group_decisions)[-10:][::-1]:
    color = "green" if legal else "red"
    st.markdown(f"<span style='color:{color}'>Actions: {actions} → Result: {result} | {'Legal' if legal else 'Violated Law'}</span>", unsafe_allow_html=True)

st.info(f"This dashboard is running RL agents ({agent_type}) with live visualization. Use the sidebar to step or auto-run the simulation.")
