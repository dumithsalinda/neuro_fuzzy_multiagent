import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from src.core.agent import Agent
from src.core.multiagent import MultiAgentSystem
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from collections import deque

# --- System Setup ---
# Create neuro-fuzzy agents
agent_count = 5
nn_config = {'input_dim': 2, 'hidden_dim': 3, 'output_dim': 1}
fis_config = None  # Or provide fuzzy rule config
agents = [Agent(model=NeuroFuzzyHybrid(nn_config, fis_config)) for _ in range(agent_count)]

# Assign groups for demo privacy
for i, agent in enumerate(agents):
    agent.group = 'G1' if i < 3 else 'G2'

system = MultiAgentSystem(agents)

# --- State Tracking ---
knowledge_events = deque(maxlen=50)
group_decisions = deque(maxlen=20)
law_violations = {a: 0 for a in agents}

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
st.title("🤖 Neuro-Fuzzy Multi-Agent System Dashboard (Live)")

# --- Simulation Controls ---
run_sim = st.sidebar.button("Step Simulation")
auto_run = st.sidebar.checkbox("Auto Step (every 2s)")

if 'step' not in st.session_state:
    st.session_state['step'] = 0

# --- Simulation Step ---
def simulate_step():
    # Each agent acts on random observation
    obs = [np.random.rand(2) for _ in agents]
    actions = []
    for i, agent in enumerate(agents):
        try:
            action = agent.act(obs[i])
            actions.append(action)
        except Exception as e:
            law_violations[agent] += 1
            actions.append(None)
    # Random knowledge sharing event
    src, dst = np.random.choice(agents, 2, replace=False)
    knowledge = {'foo': np.random.randint(0, 10), 'privacy': np.random.choice(['public', 'group-only', 'private'])}
    prev_knowledge = dict(dst.__dict__.get('online_knowledge', {}))
    src.share_knowledge(knowledge, system=system, group=src.group)
    if knowledge['privacy'] != 'private':
        knowledge_events.append((src, dst, knowledge, knowledge['privacy']))
    # Group decision
    try:
        result = system.group_decision(obs, method=np.random.choice(['mean', 'majority_vote', 'weighted_mean']), weights=np.ones(agent_count)/agent_count)
        group_decisions.append((actions, result, True))
    except Exception:
        group_decisions.append((actions, None, False))

# --- Main Loop ---
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

# Knowledge Sharing Log
st.header("Knowledge Sharing Events")
for src, dst, knowledge, privacy in list(knowledge_events)[-10:][::-1]:
    st.write(f"{src.group}:{agents.index(src)} ➡️ {dst.group}:{agents.index(dst)} | {knowledge} | Privacy: {privacy}")

# Group Decisions Log
st.header("Recent Group Decisions")
for actions, result, legal in list(group_decisions)[-10:][::-1]:
    color = "green" if legal else "red"
    st.markdown(f"<span style='color:{color}'>Actions: {actions} → Result: {result} | {'Legal' if legal else 'Violated Law'}</span>", unsafe_allow_html=True)

st.info("This dashboard is now running on your real MultiAgentSystem. Use the sidebar to step or auto-run the simulation.")
