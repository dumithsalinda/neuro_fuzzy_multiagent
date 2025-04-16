import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- Demo Data Structures (Replace with real system integration) ---
class DummyAgent:
    def __init__(self, name, knowledge=None):
        self.name = name
        self.knowledge = knowledge or {}
        self.group = None
        self.law_violations = 0
        self.color = "#1f78b4"

# Simulated agent system (replace with hooks to your MultiAgentSystem)
agents = [DummyAgent(f"A{i+1}") for i in range(5)]
agents[0].knowledge = {"foo": 1}
agents[1].knowledge = {"bar": 2}
agents[2].knowledge = {"baz": 3}
agents[3].knowledge = {"foo": 1, "bar": 2}
agents[4].knowledge = {"baz": 3, "foo": 1}

# Simulated knowledge sharing events
knowledge_events = deque(maxlen=50)
knowledge_events.append(("A1", "A2", {"foo": 1}, "public"))
knowledge_events.append(("A2", "A3", {"bar": 2}, "public"))
knowledge_events.append(("A3", "A4", {"baz": 3}, "group-only"))

# Simulated group decisions
group_decisions = deque(maxlen=20)
group_decisions.append(("mean", [1, 2, 3, 2, 1], 1.8, True))
group_decisions.append(("majority_vote", [0, 1, 1, 2, 1], 1, True))
group_decisions.append(("weighted_mean", [2, 3, 2, 2, 2], 2.2, False))

# --- Streamlit Dashboard ---
st.set_page_config(page_title="Multi-Agent System Dashboard", layout="wide")
st.title("🤖 Neuro-Fuzzy Multi-Agent System Dashboard")

# Network Graph
st.header("Agent Interaction Network")
G = nx.Graph()
for agent in agents:
    G.add_node(agent.name, knowledge=agent.knowledge, color=agent.color)
for src, dst, knowledge, privacy in knowledge_events:
    G.add_edge(src, dst, label=str(knowledge), privacy=privacy)

fig, ax = plt.subplots(figsize=(6, 4))
pos = nx.spring_layout(G, seed=42)
colors = [G.nodes[n]["color"] for n in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=colors, ax=ax, node_size=900)
edge_labels = {(u, v): G.edges[u, v]["privacy"] for u, v in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color="red")
st.pyplot(fig)

# Per-Agent Knowledge Table
st.header("Agent Knowledge State")
agent_data = [{"Agent": a.name, "Knowledge": str(a.knowledge), "Law Violations": a.law_violations} for a in agents]
st.table(agent_data)

# Knowledge Sharing Log
st.header("Knowledge Sharing Events")
for src, dst, knowledge, privacy in list(knowledge_events)[-10:][::-1]:
    st.write(f"{src} ➡️ {dst} | {knowledge} | Privacy: {privacy}")

# Group Decisions Log
st.header("Recent Group Decisions")
for method, actions, result, legal in list(group_decisions)[-10:][::-1]:
    color = "green" if legal else "red"
    st.markdown(f"<span style='color:{color}'>[{method}] Actions: {actions} → Result: {result} | {'Legal' if legal else 'Violated Law'}</span>", unsafe_allow_html=True)

st.info("This dashboard is a live visualization template. Integrate with your real MultiAgentSystem for live data!")
