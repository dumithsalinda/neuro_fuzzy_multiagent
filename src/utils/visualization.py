"""
Visualization utilities for agent rewards, explanations, and communication graphs.
"""
import matplotlib.pyplot as plt
import networkx as nx

def plot_rewards(reward_history, title="Agent Rewards Over Time"):
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_agent_explanations(explanation_history, agent_names=None, title="Agent Explanations (Q-values or Rule Activations)"):
    plt.figure(figsize=(10, 5))
    for idx, explanations in enumerate(explanation_history):
        label = agent_names[idx] if agent_names else f"Agent {idx}"
        plt.plot(explanations, label=label)
    plt.xlabel("Step")
    plt.ylabel("Explanation Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_communication_graph(messages_history, agent_count, title="Agent Communication Graph"):
    G = nx.DiGraph()
    for i in range(agent_count):
        G.add_node(f"Agent {i}")
    # messages_history: list of (from, to) tuples or dicts
    for step_msgs in messages_history:
        for msg in step_msgs:
            frm = msg.get("from")
            to = msg.get("to")
            if frm is not None and to is not None:
                G.add_edge(f"Agent {frm}", f"Agent {to}")
    plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue', arrowstyle='->', arrowsize=20)
    plt.title(title)
    plt.show()

def plot_rule_activations(rule_activations, rule_labels=None, agent_idx=0, title="Fuzzy Rule Activations"):
    """
    Plot a bar chart of rule activations for a single agent/time step.
    rule_activations: list or np.ndarray of activations
    rule_labels: list of rule names/labels (optional)
    agent_idx: int, index of the agent (optional)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    rule_activations = np.array(rule_activations)
    plt.figure(figsize=(8, 4))
    x = np.arange(len(rule_activations))
    labels = rule_labels if rule_labels is not None else [f"Rule {i}" for i in x]
    plt.bar(x, rule_activations)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.xlabel("Fuzzy Rule")
    plt.ylabel("Activation")
    plt.title(f"{title} (Agent {agent_idx})")
    plt.tight_layout()
    plt.show()
