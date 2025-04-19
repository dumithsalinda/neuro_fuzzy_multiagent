import streamlit as st

from src.core.multiagent import MultiAgentSystem
from src.core.som_cluster import SOMClusterer
import numpy as np
import streamlit as st

def som_group_agents():
    """
    Cluster agents using SOM based on their current observation vectors.
    Assign group labels and update MultiAgentSystem.
    """
    agents = st.session_state.get("agents", [])
    if not agents or "obs" not in st.session_state:
        st.warning("No agents or observations found for clustering.")
        return
    obs = st.session_state["obs"]
    # Ensure obs is a 2D array (n_agents, n_features)
    obs_matrix = np.array(obs)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape(-1, 1)
    # Get MultiAgentSystem instance
    mas = st.session_state.get("multiagent_system")
    if mas is None:
        mas = MultiAgentSystem(agents)
        st.session_state["multiagent_system"] = mas
    # Perform SOM-based grouping
    mas.auto_group_by_som(obs_matrix)
    st.session_state["groups"] = mas.groups
    st.success("Agents re-clustered using SOM!")

def simulate_step():
    # Integrated simulation logic with group knowledge sharing and collective actions
    import streamlit as st
    agents = st.session_state.get("agents", [])
    mas = st.session_state.get("multiagent_system")
    obs = st.session_state.get("obs")
    if not agents or mas is None or obs is None:
        st.warning("Simulation not initialized. Make sure agents and observations are available.")
        return
    # Group knowledge sharing if enabled
    if st.session_state.get("enable_group_knowledge", False):
        mas.share_group_knowledge(mode='average')
    # Collective action selection
    collective_mode = st.session_state.get("collective_mode", "individual")
    actions = []
    if collective_mode == "individual":
        actions = [agent.act(o) for agent, o in zip(agents, obs)]
    else:
        actions = mas.collective_action_selection(obs, mode=collective_mode)
    # Apply actions to environment (example, adapt as needed)
    env = st.session_state.get("env")
    if env is not None:
        next_obs, rewards, done, info = env.step(actions)
        st.session_state["obs"] = next_obs
        st.session_state["done"] = done
        st.session_state["rewards"] = rewards
        for i, agent in enumerate(agents):
            agent.observe(rewards[i])
    st.session_state["step"] = st.session_state.get("step", 0) + 1
