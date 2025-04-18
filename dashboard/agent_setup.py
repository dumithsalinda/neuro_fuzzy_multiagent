import streamlit as st
from dashboard_env import initialize_env_and_agents

def setup_agents_and_env():
    agent_type = st.session_state.get("agent_type", "Tabular Q-Learning")
    agent_count = st.session_state.get("agent_count", 3)
    n_obstacles = st.session_state.get("n_obstacles", 2)
    if "env" not in st.session_state or st.sidebar.button("Reset Environment"):
        st.session_state.env, st.session_state.agents = initialize_env_and_agents(agent_type, agent_count, n_obstacles)
        st.session_state.obs = (
            st.session_state.env.reset() if st.session_state.env else None
        )
        st.session_state.done = False
        st.session_state.rewards = [0 for _ in range(agent_count)]
        st.session_state.step = 0
