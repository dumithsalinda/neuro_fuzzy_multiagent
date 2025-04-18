import streamlit as st

def render_sidebar():
    st.sidebar.title("Environment & Agent Controls")
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        ["Tabular Q-Learning", "DQN RL", "Neuro-Fuzzy", "ANFIS Agent"],
        index=0,
    )
    agent_count = st.sidebar.slider("Number of Agents", 1, 10, 3)
    n_obstacles = st.sidebar.slider("Number of Obstacles", 0, 10, 2)
    st.session_state["agent_type"] = agent_type
    st.session_state["agent_count"] = agent_count
    st.session_state["n_obstacles"] = n_obstacles
