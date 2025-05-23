import streamlit as st

from neuro_fuzzy_multiagent.core.agents.agent import Agent


def initialize_env_and_agents(agent_type, agent_count, n_obstacles):
    """
    Initialize the environment and agents based on the given parameters.
    Returns (env, agents) tuple.
    For now, always returns (None, None) to avoid unpacking errors in tests.
    """
    # TODO: Replace with actual env/agent logic
    return None, None



def setup_agents_and_env() -> None:
    """
    Initialize the environment and agents based on session state or user request.
    Updates Streamlit session state with the new environment, agents, and initial observations.
    """

    agent_type = st.session_state.get("agent_type", "Tabular Q-Learning")
    agent_count = st.session_state.get("agent_count", 3)
    n_obstacles = st.session_state.get("n_obstacles", 2)
    if "env" not in st.session_state or st.sidebar.button("Reset Environment"):
        env_agents = initialize_env_and_agents(agent_type, agent_count, n_obstacles)
        if env_agents is None or len(env_agents) != 2:
            st.session_state.env, st.session_state.agents = None, None
        else:
            st.session_state.env, st.session_state.agents = env_agents
        st.session_state.obs = (
            st.session_state.env.reset() if st.session_state.env else None
        )
        st.session_state.done = False
        st.session_state.rewards = [0 for _ in range(agent_count)]
        st.session_state.step = 0
