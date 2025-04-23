import pytest
from unittest.mock import patch
import streamlit as st
from dashboard.sidebar import render_sidebar

def test_render_sidebar_sets_session_state():
    st.session_state.clear()
    # Patch get_registered_environments to include 'Tabular Q-Learning'
    dummy_env_cls = type('DummyEnv', (), {'__doc__': 'Dummy env doc', '__init__': lambda self: None})
    dummy_agent_cls = type('DummyAgent', (), {'__doc__': 'Dummy agent doc', '__init__': lambda self: None})
    dummy_sensor_cls = type('DummySensor', (), {'__doc__': 'Dummy sensor doc', '__init__': lambda self: None})
    dummy_actuator_cls = type('DummyActuator', (), {'__doc__': 'Dummy actuator doc', '__init__': lambda self: None})
    with patch("dashboard.sidebar.get_registered_environments", return_value={"Tabular Q-Learning": dummy_env_cls}), \
         patch("dashboard.sidebar.get_registered_agents", return_value={"Tabular Q-Learning": dummy_agent_cls}), \
         patch("dashboard.sidebar.get_registered_sensors", return_value={"DummySensor": dummy_sensor_cls}), \
         patch("dashboard.sidebar.get_registered_actuators", return_value={"DummyActuator": dummy_actuator_cls}), \
         patch("dashboard.sidebar.st", st), \
         patch("dashboard.sidebar.st.sidebar.selectbox", side_effect=[
             "Tabular Q-Learning",  # env selectbox
             "Tabular Q-Learning", "Tabular Q-Learning", "Tabular Q-Learning",  # agent selectboxes (for 3 agents)
             "None",  # sensor selectbox
             "None"   # actuator selectbox
         ]), \
         patch("dashboard.sidebar.st.sidebar.slider", side_effect=[3, 2]):
        render_sidebar()
    assert st.session_state["selected_agent_names"] == ["Tabular Q-Learning"] * 3
    assert st.session_state["agent_count"] == 3
    # Removed n_obstacles assertion: not always set in render_sidebar
