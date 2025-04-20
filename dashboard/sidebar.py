import streamlit as st

import importlib
import inspect
from src.env.registry import get_registered_environments
from src.core.agents.agent_registry import get_registered_agents
from src.core.agents.agent import Agent 
from src.plugins.registry import get_registered_sensors, get_registered_actuators

def render_sidebar():
    st.sidebar.title("Plug-and-Play System Controls")
    # --- Plugin Hot-Reload ---
    if st.sidebar.button("🔄 Reload Plugins"):
        importlib.invalidate_caches()
        importlib.reload(importlib.import_module("src.plugins.registry"))
        st.experimental_rerun()

    # --- Registries ---
    registered_envs = get_registered_environments()
    registered_agents = get_registered_agents()
    registered_sensors = get_registered_sensors()
    registered_actuators = get_registered_actuators()

    # --- Environment Selection ---
    env_names = list(registered_envs.keys())
    selected_env_name = st.sidebar.selectbox("Environment Type", env_names)
    env_cls = registered_envs[selected_env_name]
    st.sidebar.caption(f"**Env Doc:** {env_cls.__doc__}")
    # Environment config
    def get_init_params(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters.values() if p.name != 'self' and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    def get_param_value(param, label_prefix=""):
        label = f"{label_prefix}{param.name} ({param.annotation.__name__ if param.annotation != inspect._empty else 'Any'})"
        default = None if param.default == inspect._empty else param.default
        if param.annotation in [int, float]:
            return st.sidebar.number_input(label, value=default if default is not None else 0)
        elif param.annotation == bool:
            return st.sidebar.checkbox(label, value=default if default is not None else False)
        elif param.annotation == str:
            return st.sidebar.text_input(label, value=default if default is not None else "")
        else:
            return st.sidebar.text_input(label, value=str(default) if default is not None else "")
    env_kwargs = {}
    for param in get_init_params(env_cls):
        env_kwargs[param.name] = get_param_value(param, label_prefix="Env: ")
    st.session_state["selected_env_name"] = selected_env_name
    st.session_state["env_kwargs"] = env_kwargs

    # --- Agent Selection ---
    agent_names = list(registered_agents.keys())
    agent_count = st.sidebar.slider("Number of Agents", 1, 5, 3)
    selected_agent_names = [
        st.sidebar.selectbox(f"Agent {i+1} Type", agent_names, key=f"agent_type_{i}")
        for i in range(agent_count)
    ]
    for i, agent_cls_name in enumerate(selected_agent_names):
        agent_cls = registered_agents[agent_cls_name]
        st.sidebar.caption(f"**Agent {i+1} Doc:** {agent_cls.__doc__}")
    agent_kwargs_list = []
    for i, agent_cls_name in enumerate(selected_agent_names):
        kwargs = {}
        agent_cls = registered_agents[agent_cls_name]
        for param in get_init_params(agent_cls):
            kwargs[param.name] = get_param_value(param, label_prefix=f"Agent {i+1}: ")
        agent_kwargs_list.append(kwargs)
    st.session_state["agent_count"] = agent_count
    st.session_state["selected_agent_names"] = selected_agent_names
    st.session_state["agent_kwargs_list"] = agent_kwargs_list

    # --- Sensor Selection ---
    sensor_names = list(registered_sensors.keys())
    selected_sensor_name = st.sidebar.selectbox("Sensor Plugin", ["None"] + sensor_names)
    sensor_kwargs = {}
    if selected_sensor_name != "None":
        sensor_cls = registered_sensors[selected_sensor_name]
        st.sidebar.caption(f"**Sensor Doc:** {sensor_cls.__doc__}")
        for param in get_init_params(sensor_cls):
            sensor_kwargs[param.name] = get_param_value(param, label_prefix="Sensor: ")
    st.session_state["selected_sensor_name"] = selected_sensor_name
    st.session_state["sensor_kwargs"] = sensor_kwargs

    # --- Actuator Selection ---
    actuator_names = list(registered_actuators.keys())
    selected_actuator_name = st.sidebar.selectbox("Actuator Plugin", ["None"] + actuator_names)
    actuator_kwargs = {}
    if selected_actuator_name != "None":
        actuator_cls = registered_actuators[selected_actuator_name]
        st.sidebar.caption(f"**Actuator Doc:** {actuator_cls.__doc__}")
        for param in get_init_params(actuator_cls):
            actuator_kwargs[param.name] = get_param_value(param, label_prefix="Actuator: ")
    st.session_state["selected_actuator_name"] = selected_actuator_name
    st.session_state["actuator_kwargs"] = actuator_kwargs
