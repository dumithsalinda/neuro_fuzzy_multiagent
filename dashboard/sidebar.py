import streamlit as st

import importlib
import inspect
from src.env.registry import get_registered_environments
from src.core.agents.agent_registry import get_registered_agents
from src.core.agents.agent import Agent
from src.plugins.registry import get_registered_sensors, get_registered_actuators

import streamlit as st
import inspect
from src.env.registry import get_registered_environments
from src.core.agents.agent_registry import get_registered_agents
from src.plugins.registry import get_registered_sensors, get_registered_actuators


def render_sidebar():
    registered_envs = get_registered_environments()
    registered_agents = get_registered_agents()
    registered_sensors = get_registered_sensors()
    registered_actuators = get_registered_actuators()

    env_names = list(registered_envs.keys())
    selected_env_name = st.sidebar.selectbox("Environment Type", env_names)
    if selected_env_name in registered_envs:
        env_cls = registered_envs[selected_env_name]
        st.sidebar.caption(f"**Env Doc:** {env_cls.__doc__}")

    agent_names = list(registered_agents.keys())
    agent_count = st.sidebar.slider("Number of Agents", 1, 5, 3)
    selected_agent_names = [
        st.sidebar.selectbox(f"Agent {i+1} Type", agent_names, key=f"agent_type_{i}")
        for i in range(agent_count)
    ]
    for i, agent_cls_name in enumerate(selected_agent_names):
        if agent_cls_name in registered_agents:
            agent_cls = registered_agents[agent_cls_name]
            st.sidebar.caption(f"**Agent {i+1} Doc:** {agent_cls.__doc__}")
    n_obstacles = st.sidebar.slider("Number of Obstacles", 0, 10, 2)

    sensor_names = list(registered_sensors.keys())
    actuator_names = list(registered_actuators.keys())
    selected_sensor_name = st.sidebar.selectbox(
        "Sensor Plugin", ["None"] + sensor_names
    )
    if selected_sensor_name != "None":
        sensor_cls = registered_sensors[selected_sensor_name]
        st.sidebar.caption(f"**Sensor Doc:** {sensor_cls.__doc__}")
    selected_actuator_name = st.sidebar.selectbox(
        "Actuator Plugin", ["None"] + actuator_names
    )
    if selected_actuator_name != "None":
        actuator_cls = registered_actuators[selected_actuator_name]
        st.sidebar.caption(f"**Actuator Doc:** {actuator_cls.__doc__}")

    # --- Ensure test compatibility: always set these keys ---
    st.session_state["selected_agent_names"] = selected_agent_names
    st.session_state["agent_count"] = agent_count

    def get_init_params(cls):
        sig = inspect.signature(cls.__init__)
        return [
            p
            for p in sig.parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]

    def get_param_value(param, label_prefix=""):
        label = f"{label_prefix}{param.name} ({param.annotation.__name__ if param.annotation != inspect._empty else 'Any'})"
        default = None if param.default == inspect._empty else param.default
        if param.annotation in [int, float]:
            return st.sidebar.number_input(
                label, value=default if default is not None else 0
            )
        elif param.annotation == bool:
            return st.sidebar.checkbox(
                label, value=default if default is not None else False
            )
        elif param.annotation == str:
            return st.sidebar.text_input(
                label, value=default if default is not None else ""
            )
        else:
            return st.sidebar.text_input(
                label, value=str(default) if default is not None else ""
            )

    env_kwargs = {}
    if selected_env_name in registered_envs:
        env_cls = registered_envs[selected_env_name]
        for param in get_init_params(env_cls):
            env_kwargs[param.name] = get_param_value(param, label_prefix="Env: ")

    agent_kwargs_list = []
    for i, agent_cls_name in enumerate(selected_agent_names):
        kwargs = {}
        if agent_cls_name in registered_agents:
            agent_cls = registered_agents[agent_cls_name]
            for param in get_init_params(agent_cls):
                kwargs[param.name] = get_param_value(
                    param, label_prefix=f"Agent {i+1}: "
                )
        agent_kwargs_list.append(kwargs)

    sensor_kwargs = {}
    if selected_sensor_name != "None":
        sensor_cls = registered_sensors[selected_sensor_name]
        for param in get_init_params(sensor_cls):
            sensor_kwargs[param.name] = get_param_value(param, label_prefix="Sensor: ")

    actuator_kwargs = {}
    if selected_actuator_name != "None":
        actuator_cls = registered_actuators[selected_actuator_name]
        for param in get_init_params(actuator_cls):
            actuator_kwargs[param.name] = get_param_value(
                param, label_prefix="Actuator: "
            )

    # Store plugin instances in session state for use elsewhere
    if selected_sensor_name != "None":
        st.session_state["sensor_plugin"] = registered_sensors[selected_sensor_name](
            **sensor_kwargs
        )
    else:
        st.session_state["sensor_plugin"] = None
    if selected_actuator_name != "None":
        st.session_state["actuator_plugin"] = registered_actuators[
            selected_actuator_name
        ](**actuator_kwargs)
    else:
        st.session_state["actuator_plugin"] = None

    # Return all configs for use in main app logic
    return {
        "env_name": selected_env_name,
        "env_kwargs": env_kwargs,
        "agent_count": agent_count,
        "selected_agent_names": selected_agent_names,
        "agent_kwargs_list": agent_kwargs_list,
        "n_obstacles": n_obstacles,
        "sensor_plugin": st.session_state["sensor_plugin"],
        "actuator_plugin": st.session_state["actuator_plugin"],
    }

    # --- END OF FUNCTION: All sidebar logic is now above and only runs once per call. ---

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
        return [
            p
            for p in sig.parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]

    def get_param_value(param, label_prefix=""):
        label = f"{label_prefix}{param.name} ({param.annotation.__name__ if param.annotation != inspect._empty else 'Any'})"
        default = None if param.default == inspect._empty else param.default
        if param.annotation in [int, float]:
            return st.sidebar.number_input(
                label, value=default if default is not None else 0
            )
        elif param.annotation == bool:
            return st.sidebar.checkbox(
                label, value=default if default is not None else False
            )
        elif param.annotation == str:
            return st.sidebar.text_input(
                label, value=default if default is not None else ""
            )
        else:
            return st.sidebar.text_input(
                label, value=str(default) if default is not None else ""
            )

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

    # --- Sensor Selection (Multi) ---
    sensor_names = list(registered_sensors.keys())
    selected_sensor_names = st.sidebar.multiselect("Sensor Plugins", sensor_names)
    sensor_kwargs_list = []
    for i, sensor_name in enumerate(selected_sensor_names):
        sensor_cls = registered_sensors[sensor_name]
        st.sidebar.caption(f"**Sensor {i+1} Doc:** {sensor_cls.__doc__}")
        kwargs = {}
        for param in get_init_params(sensor_cls):
            kwargs[param.name] = get_param_value(param, label_prefix=f"Sensor {i+1}: ")
        sensor_kwargs_list.append(kwargs)
    st.session_state["selected_sensor_names"] = selected_sensor_names
    st.session_state["sensor_kwargs_list"] = sensor_kwargs_list

    # --- Actuator Selection (Multi) ---
    actuator_names = list(registered_actuators.keys())
    selected_actuator_names = st.sidebar.multiselect("Actuator Plugins", actuator_names)
    actuator_kwargs_list = []
    for i, actuator_name in enumerate(selected_actuator_names):
        actuator_cls = registered_actuators[actuator_name]
        st.sidebar.caption(f"**Actuator {i+1} Doc:** {actuator_cls.__doc__}")
        kwargs = {}
        for param in get_init_params(actuator_cls):
            kwargs[param.name] = get_param_value(
                param, label_prefix=f"Actuator {i+1}: "
            )
        actuator_kwargs_list.append(kwargs)
    st.session_state["selected_actuator_names"] = selected_actuator_names
    st.session_state["actuator_kwargs_list"] = actuator_kwargs_list

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "All configuration is hot-reloadable. Use the controls above to set up your simulation and agents."
    )

    return (
        selected_env_name,
        env_kwargs,
        agent_count,
        selected_agent_names,
        agent_kwargs_list,
        selected_sensor_names,
        sensor_kwargs_list,
        selected_actuator_names,
        actuator_kwargs_list,
    )


# --- Modular Sidebar Controls ---
def anfis_replay_sidebar() -> tuple:
    """
    Render sidebar controls for ANFIS agent online/continual learning experience replay.
    Returns a tuple of (replay_enabled, buffer_size, batch_size).
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ANFIS Agent Online/Continual Learning**")
    replay_enabled = st.sidebar.checkbox("Enable Experience Replay", value=True)
    buffer_size = st.sidebar.number_input(
        "Replay Buffer Size", min_value=10, max_value=1000, value=100, step=10
    )
    batch_size = st.sidebar.number_input(
        "Replay Batch Size", min_value=1, max_value=128, value=8, step=1
    )
    return replay_enabled, buffer_size, batch_size


def realworld_sidebar() -> None:
    """
    Render sidebar controls for real-world integration (robot, API, IoT sensor).
    Handles observe/act modes and displays results.
    """
    import json
    import requests

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Real-World Integration**")
    realworld_types = ["robot", "api", "iot_sensor"]
    realworld_mode = st.sidebar.selectbox("Mode", ["Observe", "Act"])
    realworld_type = st.sidebar.selectbox("Type", realworld_types)
    realworld_url = st.sidebar.text_input(
        "Endpoint URL", value="http://localhost:9000/data"
    )
    realworld_config = {"url": realworld_url}
    realworld_config_json = json.dumps(realworld_config)
    realworld_action = (
        st.sidebar.text_input("Action (JSON)", value='{"move": "forward"}')
        if realworld_mode == "Act"
        else None
    )
    realworld_result_placeholder = st.sidebar.empty()
    if st.sidebar.button(f"Real-World {realworld_mode}"):
        try:
            api_url = f"http://localhost:8000/realworld/{'observe' if realworld_mode == 'Observe' else 'act'}"
            headers = {"X-API-Key": "mysecretkey"}
            data = (
                {"config": realworld_config_json, "source_type": realworld_type}
                if realworld_mode == "Observe"
                else {
                    "config": realworld_config_json,
                    "target_type": realworld_type,
                    "action": realworld_action,
                }
            )
            r = requests.post(api_url, data=data, headers=headers, timeout=3)
            realworld_result_placeholder.success(f"Result: {r.text}")
        except Exception as ex:
            realworld_result_placeholder.error(f"Failed: {ex}")
