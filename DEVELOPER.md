# Plug-and-Play Developer Guide

Welcome to the plug-and-play neuro-fuzzy multi-agent platform! This guide explains how to add new environments, agents, and (optionally) plugins so that they are automatically discovered and available in the dashboard/config—**no core code changes required**.

---

## 1. Directory Structure

```
src/
  env/           # Environment plugins (BaseEnvironment subclasses)
  core/          # Agent plugins (Agent subclasses)
  plugins/       # (Future) Sensor/actuator plugins
```

---

## 2. Adding a New Environment

1. **Create a Python file in `src/env/`** (e.g., `my_custom_env.py`).
2. **Subclass `BaseEnvironment`** and implement required methods (`reset`, `step`, `get_observation`, etc).
3. **Add a docstring** describing your environment and its config options.
4. **Done!** Your environment will be auto-discovered and appear in the dashboard and config.

**Example:**
```python
from src.env.base_env import BaseEnvironment

class MyCustomEnv(BaseEnvironment):
    """A custom environment for demo purposes."""
    def reset(self):
        ...
    def step(self, action):
        ...
```

---

## 3. Adding a New Agent

1. **Create a Python file in `src/core/`** (e.g., `my_custom_agent.py`).
2. **Subclass `Agent`** and implement required methods (`act`, `learn`, etc).
3. **Add a docstring** describing your agent and its config options.
4. **Done!** Your agent will be auto-discovered and available for selection in the dashboard.

**Example:**
```python
from src.core.agent import Agent

class MyCustomAgent(Agent):
    """A custom agent for demo purposes."""
    def act(self, observation):
        ...
    def learn(self, *args, **kwargs):
        ...
```

---

## 4. How It Works
- The system uses dynamic registries (`src/env/registry.py` and `src/core/agent_registry.py`) to auto-discover all subclasses of `BaseEnvironment` and `Agent`.
- The dashboard UI and config loader use these registries to list all available modules.
- Instantiation is handled automatically based on the selected class names and sidebar/config parameters.

---

## 5. Tips & Best Practices
- Use clear, descriptive docstrings and type hints.
- Keep plugins stateless or manage state carefully.
- Write tests for your plugins.
- If your class requires special constructor arguments, document them in the class docstring.

---

## 6. Troubleshooting
- **Not showing up?**
  - Check for typos in class or file names.
  - Ensure your class subclasses the correct base (`BaseEnvironment` or `Agent`).
  - Ensure there are no import errors in your file.
- **Dashboard not updating?**
  - Restart the dashboard or use hot-reload if available.

---

## 7. Adding Plug-and-Play Sensors and Actuators

1. **Create a Python file in `src/plugins/`** (e.g., `my_sensor.py` or `my_actuator.py`).
2. **Subclass `BaseSensor` or `BaseActuator`** and implement the required methods (`read` for sensors, `write` for actuators).
3. **Add a docstring** describing your plugin and its config options.
4. **Done!** Your plugin will be auto-discovered and available for selection in the dashboard/config.

**Example Sensor:**
```python
from src.plugins.base_sensor import BaseSensor
class MySensor(BaseSensor):
    """A custom sensor."""
    def read(self):
        return 42
```

**Example Actuator:**
```python
from src.plugins.base_actuator import BaseActuator
class MyActuator(BaseActuator):
    """A custom actuator."""
    def write(self, command):
        print(command)
```

**How it works:**
- The system uses `src/plugins/registry.py` to auto-discover all subclasses of `BaseSensor` and `BaseActuator`.
- The dashboard and config loader use these registries to list all available plugins.
- Instantiation is handled automatically based on the selected class names and sidebar/config parameters.

**Troubleshooting:**
- Plugin not showing up? Check for typos, correct subclassing, and import errors.
- Dashboard not updating? Restart the dashboard or use hot-reload if available.

## 8. Example Usage Flows & Starter Templates

### Example 1: Using a Custom Sensor in the Dashboard
1. Create `src/plugins/my_sensor.py`:
    ```python
    from src.plugins.base_sensor import BaseSensor
    class MySensor(BaseSensor):
        """Returns a random value each time."""
        def read(self):
            import random
            return random.random()
    ```
2. Reload plugins in the dashboard (click "🔄 Reload Plugins").
3. Select `MySensor` as the sensor plugin in the sidebar.
4. Access it in your environment or agent:
    ```python
    sensor = st.session_state.get("sensor_plugin")
    if sensor:
        value = sensor.read()
    ```

### Example 2: Adding a New Agent
1. Create `src/core/my_agent.py`:
    ```python
    from src.core.agent import Agent
    class MyAgent(Agent):
        """Acts randomly."""
        def act(self, observation):
            import random
            return random.choice([0, 1])
        def learn(self, *args, **kwargs):
            pass
    ```
2. Select `MyAgent` in the dashboard; it will be auto-discovered.

### Example 3: Environment with Plugin Integration
```python
from src.env.base_env import BaseEnvironment
class MyEnv(BaseEnvironment):
    """Integrates with sensor and actuator plugins."""
    def __init__(self, ...):
        ...
    def step(self, action):
        sensor = st.session_state.get("sensor_plugin")
        actuator = st.session_state.get("actuator_plugin")
        if sensor:
            obs = sensor.read()
        if actuator:
            actuator.write(action)
        ...
```

---

- Contribute improvements or new features via PR!
