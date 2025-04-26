# Developer Guide: Plug-and-Play Neuro-Fuzzy Multi-Agent System

Welcome! This guide helps you extend the neuro-fuzzy multi-agent platform by adding new agents, environments, sensors, actuators, or neural networks—no core code changes required.


Welcome to the plug-and-play neuro-fuzzy multi-agent platform! This guide explains how to add new environments, agents, neural networks, sensors, and actuators so that they are automatically discovered and available in the dashboard/config—**no core code changes required**.

---

## 1. Directory Structure

```
src/
  env/           # Environment plugins (BaseEnvironment subclasses)
  core/          # Agent plugins (Agent subclasses, neural networks)
  plugins/       # Sensor/actuator plugins (BaseSensor, BaseActuator)
  utils/         # Utilities (e.g., config_loader.py)
```

---

## 2. Plugin API Reference

### Environments
- Subclass `BaseEnvironment` in `src/env/`.
- Use the `@register_plugin('environment')` decorator (already on base class).
- Implement: `reset`, `step`, `get_observation`, `get_state`, `render`.
- Add a docstring describing config options for auto-docs.

### Agents
- Subclass `Agent` in `src/core/agents/`.
- Use the `@register_plugin('agent')` decorator (already on base class).
- Add docstrings for config options and main methods.

### Sensors/Actuators
- Subclass `BaseSensor` or `BaseActuator` in `src/plugins/`.
- Use the `@register_plugin('sensor')` or `@register_plugin('actuator')` decorator (already on base class).
- Add docstrings for auto-docs.

### Neural Networks (Plug-and-Play)
- Subclass `BaseNeuralNetwork` in `src/core/neural_network.py`.
- Register using `@register_plugin('neural_network')` if you add this plugin type.
- Required methods: `forward`, `backward`, `evolve` (optional), `__init__` with config params.
- Add docstrings for config options.
- Example config (YAML):
  ```yaml
  nn_config:
    nn_type: FeedforwardNeuralNetwork
    input_dim: 4
    hidden_dim: 8
    output_dim: 2
    activation: tanh
  ```

---

## 3. Auto-Discovery & Registration
- All plugins are auto-discovered by scanning their respective directories.
- Use the `@register_plugin` decorator (already on base classes) or ensure your class is a subclass of the appropriate base class.
- No need to manually edit registries.

---

## 4. Config & Dashboard Integration
- Plugins can be selected via YAML/JSON config files or interactively in the dashboard sidebar.
- Use the dashboard “🔄 Reload Plugins” button after adding new files or making changes.
- Plugin documentation is auto-generated and viewable/downloadable in the dashboard sidebar.

---

## 5. Plugin Auto-Documentation
- Run `PYTHONPATH=. python3 src/core/plugins/generate_plugin_docs.py` to regenerate `PLUGIN_DOCS.md`.
- The dashboard sidebar displays the latest plugin documentation and provides a download button.
- Add docstrings to your plugin classes and methods for best documentation.

---

## 6. Hot-Reloading Plugins
- Use the dashboard “🔄 Reload Plugins” button to reload all plugins at runtime.
- This will clear and repopulate all plugin registries and update the dashboard UI.
- Errors during reload are displayed in the sidebar.

---

## 7. Continuous Integration (CI)
- All plugin system tests run automatically on push/PR via GitHub Actions (`.github/workflows/plugin_ci.yml`).
- CI checks that `PLUGIN_DOCS.md` is up to date. Regenerate and commit if needed.

---

## 8. Best Practices
- Always subclass the correct base and use the `@register_plugin` decorator (already present on base classes).
- Add comprehensive docstrings for all classes and methods.
- Test your plugin using the dashboard and `pytest tests/plugins/test_plugin_system.py`.
- Use the CLI validator to check your plugin:
  ```bash
  python3 scripts/validate_plugins.py
  ```
  - [OK] means your plugin implements all required methods.
  - [FAIL] means your plugin is missing required methods (see output for details).
- Keep your plugin config options clear and documented.
- Use hot-reload to iterate quickly during development.

---

## 9. Troubleshooting
- **Plugin not showing up?**
  - Check for typos in class or file names.
  - Ensure your class subclasses the correct base and is in the right directory.
  - Ensure there are no import errors in your file (check dashboard sidebar for errors on reload).
- **Dashboard not updating?**
  - Use the “🔄 Reload Plugins” button.
  - Check for errors in the sidebar after reload.
- **Docs not updating?**
  - Regenerate with `PYTHONPATH=. python3 src/core/plugins/generate_plugin_docs.py` and commit changes.
- **CI failing?**
  - Run tests locally with `pytest tests/plugins/test_plugin_system.py` and ensure docs are up to date.
- **Validator failing?**
  - Run `python3 scripts/validate_plugins.py` and check for [FAIL] lines. Fix missing methods as indicated.

---

## New Features
- **Audit Logging:** Use `log_human_decision` from `core.plugins.human_approval_log` to record human approvals/denials for agent actions. See tests/plugins/test_human_approval_log.py for usage.
- **Custom Explanation Registry:** Use `register_explanation` from `core.plugins.explanation_registry` to register custom explanation functions for agent types. See tests/plugins/test_explanation_registry.py for usage.

---

## Example: Agent Plugin
```python
from core.agents.agent import Agent
class MyAgent(Agent):
    """My custom agent."""
    __version__ = "1.0"
    def act(self, obs): ...
    def explain_action(self, obs):
        return "My explanation"
```

## Example: Sensor Plugin
```python
from core.plugins.base_sensor import BaseSensor
class MySensor(BaseSensor):
    __version__ = "1.0"
    def read(self): ...
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
from core.env.base_env import BaseEnvironment
class MyEnv(BaseEnvironment):
    """Integrates with sensor and actuator plugins."""
    __version__ = "1.0"
    def __init__(self, ...):
    def reset(self): ...
    def step(self, action): ...
```
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
