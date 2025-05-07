# Developer Guide: Neuro-Fuzzy Multi-Agent Framework

Welcome! This guide explains how to extend, customize, and contribute to the Neuro-Fuzzy Multiagent Framework. This package is developed and maintained independently; any operating system (OS) integration will be handled as a separate project. It consolidates all developer and plugin instructions into a single, up-to-date resource.

---

## 1. Welcome & Scope

This guide is for developers who want to extend the platform with new plugins, contribute code, or understand the system architecture. For contribution process, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 2. Project Structure

```
project_root/
│   README.md
│   requirements.txt
│   pyproject.toml
│
├── neuro_fuzzy_multiagent/
│   ├── core/           # Core agent classes and agent plugins
│   ├── plugins/        # Sensor/actuator plugins (BaseSensor, BaseActuator)
│   ├── env/            # Environment plugins (BaseEnvironment)
│   ├── utils/          # Utilities (e.g., config_loader.py)
│   └── ...
├── dashboard/          # Streamlit dashboard and UI
├── tests/              # All tests (mirrors main structure)
├── examples/           # Example scripts and configs
├── docs/               # Documentation
```

- **Agents:** `core/agents/`
- **Environments:** `env/`
- **Sensors/Actuators:** `plugins/`
- **Neural Networks:** `core/neural_network.py`
- **Tests:** `tests/` (mirrors main structure)

---

## 3. Plugin System Overview

The platform uses a drop-in plugin architecture for all core components:
- **Agents** (decision-makers)
- **Environments** (simulation/real-world wrappers)
- **Sensors/Actuators** (inputs/outputs)
- **Neural Networks** (plug-and-play models)

Plugins are auto-discovered at startup and can be selected via config files or the dashboard.

---

## 4. How to Add a Plugin

### Agents
- Place your agent class in `core/agents/`.
- Subclass `Agent` and use the `@register_plugin('agent')` decorator.
- Implement required methods (e.g., `act`, `reset`).
- Add docstrings and (optionally) metadata: `author`, `version`, `description`.

### Environments
- Place your environment class in `env/`.
- Subclass `BaseEnvironment` and use `@register_plugin('environment')`.
- Implement methods: `reset`, `step`, `get_observation`, `get_state`, `render`.
- Add docstrings and metadata.

### Sensors/Actuators
- Place in `plugins/`.
- Subclass `BaseSensor` or `BaseActuator` and use the appropriate decorator.
- Implement required methods (e.g., `read`, `actuate`).
- Add docstrings and metadata.

### Neural Networks
- Subclass `BaseNeuralNetwork` in `core/neural_network.py`.
- Use `@register_plugin('neural_network')` if needed.
- Implement `forward`, `backward`, `evolve` (optional), and `__init__`.

**Example:**
```python
from neuro_fuzzy_multiagent.core.agents.agent import Agent
@register_plugin('agent')
class MyAgent(Agent):
    """My custom agent."""
    author = "Your Name"
    version = "1.0"
    def act(self, obs):
        return 0
```

For advanced API details, see the [Plugin API Reference](PLUGIN_DOCS.md).

---

## 5. Configuration & Dashboard Integration

- Plugins are selected via YAML/JSON config files or interactively in the dashboard sidebar.
- Use the dashboard “🔄 Reload Plugins” button after adding new files or making changes.
- Plugin documentation is auto-generated and viewable/downloadable in the dashboard.
- Example YAML for agent config:
  ```yaml
  agent_type: NeuroFuzzyAgent
  nn_config:
    input_dim: 4
    hidden_dim: 8
    output_dim: 2
  ```

---

## 6. Coding Standards

- **Type Hints:** Required for all public functions and methods.
- **Docstrings:** Every class and function must have a clear docstring (purpose, args, returns).
- **Formatting:** Use Black, isort, and flake8 (see `pyproject.toml`).
- **Error Handling:** Use try/except for I/O, plugin loading, subprocess calls.
- **Logging:** Use the `logging` module for warnings, errors, and key actions.
- **Modularity:** Keep plugins stateless and modular where possible.

---

## 7. Testing

- All tests are in the `tests/` directory, mirroring the main structure.
- Add new tests for each feature or plugin.
- To run all tests:
  ```sh
  python3 -m pytest
  ```
- To run a specific test file:
  ```sh
  python3 -m pytest tests/core/test_my_agent.py
  ```

---

## 8. Troubleshooting & Best Practices

- **Validator Failing?**
  - Run `python3 scripts/validate_plugins.py` and check for [FAIL] lines. Fix missing methods as indicated.
- **Dashboard Not Showing Plugin?**
  - Check for typos, missing decorators, or missing docstrings.
- **Plugin Not Auto-Discovered?**
  - Ensure the file is in the correct directory and uses the correct base class.
- **Need More Examples?**
  - See `examples/` and the [Plugin API Reference](PLUGIN_DOCS.md).

---

## 9. References & Further Reading

- [Plugin API Reference](PLUGIN_DOCS.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Model Registry & Agent Integration](MODEL_REGISTRY_AND_AGENT.md)
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [README](../README.md)

---

Thank you for helping build the Neuro-Fuzzy Multi-Agent OS!
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
  - Regenerate with `PYTHONPATH=. python3 src/core/plugins/neuro_fuzzy_multiagent/generate_plugin_docs.py` and commit changes.
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
from neuro_fuzzy_multiagent.env.base_env import BaseEnvironment

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
from neuro_fuzzy_multiagent.core.agent import Agent

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
from neuro_fuzzy_multiagent.plugins.base_sensor import BaseSensor
class MySensor(BaseSensor):
    """A custom sensor."""
    def read(self):
        return 42
```

**Example Actuator:**

```python
from neuro_fuzzy_multiagent.plugins.base_actuator import BaseActuator
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
   from neuro_fuzzy_multiagent.plugins.base_sensor import BaseSensor
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
   from neuro_fuzzy_multiagent.core.agent import Agent
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
```
