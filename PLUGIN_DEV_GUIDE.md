# Plugin Developer Guide

Welcome to the Neuro-Fuzzy Multiagent Plugin System!

## Overview
- Drop-in plugin architecture for environments, agents, sensors, and actuators.
- Auto-registration and dashboard/config integration.

## Directory Structure
- Environments: `src/env/`
- Agents: `src/core/agents/`
- Sensors/Actuators: `src/plugins/`

## How to Add a New Environment
1. Create a new Python file in `src/env/`.
2. Inherit from `BaseEnvironment`.
3. Define required methods and add a docstring.
4. Set `__version__` at the top of the file.

**Example:**
```python
from core.env.base_env import BaseEnvironment

class MyGridWorld(BaseEnvironment):
    """A simple grid world environment."""
    __version__ = "1.0"
    def reset(self): ...
    def step(self, action): ...
```

## How to Add a New Agent
1. Create a new Python file in `src/core/agents/`.
2. Inherit from `BaseAgent`.
3. Define required methods and add a docstring.
4. Set `__version__` at the top of the file.

**Example:**
```python
from core.agents.agent import Agent

class RandomAgent(Agent):
    """Selects random actions."""
    __version__ = "1.0"
    def act(self, obs): ...
    def explain_action(self, obs):
        return "Random choice"
```

## How to Add Sensor/Actuator Plugins
1. Create a new Python file in `src/plugins/`.
2. Inherit from the appropriate base class (e.g., `BaseSensor`, `BaseActuator`).
3. Define required methods and add a docstring.
4. Set `__version__` at the top of the file.

**Example:**
```python
from core.plugins.base_sensor import BaseSensor

class TemperatureSensor(BaseSensor):
    """Reads temperature from environment."""
    __version__ = "1.0"
    def read(self): ...
```

## How to Use Config Files
- Use YAML or JSON to select environments, agents, sensors, actuators.
- Example config provided in `examples/`.

## Dashboard Usage
- Use the dashboard to install, update, uninstall, and review plugins.
- Submit new plugins via the sidebar form.

## Plugin API Reference
- See `src/core/plugins/` for base classes and interface docs.

## Best Practices
- Always provide a docstring and `__version__`.
- Write clear, modular code.
- Include usage examples in your README.

## Troubleshooting
- Check logs for errors during registration or loading.
- Use the automated validator before submission:

  ```bash
  python3 scripts/validate_plugins.py
  ```
  - [OK] means your plugin implements all required methods.
  - [FAIL] means your plugin is missing required methods (see output for details).
- Common errors:
  - `ModuleNotFoundError`: Check your import paths and that your plugin is in the correct directory.
  - `TypeError: Can't instantiate abstract class`: You missed a required method.
  - Plugin not showing in dashboard/docs: Ensure you used the correct base class and decorator.

If you need more help, see the project README or contact the maintainers.
