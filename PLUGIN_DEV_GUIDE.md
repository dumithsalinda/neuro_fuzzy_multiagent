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

## How to Add a New Agent
1. Create a new Python file in `src/core/agents/`.
2. Inherit from `BaseAgent`.
3. Define required methods and add a docstring.
4. Set `__version__` at the top of the file.

## How to Add Sensor/Actuator Plugins
1. Create a new Python file in `src/plugins/`.
2. Inherit from the appropriate base class (e.g., `BaseSensor`, `BaseActuator`).
3. Define required methods and add a docstring.
4. Set `__version__` at the top of the file.

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
- Use the automated validator before submission.

---
For more details, see the project README or contact the maintainers.
