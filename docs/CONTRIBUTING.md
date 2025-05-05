# Contributing to the Neuro-Fuzzy Multi-Agent Platform

Thank you for your interest in contributing! This project is evolving into an intelligent operating system (OS) platform, fundamentally different from traditional OSes. It features:
- Neuro-fuzzy agents for adaptive resource management and system optimization
- Self-learning device recognition via installable AI drivers (trained models)
- Proactive and context-aware system optimization
- Plug-and-play installation of new AI capabilities
- Prototyping of intelligent agents (e.g., adaptive resource manager, smart USB handler)
- AI driver/model installation tool for seamless integration of new learning models

The project welcomes new agents, environments, sensors, actuators, documentation, and improvements from the community.

## Ways to Contribute
- Add new agents, environments, sensors, or actuators as plugins
- Improve documentation or examples
- Report bugs or suggest features
- Submit tests or code quality improvements

---

## Adding a New Plugin

1. **Agents:**
    - Create a new file in `neuro_fuzzy_multiagent/core/`.
    - Subclass `Agent` and implement required methods.
    - Add a docstring and (optionally) metadata attributes (`author`, `version`, `description`).
2. **Environments:**
    - Create a new file in `neuro_fuzzy_multiagent/env/`.
    - Subclass `BaseEnvironment` and implement required methods.
    - Add a docstring and (optionally) metadata attributes.
3. **Sensors/Actuators:**
    - Create a new file in `neuro_fuzzy_multiagent/plugins/`.
    - Subclass `BaseSensor` or `BaseActuator` and implement required methods.
    - Add a docstring and (optionally) metadata attributes.
4. **Test:**
    - Your plugin will be auto-discovered and available in the dashboard/config with no core code changes.

---

## Plugin Metadata (Optional)
You may add metadata to your plugin classes for display in the dashboard:
```python
class MyPlugin(...):
    """My plugin docstring."""
    author = "Your Name"
    version = "1.0"
    description = "A short description."
```

---

## Code Style & Guidelines
- Follow PEP8 and project conventions.
- Use descriptive docstrings and type hints.
- Keep plugins modular and stateless if possible.
- Write tests for new features.

---

## Submitting Changes
1. Fork the repo and create a new branch.
2. Make your changes and add tests/docs as needed.
3. Open a pull request with a clear description.
4. Address any feedback from reviewers.

---

## Need Help?
Open an issue or start a discussion!
