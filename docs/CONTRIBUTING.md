# Contributing to the Neuro-Fuzzy Multi-Agent OS

Thank you for your interest in contributing! This project is building an intelligent operating system platform, fundamentally different from traditional OSes. Please read this guide fully before contributing.

---

## 1. Welcome & Project Vision

The Neuro-Fuzzy Multi-Agent OS is designed for adaptive resource management, self-learning device recognition, proactive optimization, and plug-and-play AI. For a full overview, see the [README](../README.md).

---

## 2. Getting Started

- **Clone the repo:**
  ```sh
  git clone <repo-url>
  cd neuro_fuzzy_multiagent
  ```
- **Install dependencies:**
  ```sh
  pip install -r requirements.txt
  ```
- **Run tests:**
  ```sh
  pytest tests/
  ```
- For more setup instructions, see the [Developer Guide](DEVELOPER.md).

---

## 3. Ways to Contribute

- Add new agents, environments, sensors, or actuators as plugins
- Improve documentation or examples
- Report bugs or suggest features
- Submit tests or code quality improvements
- Participate in discussions and roadmap planning

Open issues or join discussions on GitHub for help or suggestions.

---

## 4. How to Add a Plugin

The platform is designed for easy plug-and-play extension. To add a plugin:

- **Agents:** Place new agent classes in `src/core/agents/` and subclass `Agent`. Implement required methods and add a docstring. See [Developer Guide](DEVELOPER.md) for details.
- **Environments:** Place new environments in `src/env/` and subclass `BaseEnvironment`.
- **Sensors/Actuators:** Place in `src/plugins/` and subclass `BaseSensor` or `BaseActuator`.
- **Metadata:** Optionally add `author`, `version`, and `description` attributes for dashboard display.
- **Testing:** All plugins are auto-discovered and available in the dashboard/config with no core code changes.

For advanced plugin development and API details, see the [Plugin Developer Guide](PLUGIN_DEV_GUIDE.md) and [Plugin API Reference](PLUGIN_DOCS.md).

---

## 5. Code Style & Guidelines

- Follow PEP8 and project conventions.
- Use descriptive docstrings and type hints for all public functions/classes.
- Keep plugins modular and stateless if possible.
- Write tests for new features (see [tests/](../tests)).
- Use Black, isort, and flake8 for formatting (see `pyproject.toml`).

---

## 6. Submitting Changes

1. Fork the repo and create a new branch.
2. Make your changes and add tests/docs as needed.
3. Open a pull request (PR) with a clear description.
4. Address any feedback from reviewers.

**PR Checklist:**
- [ ] Code follows style guidelines and passes tests
- [ ] Documentation is updated if needed
- [ ] New features include tests
- [ ] No unrelated changes are included

---

## 7. Getting Help

- Open an issue or discussion on GitHub for questions or support.
- For sensitive matters (e.g., security), contact the maintainers at [REPLACE_WITH_EMAIL].

---

## 8. References & Links

- [Developer Guide](DEVELOPER.md)
- [Plugin Developer Guide](PLUGIN_DEV_GUIDE.md)
- [Plugin API Reference](PLUGIN_DOCS.md)
- [Model Registry & Agent Integration](MODEL_REGISTRY_AND_AGENT.md)
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [README](../README.md)

---

Thank you for helping make the Neuro-Fuzzy Multi-Agent OS better!
