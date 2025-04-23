# Developer Guide: Neuro-Fuzzy Multi-Agent System

## Project Structure Overview

```
project_root/
│   README.md
│   DEVELOPER_GUIDE.md
│   PLUGIN_DOCS.md
│   requirements.txt
│   pyproject.toml
│
├── src/
│   ├── core/
│   │   ├── agents/           # Core agent classes and agent plugins
│   │   ├── experiment/       # Experiment management, analytics, MLflow integration
│   │   ├── plugins/          # Plugin system (discovery, validation, hot reload, registry, docs)
│   │   ├── environments/     # Environment plugins and wrappers
│   │   ├── ...
│   ├── env/                  # Base environment definitions
│   └── ...
├── tests/                    # All tests (mirrors src structure)
├── examples/                 # Example scripts and configs
```

## Adding New Agents, Environments, or Plugins

- **Agents:**
  - Place new agent classes in `src/core/agents`.
  - Use the `@register_plugin('agent')` decorator.
  - Define `__plugin_name__` and `__version__` attributes.
- **Environments:**
  - Place new environments in `src/core/environments` or `src/env`.
  - Use the `@register_plugin('environment')` decorator.
- **Sensors/Actuators:**
  - Place in `src/core/plugins` and use appropriate decorators.
- **Plugin Requirements:**
  - If your plugin needs extra dependencies, add a `requirements.txt` in the plugin folder.
  - Use `PluginDependencyManager` for validation and installation.

## Running Tests

- All tests are in the `tests/` directory, mirroring the `src/` structure.
- Run all tests:
  ```sh
  python3 -m pytest
  ```
- Run a specific test file:
  ```sh
  python3 -m pytest tests/experiment/test_result_analysis.py
  ```
- For src-layout compatibility, the test files add `src/` to `sys.path`.

## Coding Standards

- **Type Hints:** Required for all public functions and methods.
- **Docstrings:** Every class and function must have a clear docstring (purpose, args, returns).
- **Logging:** Use the `logging` module for all warnings, errors, and key actions.
- **Error Handling:** Use try/except blocks for all I/O, plugin loading, and subprocess calls.
- **Formatting:** Use Black, isort, and flake8 (see `pyproject.toml`).

## Example: Writing a Minimal Plugin

```python
from core.plugins.registration_utils import register_plugin

@register_plugin('agent')
class MyAgent:
    __plugin_name__ = "MyAgent"
    __version__ = "0.1.0"
    def act(self, obs):
        return 0  # Dummy action
```

- Add this file to `src/core/agents/`.
- Add a `requirements.txt` if dependencies are needed.
- Reload plugins via the dashboard or CLI utility.

---

# Plugin Authoring Guide

## Required Plugin Structure
- Must define `__plugin_name__` and `__version__` at the class or module level.
- Register with the appropriate decorator (see above).
- Place in the correct subdirectory (`agents`, `environments`, `plugins`).

## Dependency Management
- Place a `requirements.txt` in your plugin directory for extra dependencies.
- Use the dashboard or CLI to validate/install plugin requirements.

## Hot Reloading and Validation
- Use the dashboard “Reload Plugins” button or run:
  ```sh
  python3 src/core/plugins/hot_reload.py
  ```
- Use the CLI utility to validate plugins from a URL:
  ```sh
  python3 src/core/plugins/hot_reload.py --validate-url <URL> --plugin-type agent --plugin-name MyAgent
  ```

## Example: Minimal Valid Plugin
```python
from core.plugins.registration_utils import register_plugin

@register_plugin('environment')
class MyEnv:
    __plugin_name__ = "MyEnv"
    __version__ = "0.1.0"
    def reset(self):
        pass
    def step(self, action):
        pass
```

---

# Experiment Analytics & Reporting Usage

## Using ResultAnalyzer

```python
from core.experiment.result_analysis import ResultAnalyzer
analyzer = ResultAnalyzer(output_dir="results")

# Generate and save a Markdown report
report = analyzer.generate_report(config, metrics, run_id="abc123")
report_path = analyzer.save_report(report)

# Export metrics as JSON or CSV
analyzer.export_metrics(metrics, filename="metrics.json")
analyzer.export_metrics(metrics, filename="metrics.csv")

# Plot metric histories (requires matplotlib)
history = {"accuracy": [0.7, 0.8, 0.9], "loss": [0.5, 0.3, 0.2]}
analyzer.plot_metrics(history, filename="metrics.png")

# Compute summary statistics
stats = analyzer.summary_statistics(history)
print(stats)
```

---

For more details, see the main README, PLUGIN_DOCS.md, and in-code docstrings.
