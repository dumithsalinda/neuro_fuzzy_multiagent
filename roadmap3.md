# Plug-and-Play System Improvement Roadmap (roadmap3)

## 1. Dynamic Discovery & Registration
- **Environment Registry:** Auto-discovers all subclasses of `BaseEnvironment` in `src/env/`.
- **Agent Registry:** Registry for agent types, registered via class decorators or base class scanning.
- **Sensor/Actuator Plugins:** Base interface, auto-discover in `src/plugins/`.

## 2. Config-Driven and UI-Driven Selection
- Users select environments, agents, sensors, actuators via YAML/JSON config or dashboard UI.

## 3. Plugin API & Developer Guide
- Document interfaces for Environments, Agents, Sensors/Actuators. Provide templates/examples.

## 4. Dashboard Integration
- List all available modules, allow runtime selection/configuration, display docstrings/options.

## 5. Hot-Reloading (Optional)
- Support runtime reloading of plugins.

## 6. Auto-Documentation
- Generate/display plugin docs in dashboard and/or markdown/html.

## 7. Testing & CI
- Add tests for plugin registration, loading, error handling. CI checks for compatibility/docs.

---

# Developer Documentation Outline

## 1. Overview
The plug-and-play system allows you to add, remove, or swap agents, environments, sensors, and actuators without modifying core code. Simply drop your module in the correct folder and it will be auto-registered and available in the dashboard/config.

## 2. Directory Structure
```
src/
  env/           # Environment plugins (BaseEnvironment subclasses)
  core/          # Agent plugins (BaseAgent subclasses)
  plugins/       # Sensor/actuator plugins
dashboard/       # Dashboard code
```

## 3. How to Add a New Environment
1. Create a Python file in `src/env/` (e.g., `my_custom_env.py`).
2. Subclass `BaseEnvironment` and implement required methods.
3. Add a docstring describing your environment and its config options.
4. Your environment will be auto-discovered and appear in the dashboard/config.

## 4. How to Add a New Agent
1. Create a Python file in `src/core/` (e.g., `my_custom_agent.py`).
2. Subclass `BaseAgent` or use the agent registry decorator.
3. Implement required methods (`act`, `learn`, etc.).
4. Add a docstring describing your agent and its config options.
5. Your agent will be auto-discovered.

## 5. How to Add a Sensor/Actuator Plugin
1. Create a Python file in `src/plugins/` (e.g., `my_sensor.py`).
2. Subclass `BaseSensor` or `BaseActuator`.
3. Implement required methods (`read`, `write`, etc.).
4. Add a docstring and config options.
5. Your plugin will be auto-discovered.

## 6. How to Use Config Files
Example `config.yaml`:
```yaml
environment: MyCustomEnv
agent: MyCustomAgent
sensors:
  - MySensor
actuators:
  - MyActuator
```

## 7. Dashboard Usage
- On startup, the dashboard will list all available plugins.
- Select and configure plugins via the UI.
- View documentation and config options for each plugin.

## 8. Plugin API Reference
- **BaseEnvironment:** Required methods, expected signatures.
- **BaseAgent:** Required methods, expected signatures.
- **BaseSensor/BaseActuator:** Required methods, expected signatures.

## 9. Best Practices
- Keep plugins stateless or manage state carefully.
- Use clear, descriptive docstrings and type hints.
- Write tests for your plugins.

## 10. Troubleshooting
- Plugin not showing up? Check for typos, subclassing, and docstrings.
- Dashboard not updating? Restart or use hot-reload.

---

## Next Steps
1. Implement registry/discovery code for environments, agents, and plugins.
2. Update dashboard/config to use registries.
3. Write and publish developer documentation (as `DEVELOPER.md` or in the main README).
4. Test with sample plugins.
5. Iterate based on user/developer feedback.
