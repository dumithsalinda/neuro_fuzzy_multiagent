# Neuro-Fuzzy Multi-Agent System

---

## 🟢 Beginner’s Guide

### What is this project?
This is a flexible platform for building, testing, and experimenting with teams of intelligent agents (like little robots or AI programs) that can sense, act, learn, and work together. It supports easy “plug-and-play” for new agent types, environments, sensors, and actuators—no need to change core code!

### What can you do with it?
- Simulate teams of smart agents in different environments.
- Add your own agents, environments, sensors, or actuators with just a new file.
- Use a friendly dashboard to run experiments, visualize results, and interact with agents in real time.
- Integrate real-world data or human feedback for advanced experiments.

### How do you use it?
1. **Install requirements:**  
   ```sh
   pip install -r requirements.txt
   ```
2. **Start the dashboard:**  
   ```sh
   streamlit run dashboard.py
   ```
3. **Try it out:**  
   - Select environments, agents, sensors, and actuators from the sidebar.
   - Run simulations, see results, and interact live.
   - Try adding your own plugin (see DEVELOPER.md for step-by-step instructions).

### Where to learn more?
- **PLUGIN_DOCS.md:** Full auto-generated list of all available environments, agents, neural networks, sensors, and actuators, including config options and docstrings. Regenerate with `python generate_plugin_docs.py` after adding plugins.
- **DEVELOPER.md:** How to add new agents, environments, sensors, or actuators.
- **CONTRIBUTING.md:** How to contribute your own plugins or improvements.
- **Dashboard UI:** Tooltips and docstrings are shown for every selectable module.

### Troubleshooting
- If you see errors, check that all dependencies are installed.
- Use the “🔄 Reload Plugins” button in the dashboard after adding new files.
- For more help, open an issue or discussion on the project repository.

---

## Overview
A robust, environment-independent, dynamic self-organizing neuro-fuzzy multi-agent system with:
- **Unbreakable Laws:** Hard constraints on agent and group behavior.
- **Modular Neuro-Fuzzy Agents:** Hybrid models, self-organization, and extensibility.
- **Multi-Agent Collaboration:** Communication, consensus, and privacy-aware knowledge sharing.
- **Online Learning:** Agents learn from web resources and integrate knowledge.
- **Advanced Group Decision-Making:** Mean, weighted mean, majority vote, custom aggregation.
- **Thorough Testing & Extensibility:** Easily add new laws, agent types, or aggregation methods.

## Features
- Law registration, enforcement, and compliance for all actions and knowledge.
- Privacy-aware knowledge sharing (public, private, group-only, recipient-list).
- Flexible group decision-making with law enforcement.
- Online learning and knowledge integration from the internet.
- Modular, extensible agent and system design.
- **Multi-Modal Fusion Agent:** Accepts multiple modalities (e.g., text, image) as input, fuses features for action selection. Supports dashboard demo and simulation with random or real multi-modal data.
- **Model Management & Versioning:** Save, load, list, and evaluate agent models directly from the dashboard. Compare agent versions on evaluation tasks.
- **Batch Experimentation:** Run parameter sweeps and batch experiments; export and log results.
- **Adversarial Testing:** Perturb agent observations with various adversarial methods for robustness analysis.
- **Dashboard:** Interactive dashboard for simulation, analytics, manual feedback, batch experiments, model management, and multi-modal agent demo.

## Real-Time Data Integration & Human-in-the-Loop (HITL)

- **Live Data Sources:** Agents/environments can receive real-time values from REST APIs, MQTT streams, or mock sensors, injected directly via the dashboard.
- **Human-in-the-Loop Controls:** Pause, resume, stop, override actions, and provide manual rewards or demonstrations during experiments.
- **Plug-and-Play:** Supports seamless switching between simulated and real-world data sources for agent control and observation.
- **Architecture:** Environments implement `set_external_input(agent_idx, value)` to accept live values (see `src/env/base_env.py`).
- **Dashboard UI:** Select, connect, and monitor live data sources and human feedback in real time.

## Core Modules
- `src/core/agent.py`: Agent logic, knowledge sharing, law compliance, extensibility.
- `src/core/multiagent.py`: Multi-agent system, group decision, privacy-aware knowledge sharing.
- `src/core/laws.py`: Law registration, enforcement, and compliance.
- `src/core/online_learning.py`: Online learning mixin for web knowledge.

## Multi-Modal Fusion Agent

### Overview
Supports agents that take multiple modalities (e.g., text, image) as input and use a fusion network (concatenation; extensible to attention/gating) for action selection. Includes:
- Dashboard demo tab for interactive testing with random multi-modal input.
- Simulation support for multi-modal agents (random or real data).
- Easily extendable to new fusion methods and modalities.

### How to Use
- Select "Multi-Modal Fusion Agent" in the dashboard sidebar to configure and simulate with multi-modal agents.
- Use the "Multi-Modal Demo" tab to test agent action selection on random input.
- To use real multi-modal data, update your environment to return `[modality1, modality2, ...]` per agent.

### Quickstart
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```sh
   streamlit run dashboard/main.py
   ```
3. (Optional) Run the API server:
   ```sh
   uvicorn agent_api:app --reload
   ```
4. Access the dashboard at http://localhost:8501

---

## Docker Usage

Build and run the dashboard in a container:

```sh
# Build the Docker image
sudo docker build -t neuro-fuzzy-multiagent .

# Run the dashboard on port 8501
sudo docker run -p 8501:8501 neuro-fuzzy-multiagent
```

---

## Dashboard & Google Sheets Walkthrough

- **Simulation & Batch Experiments:**
  - Run single or batch experiments, set agent counts, seeds, and steps.
  - View analytics, export results as CSV/JSON.
- **Google Sheets Integration:**
  - Upload your Google service account key (JSON), enter Spreadsheet ID and Worksheet name.
  - Sync intervention logs to/from Google Sheets for collaborative editing.
- **Explainability & Playback:**
  - Inspect fuzzy rules, group structure, and agent decisions.
  - Use scenario playback to step through and analyze episodes interactively.

---

## Troubleshooting & FAQ

- **Google Sheets Errors:** Ensure your service account has access to the target spreadsheet and the JSON key is valid.
- **Docker Issues:** Make sure Docker is installed and running, and ports are not in use.
- **Missing Dependencies:** Double-check `requirements.txt` for all needed Python packages.
- **Streamlit/Matplotlib/NetworkX errors:** Install missing packages with `pip install ...` or rebuild the Docker image.
- **Performance:** Enable "Fast Mode" in the dashboard for large-scale experiments.

---

## Running Tests

To run all tests (requires pytest):

```sh
pytest tests/
```

All dashboard and core functions are covered by the test suite. If you encounter any errors, see the test output for details and troubleshooting.

### Test Suite Organization (2025)

Tests are now organized by feature/module for clarity and maintainability:

```
tests/
  agents/         # Tests for agent classes and agent-related logic
  core/           # Core system, law, online learning, and plugin tests
  environments/   # Environment and environment-controller tests
  management/     # Multi-agent system, group, dashboard, and management tests
  integration/    # Cross-module, knowledge sharing, and integration tests
```

- Place new agent tests in `tests/agents/`, environment tests in `tests/environments/`, etc.
- Each subfolder contains an `__init__.py` for pytest discovery.
- Example: To test a new agent, add `test_my_agent.py` to `tests/agents/`.
- Update imports to use relative paths (e.g., `from .dummy_agent import DummyAgent`).

---

## Meta-Learning & AutoML

The platform supports meta-learning and automated hyperparameter optimization for agents:

### MetaAgent
- Dynamically selects the best-performing agent type (e.g., DQN, Q-Learning) based on recent performance.
- Usage:
  ```python
  from src.core.agents.meta_agent import MetaAgent
  meta = MetaAgent(candidate_agents=[(DQNAgent, {...}), (TabularQLearningAgent, {...})])
  ```
- The MetaAgent tries each candidate for a set number of steps (exploration), then switches to the best.

### HyperparameterOptimizer
- Simple optimizer for agent hyperparameters (random search; extensible to Bayesian/evolutionary).
- Usage:
  ```python
  from src.core.agents.hpo import HyperparameterOptimizer
  hpo = HyperparameterOptimizer(param_space, eval_fn)
  best_params, best_score = hpo.optimize(n_trials=100)
  ```
- Use to tune agent configs for best performance.

See `src/core/agents/meta_agent.py` and `src/core/agents/hpo.py` for details and extension points.

## Example Usage
```python
from core.agent import Agent
from core.multiagent import MultiAgentSystem
from core.laws import register_law, LawViolation

# Register a custom law
@register_law
def no_negative_actions(action, state=None):
    return (action >= 0).all()

# Create agents and system
agents = [Agent(model=None) for _ in range(3)]
system = MultiAgentSystem(agents)

# Group decision (mean)
obs = [None, None, None]
try:
    consensus = system.coordinate_actions(obs)
except LawViolation as e:
    print("Law broken:", e)

# Share knowledge (public)
knowledge = {'rule': 'IF x > 0 THEN y = 1', 'privacy': 'public'}
agents[0].share_knowledge(knowledge, system=system)
```

## Extensibility
- **Add a new law:** Define a function and decorate with `@register_law`.
- **Add a new agent type:** Subclass `Agent` and override methods as needed.
- **Add a group aggregation method:** Use `MultiAgentSystem.group_decision(..., method="custom", custom_fn=...)`.

## Project Goals
- Environment-independent, dynamic, and robust multi-agent learning.
- Strict compliance with unbreakable laws.
- Modular, extensible, and well-documented codebase.

---
For more details, see module/class/method docstrings in the source code.
trained_model = transfer_learning(env1, env2, model, feat, steps=10)
```

### Integration with Neuro-Fuzzy Hybrid Model
You can use the `NeuroFuzzyHybrid` model in place of the neural network in transfer learning:
```python
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
# ... setup fuzzy system and configs ...
model = NeuroFuzzyHybrid(nn_config, fis_config)
trained_model = transfer_learning(env1, env2, model, feat, steps=10)
```

### Testing
- Run all tests with `pytest` in the project root.
- See `tests/test_environment.py` for environment and transfer learning tests.
- See `tests/test_neuro_fuzzy.py` for hybrid model tests.

## Requirements
See requirements.txt
