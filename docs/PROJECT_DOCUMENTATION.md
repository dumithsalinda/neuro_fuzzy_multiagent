# Neuro-Fuzzy Multi-Agent System: Project Documentation

## 1. Project Overview

This project is evolving into an **intelligent operating system (OS)** platform, fundamentally different from traditional OSes by embedding neuro-fuzzy multiagent intelligence at its core. The system enables:

- Adaptive resource management and optimization using neuro-fuzzy agents
- Self-learning device recognition through installable AI drivers (trained models)
- Proactive system optimization and context-aware user experiences
- Plug-and-play installation of new AI capabilities, extending intelligence to new hardware or domains
- Prototyping of intelligent agents (e.g., adaptive resource manager, smart USB handler)
- AI driver/model installation tool for seamless integration of new learning models and dynamic adaptation

It integrates neural networks, fuzzy logic, and human feedback to enable robust, explainable, and collaborative AI agents. The system supports multi-modal input (vision, audio, text, sensors), distributed/cloud execution, real-time dashboard visualization, and advanced robustness/safety features.

---

## 2. Directory Structure & Key Components

```
neuro_fuzzy_multiagent/
├── core/                # Core logic: agents, neural networks, utilities
├── plugins/             # Sensor/actuator plugins
├── env/                 # Environment plugins
├── dashboard/           # Streamlit dashboard UI and supporting modules
├── examples/            # Example/demo scripts for various use-cases
├── tests/               # Comprehensive test suite for all modules
├── README.md            # Project overview and quickstart
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization support
├── ...                  # Utility scripts and configs
```

For detailed development and contribution instructions, see:
- [Developer Guide](DEVELOPER.md)
- [Contribution Guide](CONTRIBUTING.md)

---

## Documentation

- [Developer Guide](DEVELOPER.md) — How to extend the OS, create plugins, and contribute code
- [Contribution Guide](CONTRIBUTING.md) — How to report issues, suggest features, and contribute
- [Plugin API Reference](PLUGIN_DOCS.md) — Plugin types and configuration options
- [Model Registry & Agent Integration](MODEL_REGISTRY_AND_AGENT.md)
- [Distributed Execution Guide](README_DISTRIBUTED.md)
- [API Docs & Generation](README_API_DOCS.md)

---

## Main Modules & Their Roles

### Core Modules
- **core/**: Agents, neural/fuzzy logic, utilities
- **plugins/**: Sensor/actuator plugins
- **env/**: Environment plugins
- **dashboard/**: Streamlit UI, analytics, and collaboration
- **examples/**: Example/demo scripts
- **tests/**: Comprehensive test suite

For full extension instructions, see [Developer Guide](DEVELOPER.md). For contribution process, see [Contribution Guide](CONTRIBUTING.md).

- **internet_learning/**: Integrates web/video knowledge for agent learning.
- **multiagent/**: Collaboration and environment abstractions.

---

## 4. Dynamic Agent Management & Hot-Reloading

### Runtime Agent Addition, Removal, and Configuration Reload

This system supports **plug-and-play agent architecture** with dynamic agent management and hot-reloading of agent configurations. Agents can be added, removed, or reconfigured at runtime without restarting the system.

#### Main APIs:

- **AgentManager** (`src/core/agent_manager.py`):
  - `add_agent(config, group=None)`: Add an agent at runtime using a YAML/JSON/dict config.
  - `remove_agent(agent)`: Remove an agent from the system.
  - `reload_agent_config(agent, config_or_path)`: Reload an agent's configuration from file or dict at runtime.
- **NeuroFuzzyAgent** (`src/core/agent.py`):
  - `reload_config(config_or_path)`: Reload this agent's configuration in place.

#### Example Usage

```python
from neuro_fuzzy_multiagent.core.agent_manager import AgentManager
from neuro_fuzzy_multiagent.core.message_bus import MessageBus

bus = MessageBus()
manager = AgentManager(bus=bus)

# Add agent from config dict or YAML/JSON file
config = {
    'agent_type': 'NeuroFuzzyAgent',
    'nn_config': {'input_dim': 2, 'hidden_dim': 4, 'output_dim': 1},
    'fis_config': None,
    'meta_controller': {},
    'universal_fuzzy_layer': None
}
agent = manager.add_agent(config)

# Hot-reload agent config at runtime
new_config_path = 'path/to/new_config.yaml'  # or pass a dict
manager.reload_agent_config(agent, new_config_path)

# Remove agent
manager.remove_agent(agent)
```

#### Example YAML Agent Config

```yaml
agent_type: NeuroFuzzyAgent
nn_config:
  input_dim: 3
  hidden_dim: 6
  output_dim: 2
fis_config: null
meta_controller: {}
universal_fuzzy_layer: null
```

#### Notes & Best Practices

- Supported config formats: YAML, JSON, or Python dict.
- Only keys present in the config will be updated; missing keys retain their previous values.
- For neural network changes, the model is re-instantiated with new parameters.
- For fuzzy system changes, rules are regenerated if the required keys are present.
- Avoid changing the agent type at runtime; create a new agent instead.

---

## 5. Key Workflows

### Evolving Fuzzy Rule Bases & Adaptive Hybrid Learning

The system supports dynamic evolution and adaptation of fuzzy rule bases, as well as automatic switching between neural, fuzzy, and hybrid inference modes for robust agent performance.

#### 1. Evolving Fuzzy Rule Bases

- Use `agent.evolve_rules(recent_inputs, min_avg_activation)` to prune dynamic fuzzy rules whose average firing strength (activation) on recent data falls below the specified threshold.
- This maintains a compact, relevant, and adaptive fuzzy rule base for each agent.

**Example:**

```python
# Prune weak rules using recent data
recent_inputs = [np.array([0.1, 0.9]), np.array([0.2, 0.8]), ...]
pruned = agent.evolve_rules(recent_inputs, min_avg_activation=0.05)
```

#### 2. Adaptive Hybrid Learning & Mode Switching

- Use `agent.auto_switch_mode(error_history, thresholds)` to automatically switch the agent's inference mode based on recent error history.
- The agent will switch between 'neural', 'fuzzy', and 'hybrid' modes to optimize performance and robustness.

**Example:**

```python
# Switch mode if error is high
error_history = [0.12, 0.11, 0.15, 0.18, 0.2]
current_mode = agent.auto_switch_mode(error_history)
```

- Both features are available in the `NeuroFuzzyAgent` API, and can be integrated into training loops or experiment logic for self-organizing, adaptive agents.

### A. Agent Setup & Training

- Agents are instantiated via the dashboard or scripts.
- Supports multi-modal input (vision, audio, text, etc.).
- Training can be neural (deep RL), fuzzy (rule-based), or neuro-fuzzy (fusion).
- Human feedback can add/adjust fuzzy rules (dynamic, never overwriting core rules).

### B. Dashboard Usage

- Streamlit-based UI for:
  - Agent selection and configuration
  - Real-time simulation and visualization (positions, groups, SOM grid)
  - Batch/parallel experiments with analytics and downloads
  - Human-agent chat, feedback, and explainability
  - Intervention logging and knowledge sharing

### C. Robustness & Safety

- Wrappers for adding noise/adversarial perturbations to observations/actions.
- Runtime safety monitoring with constraint violation logging.
- Dashboard toggles and analytics for robustness features.

### D. Distributed/Cloud Execution

- Ray-based scripts and configs for scaling agents across clusters.
- Dockerfile and deployment scripts for reproducibility.

---

## 5. Dependencies & Running the Project

- **Python 3.8+** (see `requirements.txt`)
- **Major dependencies:** `numpy`, `torch`, `streamlit`, `matplotlib`, `seaborn`, `pandas`, `gspread`, `pytest`, `networkx`, `scikit-learn`, etc.
- **To run the dashboard:**
  ```sh
  pip install -r requirements.txt
  streamlit run neuro_fuzzy_multiagent/dashboard/neuro_fuzzy_multiagent/main.py
  ```
- **To run tests:**
  ```sh
  pytest
  ```
- **To run examples:**
  ```sh
  python neuro_fuzzy_multiagent/examples/demo_multiagent_rl.py
  ```

---

## 6. Extensibility & Customization

- **Agents:** Add new agent types in `src/core/`, register in dashboard.
- **Fuzzy Rules:** Extend fuzzy logic and rule management in `fuzzy_system.py`.
- **Human Feedback:** Customize chat and feedback parsing in `dashboard/chat.py`.
- **Robustness:** Add new wrappers in `robustness_wrappers.py`.
- **Distributed Execution:** Modify Ray configs and scripts as needed.
- **Visualization:** Add new analytics/plots in `dashboard/visualization.py`.

---

## 7. Testing & CI

- All modules and features are covered by tests in `tests/`.
- Tests include agent logic, dashboard flows, robustness, and integration.
- Use `pytest` for local testing and CI/CD pipelines.

---

## 8. Roadmap & Future Work

- See `ROADMAP.md` for planned features:
  - Advanced multi-modal fusion
  - Cloud scaling and distributed learning
  - Enhanced human-agent collaboration
  - Real-world integration and deployment
  - More robust and explainable AI features

---

## Appendix: Key Files (Quick Reference)

| Path                                 | Purpose/Description                 |
| ------------------------------------ | ----------------------------------- |
| src/core/neuro_fuzzy_fusion_agent.py | Main neuro-fuzzy agent logic        |
| src/core/fuzzy_system.py             | Fuzzy inference and rule management |
| src/core/robustness_wrappers.py      | Robustness and safety wrappers      |
| neuro_fuzzy_multiagent/dashboard/neuro_fuzzy_multiagent/main.py                    | Streamlit dashboard entry point     |
| dashboard/visualization.py           | Agent/group/SOM visualizations      |
| dashboard/simulation.py              | Simulation and experiment logic     |
| dashboard/chat.py                    | Human-agent chat and feedback       |
| examples/                            | Example/demo scripts                |
| tests/                               | Test suite                          |
| requirements.txt                     | Python dependencies                 |
| Dockerfile                           | Containerization                    |
| ROADMAP.md                           | Project roadmap and milestones      |
