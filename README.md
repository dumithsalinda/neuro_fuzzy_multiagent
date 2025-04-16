# Neuro-Fuzzy Multi-Agent System

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
2. Run the API server:
   ```sh
   uvicorn agent_api:app --reload
   ```
3. Send requests as above to `/observe/multimodal`.

---

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

## Running Tests
To run all tests:
```sh
PYTHONPATH=src pytest --disable-warnings -q
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
