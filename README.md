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

## Core Modules
- `src/core/agent.py`: Agent logic, knowledge sharing, law compliance, extensibility.
- `src/core/multiagent.py`: Multi-agent system, group decision, privacy-aware knowledge sharing.
- `src/core/laws.py`: Law registration, enforcement, and compliance.
- `src/core/online_learning.py`: Online learning mixin for web knowledge.

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
