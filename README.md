# Environment-Independent Dynamic Self-Organizing Neuro-Fuzzy Multi-Agent System

This project combines neural networks, fuzzy logic, multi-agent collaboration, and self-organization to create an adaptive, robust AI system that can operate across diverse environments without explicit reprogramming.

## Features
- Neural network and fuzzy logic core
- Multi-agent self-organization and collaboration
- Global rule set enforcement
- Internet and video-based learning
- **Environment abstraction and transfer learning** (Phase 2)

## Directory Structure
```
core/
    neural_network.py
    fuzzy_logic.py
    agent.py
    evolution.py
    rules.py
multiagent/
    collaboration.py
    environment.py
environment/
    abstraction.py
    transfer_learning.py
internet_learning/
    web_search.py
    video_learning.py
    knowledge_base.py
main.py
requirements.txt
```

## Usage
- Run `python main.py` to start the system.
- Customize agents, rules, and learning modules as needed.

### Environment Abstraction Example
```python
from src.environment.abstraction import SimpleEnvironment

env = SimpleEnvironment(dim=4)
state = env.reset()  # Random initial state
features = env.extract_features()  # Identity mapping by default
```

### Transfer Learning Example
```python
from src.environment.abstraction import SimpleEnvironment
from src.environment.transfer_learning import FeatureExtractor, transfer_learning
from src.core.neural_network import FeedforwardNeuralNetwork

# Source and target environments
env1 = SimpleEnvironment(dim=4)
env2 = SimpleEnvironment(dim=4)

# Feature extractor and model
feat = FeatureExtractor(input_dim=4, output_dim=2)
model = FeedforwardNeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)

# Run transfer learning (pretrain on env1, finetune on env2)
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
