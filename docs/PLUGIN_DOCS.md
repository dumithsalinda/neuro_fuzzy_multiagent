# Plugin API Documentation

The plugin system enables easy extension of the platform with new agents, environments, sensors, actuators, and neural networks. All plugins are auto-discovered and available in the dashboard/config—no core code changes required.



## Agent Plugins

### MetaAgent

MetaAgent adapts its own learning algorithm or architecture based on performance metrics.
It can switch between different agent models (DQN, NeuroFuzzy, etc.) or tune hyperparameters on the fly.

**Config options:**

- `candidate_agents`: Any (default: required)
- `selection_strategy`: Any (default: None)
- `perf_metric`: Any (default: reward)
- `window`: Any (default: 10)

### NeuroFuzzyFusionAgent

Multi-modal agent combining neural and fuzzy logic for decision making.
Accepts multiple input modalities, fuses both neural and fuzzy outputs.

**Config options:**

- `input_dims`: Any (default: required)
- `hidden_dim`: Any (default: required)
- `output_dim`: Any (default: required)
- `fusion_type`: Any (default: concat)
- `fuzzy_config`: Any (default: None)
- `fusion_alpha`: Any (default: 0.5)
- `device`: Any (default: None)

### TabularQLearningAgent

Generic agent that interacts with an environment using a model and policy.

Supports online learning, dynamic group membership, plug-and-play fuzzy logic (UniversalFuzzyLayer),
and a standardized communication API (send_message, receive_message).

Args:
    model: The agent's underlying model (e.g., DQN, NeuroFuzzyHybrid).
    policy: Callable for action selection (optional).
    bus: Optional message bus for inter-agent communication.
    group: Optional group identifier.

**Config options:**

- `n_states`: Any (default: None)
- `n_actions`: Any (default: 2)
- `alpha`: Any (default: 0.1)
- `gamma`: Any (default: 0.99)
- `epsilon`: Any (default: 0.1)

### MetaRLAgent

Meta-Reinforcement Learning Agent.
This agent adapts its own learning strategy (e.g., learning rate, exploration policy, even algorithm) based on meta-feedback about its performance.

**Config options:**

- `base_agent_cls`: Any (default: required)
- `base_agent_kwargs`: Any (default: required)
- `meta_lr`: Any (default: 0.1)
- `meta_window`: Any (default: 10)

### MultiModalFusionAgent

Agent that accepts multiple modalities as input and uses a fusion network for policy/Q-value computation.
Easily extensible: add new fusion strategies in FusionNetwork and update here.
TODO: Add support for training (currently eval-only), more modalities, and advanced fusion methods (e.g., attention, gating).

**Config options:**

- `input_dims`: Any (default: required)
- `hidden_dim`: Any (default: required)
- `output_dim`: Any (default: required)
- `fusion_type`: Any (default: concat)
- `lr`: Any (default: 0.001)
- `gamma`: Any (default: 0.99)

### DQNAgent

Generic agent that interacts with an environment using a model and policy.

Supports online learning, dynamic group membership, plug-and-play fuzzy logic (UniversalFuzzyLayer),
and a standardized communication API (send_message, receive_message).

Args:
    model: The agent's underlying model (e.g., DQN, NeuroFuzzyHybrid).
    policy: Callable for action selection (optional).
    bus: Optional message bus for inter-agent communication.
    group: Optional group identifier.

**Config options:**

- `state_dim`: Any (default: required)
- `action_dim`: Any (default: required)
- `alpha`: Any (default: 0.001)
- `gamma`: Any (default: 0.99)
- `epsilon`: Any (default: 0.1)

### MultiModalDQNAgent

Generic agent that interacts with an environment using a model and policy.

Supports online learning, dynamic group membership, plug-and-play fuzzy logic (UniversalFuzzyLayer),
and a standardized communication API (send_message, receive_message).

Args:
    model: The agent's underlying model (e.g., DQN, NeuroFuzzyHybrid).
    policy: Callable for action selection (optional).
    bus: Optional message bus for inter-agent communication.
    group: Optional group identifier.

**Config options:**

- `input_dims`: Any (default: required)
- `action_dim`: Any (default: required)
- `alpha`: Any (default: 0.001)
- `gamma`: Any (default: 0.99)
- `epsilon`: Any (default: 0.1)

### ANFISHybrid

Minimal Adaptive Neuro-Fuzzy Inference System (ANFIS) hybrid model.
- Supports Gaussian membership functions (tunable params)
- Rule weights are learnable (simple gradient update placeholder)
- Forward: computes fuzzy rule firing strengths, weighted sum for output
- Dynamic: can add/prune rules based on usage and error

**Config options:**

- `input_dim`: Any (default: required)
- `n_rules`: Any (default: required)

### AgentFeatureSOM

No docstring.

**Config options:**

- `x`: Any (default: 5)
- `y`: Any (default: 5)
- `input_len`: Any (default: 3)
- `sigma`: Any (default: 1.0)
- `learning_rate`: Any (default: 0.5)

### DummyAgent

A minimal agent for testing plug-and-play registration.
Always selects action 0 and does not learn.


### NeuroFuzzyANFISAgent

Agent wrapper for the ANFISHybrid neuro-fuzzy model.
Supports act (forward), observe (online update), experience replay for continual learning,
and meta-learning hooks (e.g., adaptive learning rate, learning-to-learn).

meta_update_fn: Optional callback called after each update.
  Signature: fn(agent, step: int) -> None
  agent: the NeuroFuzzyANFISAgent instance
  step: current update step

**Config options:**

- `input_dim`: Any (default: required)
- `n_rules`: Any (default: required)
- `lr`: Any (default: 0.01)
- `buffer_size`: Any (default: 100)
- `replay_enabled`: Any (default: True)
- `replay_batch`: Any (default: 8)
- `meta_update_fn`: Any (default: None)


## Environment Plugins

### GymEnvWrapper

Wraps an OpenAI Gym environment to make it compatible with the Neuro-Fuzzy Multi-Agent platform.

**Config options:**

- `env_name`: Any (default: required)
- `kwargs`: Any (default: required)


## Sensor Plugins

### ExampleSensor

Example sensor plugin that returns a fixed observation.

**Config options:**

- `observation`: Any (default: 0)


## Actuator Plugins

### ExampleActuator

Example actuator plugin that sets and stores the last action.

