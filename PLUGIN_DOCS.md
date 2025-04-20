# Plugin API Reference


## Environments

### AdversarialGridworldEnv

Multi-agent gridworld with pursuers and evaders.
Pursuers try to catch evaders; evaders try to reach target or avoid capture.

**Config options:**

- `grid_size`: Any (default: 5)
- `n_pursuers`: Any (default: 1)
- `n_evaders`: Any (default: 1)
- `n_obstacles`: Any (default: 0)

### IoTSensorFusionEnv

Hybrid environment: gridworld with simulated IoT sensors (can be extended to real sensors).
State = [agent positions, temperature, humidity, light].
Optional: replace sensor simulation with real-world data.

**Config options:**

- `grid_size`: Any (default: 5)
- `n_agents`: Any (default: 2)
- `use_real_sensors`: Any (default: False)

### MultiAgentGridworldEnv

Example refactored environment: Multi-Agent Gridworld
Now inherits from BaseEnvironment and implements required methods.

**Config options:**

- `grid_size`: Any (default: 5)
- `n_agents`: Any (default: 3)
- `n_obstacles`: Any (default: 2)

### MultiAgentResourceEnv

Multi-agent resource collection environment (grid).
Agents collect resources for reward. Supports cooperative, competitive, or mixed.

**Config options:**

- `grid_size`: Any (default: 5)
- `n_agents`: Any (default: 2)
- `n_resources`: Any (default: 3)
- `mode`: Any (default: competitive)

### NoisyEnvironment

Environment with a random state vector plus Gaussian noise.
Useful for simulating domain shift and testing transfer learning robustness.

**Config options:**

- `dim`: Any (default: 3)
- `noise_std`: Any (default: 0.5)

### RealWorldAPIEnv

Example environment for real-world API/sensor/robot integration.
Implements optional hooks for connection, action, and sensor data.
Replace stub logic with actual API/robot code as needed.

**Config options:**

- `config`: Any (default: None)

### SimpleContinuousEnv

Simple continuous environment for DQN (e.g., 2D point to goal).


### SimpleDiscreteEnv

Simple discrete environment for tabular Q-learning (e.g., N-state chain).

**Config options:**

- `n_states`: Any (default: 5)
- `n_actions`: Any (default: 2)

### SimpleEnvironment

A simple toy environment with a random state vector.
Useful for basic testing or as a source domain for transfer learning.

Parameters
----------
dim : int
    Dimensionality of the state vector.

**Config options:**

- `dim`: Any (default: 3)


## Agents

### Agent

Generic agent that interacts with an environment using a model and policy.

Supports online learning, dynamic group membership, plug-and-play fuzzy logic (UniversalFuzzyLayer),
and a standardized communication API (send_message, receive_message).

Args:
    model: The agent's underlying model (e.g., DQN, NeuroFuzzyHybrid).
    policy: Callable for action selection (optional).
    bus: Optional message bus for inter-agent communication.
    group: Optional group identifier.

**Config options:**

- `model`: typing.Any (default: required)
- `policy`: typing.Optional[typing.Callable] (default: None)
- `bus`: typing.Optional[typing.Any] (default: None)
- `group`: typing.Optional[str] (default: None)

### NeuroFuzzyAgent

Agent that uses a NeuroFuzzyHybrid model to select actions.
Supports modular self-organization of fuzzy rules, membership functions, and neural network weights.
Supports runtime mode switching between neural, fuzzy, and hybrid inference.

**Config options:**

- `nn_config`: Any (default: required)
- `fis_config`: Any (default: required)
- `policy`: Any (default: None)
- `meta_controller`: Any (default: None)
- `universal_fuzzy_layer`: Any (default: None)

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

### DummyAgent

A minimal agent for testing plug-and-play registration.
Always selects action 0 and does not learn.


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


## Neural Networks

### FeedforwardNeuralNetwork

Minimal feedforward neural network for hybrid neuro-fuzzy systems.
input_dim should match the feature vector dimension for the agent's input type (e.g., 768 for BERT, 512 for ResNet18).
Extendable for hybrid learning (backpropagation + evolution).

**Config options:**

- `input_dim`: Any (default: required)
- `hidden_dim`: Any (default: required)
- `output_dim`: Any (default: required)
- `activation`: Any (default: tanh)

### ConvolutionalNeuralNetwork

Example CNN for plug-and-play extension. (This is a stub; actual implementation needed for real use.)

**Config options:**

- `input_shape`: Any (default: required)
- `num_filters`: Any (default: required)
- `kernel_size`: Any (default: required)
- `output_dim`: Any (default: required)
- `activation`: Any (default: <ufunc 'tanh'>)


## Sensors

### DummySensor

A minimal sensor plugin for plug-and-play testing. Returns a constant value.

**Config options:**

- `args`: Any (default: required)
- `kwargs`: Any (default: required)


## Actuators

### DummyActuator

A minimal actuator plugin for plug-and-play testing. Prints the received command.

**Config options:**

- `args`: Any (default: required)
- `kwargs`: Any (default: required)
