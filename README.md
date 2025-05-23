# Neuro-Fuzzy Multiagent Framework

A modular, plug-and-play platform for building, experimenting with, and deploying intelligent neuro-fuzzy agents in Python.

**Note:** This package is a standalone neuro-fuzzy multiagent framework, not an operating system. It provides a flexible and extensible platform for developing and testing intelligent agents, but it does not include operating system features.

---

## Table of Contents

- [Quickstart & Requirements](#quickstart--requirements)
- [Key Features](#key-features)
- [Overview](#overview)
- [Advanced Integrations](#advanced-integrations)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Community & Conduct](#community--conduct)
- [Security Policy](#security-policy)

## Quickstart & Requirements

> **Note:**
>
> - For ROS features, install ROS and `rospy` separately (not via pip).
> - For MQTT and API server features, `paho-mqtt`, `fastapi`, and `uvicorn` are required (now included in requirements.txt).

1. **Install requirements:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the dashboard:**
   ```sh
   streamlit run dashboard.py
   ```
3. **Explore:**
   - Select agents, environments, sensors, and actuators in the sidebar.
   - Run simulations, visualize results, and interact live.
   - Add your own plugin (see [DEVELOPER.md](docs/DEVELOPER.md)).

---

## Key Features

- **Intelligent neuro-fuzzy agents** for adaptive resource management and system optimization
- **Self-learning device recognition** via installable AI drivers (trained models)
- **Proactive and context-aware system optimization** for dynamic user experiences
- **Plug-and-play installation of new AI capabilities** (extend intelligence to new hardware or domains)
- **Prototyping of intelligent agents** (e.g., adaptive resource manager, smart USB handler)
- **AI driver/model installation tool** for seamless integration of new learning models
- Modular neuro-fuzzy agents, easy plug-and-play extension
- Privacy-aware knowledge sharing and multi-agent collaboration
- Online learning and knowledge integration
- Multi-modal fusion agent support (text, image, etc.) _(in development: advanced fusion methods, e.g., attention, gating)_
- Model management, versioning, and batch experimentation
- Robust dashboard for simulation, analytics, and HITL (human-in-the-loop)
- **Note:** Some advanced features are in development. See the FAQ for details.

---

## Overview

A robust, environment-independent, dynamic self-organizing neuro-fuzzy multi-agent system with:

- **Unbreakable Laws:** Hard constraints on agent and group behavior.
- **Modular Neuro-Fuzzy Agents:** Hybrid models, self-organization, and extensibility.
- **Multi-Agent Collaboration:** Communication, consensus, and privacy-aware knowledge sharing.
- **Online Learning:** Agents learn from web resources and integrate knowledge.
- **Advanced Group Decision-Making:** Mean, weighted mean, majority vote, custom aggregation.
- **Thorough Testing & Extensibility:** Easily add new laws, agent types, or aggregation methods.

---

## Advanced Integrations

### IoT Sensor Integration

- Simulated or real IoT sensors for smart environment demos and agent interaction.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.plugins.iot_sensor import IoTSensor
  def my_callback(name, value):
      print(f"Sensor {name}: {value}")
  sensor = IoTSensor('temp_sensor', interval=1.0, callback=my_callback)
  sensor.start()
  # ...
  sensor.stop()
  ```

### ROS/IoT Integration _(in development)_

- Minimal ROS bridge for agent-environment communication via ROS topics.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.plugins.ros_bridge import ROSBridge
  bridge = ROSBridge(node_name='nfma_ros_bridge')
  pub = bridge.create_publisher('/nfma/test', queue_size=1)
  bridge.publish('/nfma/test', 'hello world')
  ```

### Meta-Reinforcement Learning _(in development)_

- Agents that adapt their own learning strategies over time using meta-feedback.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.agents.meta_rl_agent import MetaRLAgent
  meta_agent = MetaRLAgent(base_agent_cls=TabularQLearningAgent, base_agent_kwargs={'lr': 0.1}, meta_lr=0.05, meta_window=5)
  # Use meta_agent in place of any agent; it will adjust its learning rate based on reward trends.
  ```

### Distributed Multi-Agent Systems

- Support for distributed agent deployment and communication.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.distributed import DistributedAgent
  agent = DistributedAgent('agent_name', host='localhost', port=8080)
  agent.start()
  # ...
  agent.stop()
  ```

---

## Documentation

- [Project Documentation](docs/PROJECT_DOCUMENTATION.md) — Full system overview and architecture
- [Developer Guide](docs/DEVELOPER.md) — How to extend the OS, create plugins, and contribute code
- [Contribution Guide](docs/CONTRIBUTING.md) — How to report issues, suggest features, and contribute
- [Plugin API Reference](docs/PLUGIN_DOCS.md) — Plugin types and configuration options
- [Model Registry & Agent Integration](docs/MODEL_REGISTRY_AND_AGENT.md)
- [Distributed Execution Guide](docs/README_DISTRIBUTED.md)
- [API Docs & Generation](docs/README_API_DOCS.md)

For questions or suggestions, open an issue or discussion on GitHub. Contributions are welcome!

---

## License

MIT

---

## Running Tests

To run all tests (requires pytest):

```sh
pytest tests/
```

All dashboard and core functions are covered by the test suite. If you encounter any errors, see the test output for details and troubleshooting. 2. Run the dashboard:

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

The platform supports meta-learning and automated hyperparameter optimization for agents:

### MetaAgent

- Dynamically selects the best-performing agent type (e.g., DQN, Q-Learning) based on recent performance.
- Usage:
  ```python
  from neuro_fuzzy_multiagent.core.agents.meta_agent import MetaAgent
  meta = MetaAgent(candidate_agents=[(DQNAgent, {...}), (TabularQLearningAgent, {...})])
  ```
- The MetaAgent tries each candidate for a set number of steps (exploration), then switches to the best.

### HyperparameterOptimizer

- Simple optimizer for agent hyperparameters (random search; extensible to Bayesian/evolutionary).
- Usage:
  ```python
  from neuro_fuzzy_multiagent.core.agents.hpo import HyperparameterOptimizer
  hpo = HyperparameterOptimizer(param_space, eval_fn)
  best_params, best_score = hpo.optimize(n_trials=100)
  ```
- Use to tune agent configs for best performance.

See `src/core/agents/meta_agent.py` and `src/core/agents/hpo.py` for details and extension points.

---

## Interoperability

### OpenAI Gym/Env Compatibility

- Use standard RL environments with the platform via `GymEnvWrapper`.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.environments.gym_env_wrapper import GymEnvWrapper
  env = GymEnvWrapper('CartPole-v1')
  obs = env.reset()
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  ```
- Enables benchmarking and comparison with widely-used RL tasks.

### ROS/IoT Integration

- Minimal ROS bridge for agent-environment communication via ROS topics.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.plugins.ros_bridge import ROSBridge
  bridge = ROSBridge(node_name='nfma_ros_bridge')
  pub = bridge.create_publisher('/nfma/test', queue_size=1)
  bridge.publish('/nfma/test', 'hello world')
  ```
- Allows integration with robotics middleware and real-world sensors/actuators.
- Requires `rospy` (ROS Python client library).

---

## Meta-Reinforcement Learning

- Agents that adapt their own learning strategies over time using meta-feedback.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.agents.meta_rl_agent import MetaRLAgent
  meta_agent = MetaRLAgent(base_agent_cls=TabularQLearningAgent, base_agent_kwargs={'lr': 0.1}, meta_lr=0.05, meta_window=5)
  # Use meta_agent in place of any agent; it will adjust its learning rate based on reward trends.
  ```
- Supports meta-learning of hyperparameters such as learning rate, exploration policy, etc.

## IoT Sensor Integration

- Simulated or real IoT sensors for smart environment demos and agent interaction.
- Example:
  ```python
  from neuro_fuzzy_multiagent.core.plugins.iot_sensor import IoTSensor
  def my_callback(name, value):
      print(f"Sensor {name}: {value}")
  sensor = IoTSensor('temp_sensor', interval=1.0, callback=my_callback)
  sensor.start()
  # ...
  sensor.stop()
  ```
- Can be extended to connect to real hardware (MQTT, HTTP, serial, etc).

## General IoT Device Integration

- Use the `IoTDevice` class for most smart devices (simulated, MQTT, HTTP/REST, etc).
- Supports reading data and sending commands.

### Simulated Device Example

```python
from neuro_fuzzy_multiagent.core.plugins.iot_common import IoTDevice
def cb(name, value):
    print(f"Simulated {name}: {value}")
device = IoTDevice('sim_sensor', mode='sim', interval=1.0, callback=cb)
device.start()
# ...
device.stop()
```

### MQTT Device Example

```python
from neuro_fuzzy_multiagent.core.plugins.iot_common import IoTDevice
def cb(name, value):
    print(f"MQTT {name}: {value}")
mqtt_conf = {'broker': 'localhost', 'topic': 'home/livingroom/temperature'}
device = IoTDevice('livingroom_temp', mode='mqtt', mqtt_config=mqtt_conf, callback=cb)
device.start()
# To send a command:
device.send_command('ON')
```

### HTTP Device Example

```python
from neuro_fuzzy_multiagent.core.plugins.iot_common import IoTDevice
def cb(name, value):
    print(f"HTTP {name}: {value}")
http_conf = {'url': 'http://device_ip/api/temperature', 'method': 'GET'}
device = IoTDevice('temp_sensor', mode='http', http_config=http_conf, interval=10, callback=cb)
device.start()
# To send a command:
device.send_command(22)  # e.g., set thermostat to 22°C
```

- Extend or subclass for advanced protocols (Zigbee, BLE, etc).

---

# Deployment Guide: Self-Driving Car & Smart Home Integration

This guide explains how to deploy the Neuro-Fuzzy Multi-Agent System in both a self-driving car and a smart home, and how to enable communication between them.

---

## Plugin Auto-Discovery & Versioning

- Plugins in the `src/core/plugins/` directory are auto-discovered, versioned, and can be hot-reloaded at runtime.
- Each plugin should define `__plugin_name__` and `__version__` attributes.

**Example:**

```python
from neuro_fuzzy_multiagent.core.plugins.auto_discovery import PluginRegistry
registry = PluginRegistry(plugin_dir='src/core/plugins', base_package='src.core.plugins')
registry.discover_plugins()
print(registry.list_plugins())  # [(name, version), ...]
plugin = registry.get_plugin('MyPlugin')
registry.reload_plugin('MyPlugin')  # Hot-reload
```

## Plugin Hot-Reloading

- Reload plugin code at runtime without restarting the dashboard or main process.
- Use `reload_plugin('PluginName')` or `reload_all()` on the registry.
- Useful for rapid development and production updates.

## Distributed Experiment Orchestration

- Run large-scale experiments using Ray for distributed agent/environment management.
- Launch multiple agents/environments as Ray actors; collect results in parallel.

**Example:**

```python
from neuro_fuzzy_multiagent.core.management.distributed_orchestrator import DistributedExperimentOrchestrator
from neuro_fuzzy_multiagent.core.agents.dummy_agent import DummyAgent
from neuro_fuzzy_multiagent.core.environments.gym_env_wrapper import GymEnvWrapper
orchestrator = DistributedExperimentOrchestrator(DummyAgent, GymEnvWrapper, agent_kwargs={}, env_kwargs={'env_name': 'CartPole-v1'}, num_agents=8)
orchestrator.launch()
results = orchestrator.run_episode(steps=100)
print(results)
orchestrator.shutdown()
```

- Scales to clusters for research or production workloads.

## 1. Self-Driving Car Deployment

- **Platform:** Onboard computer (e.g., NVIDIA Jetson, Raspberry Pi, or automotive PC) running Linux.
- **Integration:**
  - Use the `ROSBridge` plugin to connect agents to the car’s ROS (Robot Operating System) stack.
  - Agents can subscribe to topics like `/camera/image_raw`, `/lidar/points`, `/vehicle/odometry`, and publish to `/vehicle/cmd_vel`, `/vehicle/steering`.
  - Use `IoTSensor` for non-ROS sensors (CAN bus, GPS, IMU, etc).
- **Deployment:**
  - Package agents and environments as a ROS node or standalone Python process.
  - Launch via ROS launch files or systemd.

**Example:**

```python
from neuro_fuzzy_multiagent.core.plugins.ros_bridge import ROSBridge
bridge = ROSBridge(node_name='car_agent')
def steering_callback(topic, msg):
    # Process incoming sensor data
    ...
bridge.create_subscriber('/vehicle/odometry', steering_callback)
bridge.create_publisher('/vehicle/cmd_vel')
```

## 2. Smart Home Deployment

- **Platform:** Home server, Raspberry Pi, or cloud VM.
- **Integration:**
  - Use `IoTSensor` to connect to smart devices (thermostats, lights, security sensors) via MQTT, HTTP, or GPIO.
  - Use `ROSBridge` if the smart home uses ROS.
- **Deployment:**
  - Run agents as background services, Docker containers, or integrate with home automation (e.g., Home Assistant).

**Example:**

```python
from neuro_fuzzy_multiagent.core.plugins.iot_sensor import IoTSensor
def temp_callback(name, value):
    # Adjust thermostat or notify agent
    ...
sensor = IoTSensor('living_room_temp', interval=5, callback=temp_callback)
sensor.start()
```

## 3. Communication Between Car and Smart Home

- **Option 1: ROS Network**
  - Both systems use ROS and share a ROS master/network (VPN or LAN).
  - Agents publish/subscribe to shared topics (e.g., `/home/alert`, `/car/status`).
- **Option 2: MQTT Broker**
  - Both systems connect to a shared MQTT broker (local or cloud).
  - Publish/subscribe to topics like `car/status`, `home/alert`, `energy/usage`.
  - Use `IoTSensor` or a custom MQTT plugin.
- **Option 3: REST API/Webhooks**
  - Each system exposes HTTP endpoints for event-driven communication.
  - Agents send POST requests to each other for actions or alerts.

**Example: ROS/MQTT Cross-Communication**

```python
# Car agent publishes status to /car/status (ROS or MQTT)
# Home agent subscribes and adjusts home settings
# Home agent publishes alerts to /home/alert, car agent receives and reacts
```

---

## Community & Conduct

We welcome contributions from everyone! Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating. By contributing, you agree to abide by it and help foster a welcoming, safe, and productive environment for all.

## Security Policy

Please see our [Security Policy](SECURITY.md) for how to report vulnerabilities and best practices for using this project securely.

## 4. Security & Reliability

- Use secure communication: VPN, TLS for MQTT, authentication for APIs.
- Monitor system health and logs.
- Consider fail-safe and fallback strategies (emergency stop, manual override).

## 5. Will They Communicate?

**Yes**—with ROS, MQTT, or HTTP, agents in the car and smart home can communicate, share data, and coordinate actions in real time or asynchronously.

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

For all extension and plugin instructions—including adding new laws, agent types, group aggregation methods, sensors, actuators, and more—see the unified [Developer Guide](docs/DEVELOPER.md).

- **Add a new law:** Define a function and decorate with `@register_law` (see Developer Guide for details).
- **Add a new agent type:** Subclass `Agent` and override required methods (see Developer Guide for best practices).
- **Add a group aggregation method:** Use `MultiAgentSystem.group_decision(..., method="custom", custom_fn=...)`.

## Project Goals

- Build an intelligent, environment-independent, and robust multi-agent OS.
- Enable adaptive, explainable, and modular neuro-fuzzy agents for resource management and optimization.
- Strict compliance with unbreakable laws and safe AI principles.
- Support plug-and-play extensibility and well-documented contributions.

---

For more details, see module/class/method docstrings in the source code, and refer to the [Developer Guide](docs/DEVELOPER.md) for in-depth extension and integration instructions.

### Integration with Neuro-Fuzzy Hybrid Model
You can use the `NeuroFuzzyHybrid` model in place of the neural network in transfer learning:
```python
from neuro_fuzzy_multiagent.core.neuro_fuzzy import NeuroFuzzyHybrid
# ... setup fuzzy system and configs ...
model = NeuroFuzzyHybrid(nn_config, fis_config)
trained_model = transfer_learning(env1, env2, model, feat, steps=10)
```

### Testing

- Run all tests with `pytest` in the project root:
  ```sh
  pytest tests/
  ```
- See `tests/test_environment.py` for environment and transfer learning tests.
- See `tests/test_neuro_fuzzy.py` for hybrid model tests.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

