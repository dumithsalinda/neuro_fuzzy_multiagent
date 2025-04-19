# Examples and Demos

This folder contains example scripts and demonstration code for the Neuro-Fuzzy Multi-Agent System project. Move or add your demo and example scripts here for better project organization.

---

## Agent Config YAML Examples

The following YAML files demonstrate how to instantiate each supported agent type dynamically using the agent factory. Each config can be loaded with `create_agent_from_file`.

### 1. [agent_config_example.yaml](./agent_config_example.yaml)
- **Type:** NeuroFuzzyAgent
- **Required fields:**
  - `agent_type: NeuroFuzzyAgent`
  - `nn_config`: `{input_dim, hidden_dim, output_dim}`
  - `fis_config` (optional)
  - `meta_controller` (optional)
  - `universal_fuzzy_layer` (optional)

### 2. [agent_config_fusion.yaml](./agent_config_fusion.yaml)
- **Type:** NeuroFuzzyFusionAgent
- **Required fields:**
  - `input_dims` (list of int)
  - `hidden_dim` (int)
  - `output_dim` (int)
- **Optional:** `fusion_type`, `fuzzy_config`, `fusion_alpha`, `device`

### 3. [agent_config_dqn.yaml](./agent_config_dqn.yaml)
- **Type:** DQNAgent
- **Required fields:**
  - `state_dim` (int)
  - `action_dim` (int)
- **Optional:** `alpha`, `gamma`, `epsilon`

### 4. [agent_config_multimodal_dqn.yaml](./agent_config_multimodal_dqn.yaml)
- **Type:** MultiModalDQNAgent
- **Required fields:**
  - `input_dims` (list of int)
  - `action_dim` (int)
- **Optional:** `alpha`, `gamma`, `epsilon`

### 5. [agent_config_anfis.yaml](./agent_config_anfis.yaml)
- **Type:** NeuroFuzzyANFISAgent
- **Required fields:**
  - `input_dim` (int)
  - `n_rules` (int)
- **Optional:** `lr`, `buffer_size`, `replay_enabled`, `replay_batch`, `meta_update_fn`

### 6. [agent_config_multimodal_fusion.yaml](./agent_config_multimodal_fusion.yaml)
- **Type:** MultiModalFusionAgent
- **Required fields:**
  - `input_dims` (list of int)
  - `hidden_dim` (int)
  - `output_dim` (int)
- **Optional:** `fusion_type`, `lr`, `gamma`

---

**Usage:**
```python
from src.core.agent_factory import create_agent_from_file
agent = create_agent_from_file('examples/agent_config_fusion.yaml')
```

For more details on config options, see the source code and agent class docstrings.
