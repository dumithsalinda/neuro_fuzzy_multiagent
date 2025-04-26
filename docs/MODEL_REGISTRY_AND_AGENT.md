# Neuro-Fuzzy Multiagent OS: Model Registry & Agent Integration Guide

## Overview
This guide explains how to register, validate, and manage neural network models in the NFMA-OS model registry, and how to build agents that dynamically load and hot-reload models.

---

## 1. Model Registration & Validation

### Model Directory Structure
Each model must be in its own directory, containing at least:
- `model.json` — Metadata file (see below)
- Model weights file (e.g., `model.onnx`, `model.pb`, etc.)

### Example `model.json`
```json
{
  "name": "example_model",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Short description of the model.",
  "supported_device": "device_type",
  "input_schema": "float32[1,2]",
  "output_schema": "float32[1]",
  "hash": "sha256:<actual_sha256_of_model_file>",
  "model_type": "cnn",
  "framework": "onnx",
  "model_dir": "/absolute/path/to/model_dir"
}
```
- The `hash` field must match the SHA-256 hash of the model file.
- Add a `signature` field if cryptographic signing is used.

### Registering a Model (CLI)
```sh
./scripts/nfmaos_model.py register /path/to/model_dir
```
- Validates schema, hash, and (optionally) signature.

### Listing, Inspecting, and Removing Models
```sh
./scripts/nfmaos_model.py list
./scripts/nfmaos_model.py inspect <model_name>
./scripts/nfmaos_model.py remove <model_name>
```

---

## 2. Model Registry & Loader Usage

- The registry stores model metadata as JSON files in a directory (default: `/opt/nfmaos/registry`).
- The loader validates metadata, checks hashes, and loads models for inference.

**Python usage:**
```python
from src.utils.model_registry import ModelRegistry
from src.utils.model_loader import ModelLoader

registry = ModelRegistry("/opt/nfmaos/registry")
for model_name in registry.list_models():
    meta = registry.get_model_metadata(model_name)
    model_dir = meta["model_dir"]
    loader = ModelLoader(model_dir)
    # loader.predict(input_data)
```

---

## 3. Agent Integration & Hot-Reloading

- Use `RegistryWatcher` to monitor model changes and reload models dynamically.
- Example agent:

```python
from src.utils.registry_watcher import RegistryWatcher

class HotReloadAgent:
    ... # see examples/agent_with_model_hot_reload.py
```

---

## 4. Security & Validation
- Model metadata is validated against a schema.
- Hashes are checked for integrity.
- Optional: Signature verification using a public key (set `NFMAOS_MODEL_PUBKEY`).
- Permission checks and audit logging are recommended for production.

---

## 5. Testing
- Automated tests are in `tests/`.
- Use a real ONNX model for ONNXRuntime-based tests.

---

## 6. Troubleshooting
- **Hash mismatch:** Ensure the hash in `model.json` matches the actual model file.
- **Signature error:** Set the correct public key and ensure the model is signed.
- **ONNXRuntime error:** Use a valid ONNX model file.
- **Dependency errors:** Install required packages: `onnxruntime`, `onnx`, `tensorflow` (if needed).

---

## 7. References
- See also: `src/utils/model_registry.py`, `src/utils/model_loader.py`, `src/utils/registry_watcher.py`, `examples/agent_with_model_hot_reload.py`
- CLI: `scripts/nfmaos_model.py`

---

## 8. Project Improvement and Roadmap Ideas

Here are advanced ways to further elevate the neuro-fuzzy multiagent OS project:

### Deepen the Intelligence Layer
- Adaptive resource management agents
- Self-learning device recognition
- Explainable neuro-fuzzy agent decisions

### Expand the Model and Agent Ecosystem
- AI driver/agent marketplace
- Multi-framework support (PyTorch, TensorFlow, ONNX, scikit-learn)
- Agent orchestration (collaborative/competitive multi-agent systems)

### Enhance Security and Robustness
- Cryptographic signing and permission checks for models/agents
- Sandboxing and isolation of agent execution
- Audit logging for traceability

### Improve Usability and Developer Experience
- Comprehensive documentation, tutorials, and API references
- User-friendly CLI and (optionally) GUI dashboard
- Hot-reload and live tuning of agents

### Testing, Simulation, and Benchmarks
- Automated unit, integration, and stress testing
- Simulation environment for agent/model testing
- Performance benchmarks for adaptability and resource use

### Community and Collaboration
- Public roadmap and contribution guidelines
- Hackathons and contests for new agent/model ideas
- Partnerships with universities, vendors, and open-source orgs

### Broaden Application Domains
- Edge/IoT, robotics, industrial automation
- Cloud/datacenter optimization
- Accessibility, personalization, and adaptive UX

### Visionary Features
- Self-healing and self-optimizing agents
- Natural language interface for agent/system configuration
- Federated/distributed learning for privacy-preserving adaptation

#### Summary Table

| Area                | Improvement Ideas                                         |
|---------------------|----------------------------------------------------------|
| Intelligence        | Adaptive, explainable, and collaborative agents          |
| Ecosystem           | AI driver marketplace, multi-framework, orchestration    |
| Security            | Signing, sandboxing, auditing                            |
| Usability           | Docs, CLI/GUI, live tuning                              |
| Testing             | Automated, simulation, benchmarking                      |
| Community           | Roadmap, hackathons, partnerships                        |
| Applications        | Edge, cloud, accessibility, personalization              |
| Visionary           | Self-healing, NL interface, federated learning           |

---

For further details, see the code docstrings and the main OS_DETAILED_PLAN.md.
