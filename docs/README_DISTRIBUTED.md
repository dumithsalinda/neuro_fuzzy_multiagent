# Distributed Execution with Ray

> **Note:** For distributed agent/plugin development and troubleshooting, see the [Developer Guide](DEVELOPER.md).

## Local Ray Cluster

1. Launch the Ray cluster:
   ```bash
   bash neuro_fuzzy_multiagent/scripts/launch_ray_cluster.sh
   ```

2. Submit a distributed experiment job:
   ```bash
   ray submit neuro_fuzzy_multiagent/config/ray-cluster.yaml neuro_fuzzy_multiagent/run_distributed_fusion_experiment.py
   ```

## Cloud/Remote Cluster (Template)
- Edit `neuro_fuzzy_multiagent/config/ray-cluster.yaml` to use your cloud provider (see Ray docs for AWS/GCP/Azure setup).
- Make sure all nodes have access to your code and dependencies.

## Requirements
- Python 3.8+
- Ray (`pip install ray`)
- All project dependencies (`pip install -r requirements.txt`)

## Troubleshooting
- **Ray not found:** Ensure Ray is installed in your Python environment.
- **Cluster connection issues:** Check firewall, YAML config, and cloud provider setup.
- **Plugin/agent not running as expected:** See [Developer Guide](DEVELOPER.md) for debugging distributed plugins.
- **Missing dependencies:** Double-check `requirements.txt` and sync all nodes.

## Notes
- The NeuroFuzzyFusionAgent is fully compatible with Ray distributed execution.
- See `neuro_fuzzy_multiagent/run_distributed_fusion_experiment.py` for an example.
