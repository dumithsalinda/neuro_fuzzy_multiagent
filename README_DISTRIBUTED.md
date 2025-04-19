# Distributed Execution with Ray

## Local Ray Cluster

1. Launch the Ray cluster:
   ```bash
   bash launch_ray_cluster.sh
   ```

2. Submit a distributed experiment job:
   ```bash
   ray submit ray-cluster.yaml run_distributed_fusion_experiment.py
   ```

## Cloud/Remote Cluster (Template)
- Edit `ray-cluster.yaml` to use your cloud provider (see Ray docs for AWS/GCP/Azure setup).
- Make sure all nodes have access to your code and dependencies.

## Requirements
- Python 3.8+
- Ray (`pip install ray`)
- All project dependencies (`pip install -r requirements.txt`)

## Notes
- The NeuroFuzzyFusionAgent is fully compatible with Ray distributed execution.
- See `run_distributed_fusion_experiment.py` for an example.
