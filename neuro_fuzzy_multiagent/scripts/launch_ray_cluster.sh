#!/bin/bash
# Launch Ray cluster locally using the provided cluster config
echo "Launching Ray cluster..."
ray up ray-cluster.yaml -y
echo "Ray cluster launched. To submit a job, run:"
echo "ray submit ray-cluster.yaml run_distributed_fusion_experiment.py"
