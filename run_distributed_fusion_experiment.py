"""
Example script: Distributed inference with NeuroFuzzyFusionAgent using Ray
"""

import numpy as np

from neuro_fuzzy_multiagent.core.distributed_agent_executor import (
    run_agents_distributed,
)
from neuro_fuzzy_multiagent.core.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent

# Example configs for 3 agents (can scale up)
input_dims = [4, 3]  # e.g., two modalities: 4-dim and 3-dim
hidden_dim = 16
output_dim = 5  # e.g., 5 possible actions
fusion_type = "concat"
fusion_alpha = 0.6
fuzzy_config = None  # Use default fuzzy system

# Instantiate several NeuroFuzzyFusionAgents
num_agents = 3
agents = [
    NeuroFuzzyFusionAgent(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        fusion_type=fusion_type,
        fuzzy_config=fuzzy_config,
        fusion_alpha=fusion_alpha,
    )
    for _ in range(num_agents)
]

# Generate example multi-modal observations for each agent
# Each agent gets a list of two modalities: [modality1, modality2]
observations = [[np.random.rand(4), np.random.rand(3)] for _ in range(num_agents)]

# Run distributed inference
print("Running distributed NeuroFuzzyFusionAgent inference with Ray...")
actions = run_agents_distributed(agents, observations)

for idx, (obs, action) in enumerate(zip(observations, actions)):
    print(f"Agent {idx}: obs={obs}, action={action}")

print("Done.")
