import torch
import torch.nn as nn
from .fusion import FusionNetwork

class MultiModalFusionAgent:
    """
    Agent that accepts multiple modalities as input and uses a fusion network for policy/Q-value computation.
    """
    def __init__(self, input_dims, hidden_dim, output_dim):
        self.model = FusionNetwork(input_dims, hidden_dim, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # For demonstration, not training yet

    def act(self, obs_list):
        # obs_list: list of numpy arrays or tensors, one per modality
        features = [torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0) for x in obs_list]
        with torch.no_grad():
            qvals = self.model(features)
        action = torch.argmax(qvals, dim=-1).item()
        return action

    def __repr__(self):
        return f"MultiModalFusionAgent(input_dims={self.model.input_dims})"
