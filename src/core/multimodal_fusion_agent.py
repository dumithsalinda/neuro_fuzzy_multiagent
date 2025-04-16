import torch
import torch.nn as nn
from .fusion import FusionNetwork

class MultiModalFusionAgent:
    """
    Agent that accepts multiple modalities as input and uses a fusion network for policy/Q-value computation.
    Easily extensible: add new fusion strategies in FusionNetwork and update here.
    TODO: Add support for training (currently eval-only), more modalities, and advanced fusion methods (e.g., attention, gating).
    """
    def __init__(self, input_dims, hidden_dim, output_dim, fusion_type='concat'):
        """
        Args:
            input_dims (list[int]): List of input dimensions for each modality (e.g., [img_dim, txt_dim]).
            hidden_dim (int): Hidden layer size for fusion network.
            output_dim (int): Number of possible actions.
            fusion_type (str): Fusion strategy ('concat', 'attention', 'gating').
        """
        self.model = FusionNetwork(input_dims, hidden_dim, output_dim, fusion_type=fusion_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # For demonstration, not training yet
        # TODO: Add support for online/continual learning and training

    def act(self, obs_list):
        """
        Select action given a list of modality features.
        Args:
            obs_list (list[np.ndarray or torch.Tensor]): Features for each modality.
        Returns:
            int: Selected action (argmax Q-value)
        """
        features = [torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0) for x in obs_list]
        with torch.no_grad():
            qvals = self.model(features)
        action = torch.argmax(qvals, dim=-1).item()
        return action

    def __repr__(self):
        return f"MultiModalFusionAgent(input_dims={self.model.input_dims})"
        # TODO: Add more informative metadata for model management
