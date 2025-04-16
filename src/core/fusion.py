import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    """
    Simple fusion network for concatenated multi-modal features.
    Accepts a list of feature vectors from different modalities and outputs Q-values or actions.
    """
    def __init__(self, input_dims, hidden_dim, output_dim):
        super().__init__()
        self.input_dims = input_dims
        self.total_input_dim = sum(input_dims)
        self.fc1 = nn.Linear(self.total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, features):
        # features: list of tensors, one per modality
        x = torch.cat(features, dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
