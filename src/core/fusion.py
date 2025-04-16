import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    """
    Flexible fusion network supporting concatenation, attention, and gating.
    Accepts a list of feature vectors from different modalities and outputs Q-values or actions.
    Easily extensible for new fusion methods.
    """
    def __init__(self, input_dims, hidden_dim, output_dim, fusion_type='concat'):
        """
        Args:
            input_dims (list[int]): List of input dims for each modality.
            hidden_dim (int): Hidden layer size.
            output_dim (int): Number of actions.
            fusion_type (str): 'concat', 'attention', or 'gating'
        """
        super().__init__()
        self.input_dims = input_dims
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.fusion_dim = sum(input_dims)
            self.fc1 = nn.Linear(self.fusion_dim, hidden_dim)
        elif fusion_type == 'attention':
            self.fusion_dim = max(input_dims)
            self.attn = nn.Parameter(torch.ones(len(input_dims)), requires_grad=True)
            self.fc1 = nn.Linear(self.fusion_dim, hidden_dim)
        elif fusion_type == 'gating':
            self.fusion_dim = sum(input_dims)
            self.gate = nn.Sequential(
                nn.Linear(self.fusion_dim, len(input_dims)),
                nn.Sigmoid()
            )
            self.fc1 = nn.Linear(self.fusion_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): List of tensors, one per modality (shape: [batch, dim])
        Returns:
            Tensor: Q-values or logits, shape [batch, output_dim]
        """
        if self.fusion_type == 'concat':
            x = torch.cat(features, dim=-1)
        elif self.fusion_type == 'attention':
            import torch.nn.functional as F
            weights = F.softmax(self.attn, dim=0)  # shape: [n_modalities]
            # Pad features to same dim (max), then weighted sum
            x = sum(w * F.pad(f, (0, self.fusion_dim - f.shape[-1])) for w, f in zip(weights, features))
        elif self.fusion_type == 'gating':
            x_cat = torch.cat(features, dim=-1)
            gates = self.gate(x_cat)  # shape: [batch, n_modalities]
            gated = [gates[:, i].unsqueeze(-1) * f for i, f in enumerate(features)]
            x = torch.cat(gated, dim=-1)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
        # TODO: Add more advanced fusion methods as needed
