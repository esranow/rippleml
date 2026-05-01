import torch
import torch.nn as nn
from typing import List, Dict

class MultiFieldMLP(nn.Module):
    """
    Shared trunk with multiple heads for multi-field PINNs.
    
    fields: List of field names (e.g. ["u", "v", "p"])
    hidden: Number of neurons per hidden layer in the trunk.
    layers: Number of hidden layers in the trunk.
    """
    def __init__(self, in_dim: int, fields: List[str], hidden: int = 50, layers: int = 4):
        super().__init__()
        self.fields = fields
        
        # Shared trunk: 'layers' specifies the number of hidden layers
        trunk_layers = []
        for i in range(layers):
            trunk_layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            trunk_layers.append(nn.Tanh())
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Multiple heads: one linear output head per field
        self.heads = nn.ModuleDict({
            field: nn.Linear(hidden, 1) for field in fields
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dict of field tensors {field: (N, 1)}."""
        latent = self.trunk(x)
        return {
            field: head(latent) for field, head in self.heads.items()
        }
