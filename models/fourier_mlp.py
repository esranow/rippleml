import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from ripple.models.registry import register_model
from ripple.models.mlp import MLP

@register_model("fourier_mlp")
class FourierMLP(nn.Module):
    """
    MLP with Random Fourier Feature Mapping.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        sigma: float = 1.0,
        mapping_size: int = 256,
        activation: str = "tanh",
        **kwargs: Any
    ):
        """
        Initialize Fourier MLP.

        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            hidden_layers (List[int]): Hidden layers for the MLP.
            sigma (float): Standard deviation for Gaussian mapping.
            mapping_size (int): Size of the Fourier feature mapping (number of features).
            activation (str): Activation for MLP.
        """
        super().__init__()
        self.sigma = sigma
        self.mapping_size = mapping_size
        
        # B matrix for mapping: (input_dim, mapping_size)
        # We perform mapping: x -> [cos(2pi B x), sin(2pi B x)]
        # This projects input_dim to 2 * mapping_size
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * sigma)
        
        self.mlp = MLP(
            input_dim=2 * mapping_size,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Fourier mapping.

        Args:
            x (torch.Tensor): (B, N, F_in)

        Returns:
            torch.Tensor: (B, N, F_out)
        """
        # x: (B, N, F_in)
        # B: (F_in, M)
        # x @ B -> (B, N, M)
        projected = (2 * np.pi) * (x @ self.B)
        
        # Fourier features
        features = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
        
        return self.mlp(features)
