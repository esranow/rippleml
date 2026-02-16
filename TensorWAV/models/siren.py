import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from TensorWAV.models.registry import register_model

class SineLayer(nn.Module):
    """
    Linear layer with Sine activation and specialized initialization.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                          np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

@register_model("siren")
class Siren(nn.Module):
    """
    Sinusoidal Representation Network (SIREN).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        omega_0: float = 30.0,
        **kwargs: Any
    ):
        """
        Initialize SIREN.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_layers (List[int]): List of hidden dimensions.
            omega_0 (float): Frequency scaling factor.
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Hidden layers
        for i, h_dim in enumerate(hidden_layers):
            is_first = (i == 0)
            layers.append(SineLayer(in_dim, h_dim, is_first=is_first, omega_0=omega_0))
            in_dim = h_dim
            
        # Last layer is usually linear for regression tasks in SIREN papers
        # The prompt implies a general SIREN model. Usually the final layer is linear
        # to match the output range, but sometimes sine is used if we want bounded output.
        # Standard implementation (Sitzmann et al.) uses Linear for the last layer.
        self.net = nn.Sequential(*layers)
        self.last_linear = nn.Linear(in_dim, output_dim)
        
        # Init last linear layer
        with torch.no_grad():
            self.last_linear.weight.uniform_(-np.sqrt(6 / in_dim) / omega_0, 
                                           np.sqrt(6 / in_dim) / omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): (B, N, F_in)
            
        Returns:
            torch.Tensor: (B, N, F_out)
        """
        out = self.net(x)
        return self.last_linear(out)
