import torch
import torch.nn as nn
from typing import List, Union, Dict, Any
from ripple.models.registry import register_model

@register_model("mlp")
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) supporting (B, N, F) input.
    Operates on the last dimension F.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        **kwargs: Any
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            hidden_layers (List[int]): List of hidden layer sizes.
            activation (str): Activation function name (e.g., "tanh", "relu", "gelu").
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Resolve activation function
        act_fn = self._get_activation(activation)

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """
        Get activation module by name.
        """
        acc = name.lower()
        if acc == "tanh":
            return nn.Tanh()
        elif acc == "relu":
            return nn.ReLU()
        elif acc == "gelu":
            return nn.GELU()
        elif acc == "sigmoid":
            return nn.Sigmoid()
        elif acc == "identity":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, F_in).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, F_out).
        """
        # Linear layers in PyTorch support arbitrary batch dimensions inputs (*, H_in)
        return self.net(x)
