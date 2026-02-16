"""
HybridGradientBlock — Computes ∇u via autograd with an MLP correction.

Physics:
  - Gradient ∇u computed using torch.autograd.grad.

Learnable:
  - Small MLP residual added to the physics gradient.
"""

import math
import logging
from typing import Optional, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HybridGradientBlock(nn.Module):
    """
    Hybrid gradient block: autograd ∇u + learned MLP correction.

    Args:
        spatial_dim: Number of spatial dimensions.
        correction_hidden: MLP hidden size.
        use_correction: Toggle learnable correction.
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        correction_hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.use_correction = use_correction

        if use_correction:
            # Input: u(1) + coords(D) → output D
            self.correction_net = nn.Sequential(
                nn.Linear(spatial_dim + 1, correction_hidden),
                nn.Tanh(),
                nn.Linear(correction_hidden, correction_hidden),
                nn.Tanh(),
                nn.Linear(correction_hidden, spatial_dim),
            )
            nn.init.zeros_(self.correction_net[-1].weight)
            nn.init.zeros_(self.correction_net[-1].bias)

    def _compute_gradient(
        self, u: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Autograd gradient ∇u.  u: (B,N,1), coords: (B,N,D)."""
        grad_u = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]
        return grad_u  # (B, N, D)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hybrid gradient.

        Args:
            u: ``(B, N, 1)``
            coords: ``(B, N, D)`` (requires grad)
            params: Unused.

        Returns:
            Gradient ``(B, N, D)``
        """
        if coords is None:
            raise ValueError("coords required for gradient computation")
        grad_u = self._compute_gradient(u, coords)

        if self.use_correction:
            nn_in = torch.cat([u, coords], dim=-1)
            correction = self.correction_net(nn_in)
            return grad_u + correction
        return grad_u


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    B, N = 1, 200
    x = torch.linspace(-1, 1, N).reshape(1, N, 1).requires_grad_(True)
    u = torch.sin(math.pi * x)

    block = HybridGradientBlock(spatial_dim=1, use_correction=False)
    grad_u = block(u, coords=x)

    fig, ax = plt.subplots(figsize=(6, 3))
    xv = x.detach().squeeze().numpy()
    ax.plot(xv, grad_u.detach().squeeze().numpy(), label="Computed ∇u")
    ax.plot(xv, math.pi * np.cos(math.pi * xv), "--", label="Analytic π cos(πx)")
    ax.set_title("Gradient block")
    ax.legend()
    plt.tight_layout()
    plt.savefig("demo_gradient.png", dpi=100)
    plt.close()
    print("Saved demo_gradient.png")
