"""
AdaptiveSamplingBlock — Estimates the PDE residual magnitude and returns
normalised sampling weights for collocation point refinement.

Physics:
  - Uses a quick residual estimate (gradient magnitude proxy).

Learnable:
  - Small MLP that refines the raw importance weights.
"""

import logging
from typing import Optional, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AdaptiveSamplingBlock(nn.Module):
    """
    Adaptive sampling weight estimator.

    Args:
        input_dim: Dimension of ``(u, coords)`` features.
        hidden: MLP hidden size.
        use_correction: Whether to apply the learnable refinement.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.use_correction = use_correction

        if use_correction:
            self.refine_net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )

    def _raw_importance(
        self, u: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Proxy importance from gradient magnitude."""
        grad = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]
        return torch.sum(grad ** 2, dim=-1, keepdim=True)  # (B, N, 1)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute normalised sampling weights.

        Args:
            u: ``(B, N, 1)`` field.
            coords: ``(B, N, D)`` (requires grad).

        Returns:
            Weights ``(B, N, 1)`` summing to 1 along N.
        """
        if coords is None:
            raise ValueError("coords required")

        importance = self._raw_importance(u, coords)

        if self.use_correction:
            nn_in = torch.cat([u, coords[..., :1]], dim=-1)
            correction = self.refine_net(nn_in)
            importance = importance + correction

        # Normalise
        weights = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
        return weights


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    import math

    B, N = 1, 200
    x = torch.linspace(-1, 1, N).reshape(1, N, 1).requires_grad_(True)
    u = torch.sin(3 * math.pi * x)

    block = AdaptiveSamplingBlock(input_dim=2, use_correction=False)
    weights = block(u, coords=x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    xv = x.detach().squeeze().numpy()
    axes[0].plot(xv, u.detach().squeeze().numpy(), label="u")
    axes[0].set_title("Field u")
    axes[0].legend()
    axes[1].plot(xv, weights.detach().squeeze().numpy(), label="Sampling weight")
    axes[1].set_title("Adaptive sampling weights")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("demo_adaptivesampler.png", dpi=100)
    plt.close()
    print("Saved demo_adaptivesampler.png")
