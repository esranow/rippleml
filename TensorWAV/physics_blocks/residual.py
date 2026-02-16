"""
HybridWaveResidualBlock — Computes the PDE residual for wave-type equations
with an optional learnable NN correction.

Physics:
  residual = a * u_tt + b * u_t − c * Lap(u) + NN_correction(u)

APIs:
  - forward(u, coords, params)   → corrected residual tensor
  - residual(u, coords, params)  → raw residual (same as forward)
  - loss(u, coords, params)      → mean-squared residual scalar
"""

import math
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class _CorrectionMLP(nn.Module):
    """Tiny MLP for residual correction."""

    def __init__(self, input_dim: int, hidden: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridWaveResidualBlock(nn.Module):
    """
    Hybrid wave-equation residual block.

    Computes ``a * u_tt + b * u_t − c * Lap(u) + correction(u)`` where the
    Laplacian is computed via autograd and the correction is a small MLP
    (deactivated by default so that analytic solutions give near-zero residual).

    Args:
        a: Coefficient for u_tt (default 1.0).
        b: Coefficient for u_t (default 0.0).
        c: Coefficient for Laplacian(u) (wave speed squared, default 1.0).
        spatial_dim: Number of spatial dimensions.
        correction_hidden: MLP hidden size.
        use_correction: Toggle MLP correction on/off.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 1.0,
        spatial_dim: int = 1,
        correction_hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.spatial_dim = spatial_dim
        self.use_correction = use_correction

        # coords = (x1,..,xD, t) → D+1; input to MLP = coords + u  → D+2
        if use_correction:
            mlp_in = spatial_dim + 2  # coords(D+1) + u(1)
            self.correction_net = _CorrectionMLP(mlp_in, correction_hidden, 1)
        else:
            self.correction_net = None

    # ------------------------------------------------------------------ #
    def _compute_derivatives(
        self, u: torch.Tensor, coords: torch.Tensor
    ):
        """Return u_t, u_tt, and Laplacian(u) via autograd.

        Args:
            u: (B, N, 1)
            coords: (B, N, D+1) last dim = time, requires grad.

        Returns:
            u_t, u_tt, laplacian — each (B, N, 1)
        """
        grad = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]  # (B, N, D+1)

        u_t = grad[..., -1:]  # (B, N, 1)

        # u_tt
        grad_ut = torch.autograd.grad(
            u_t, coords,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True,
        )[0]
        u_tt = grad_ut[..., -1:]  # (B, N, 1)

        # Laplacian: sum u_{x_i x_i}
        D = coords.shape[-1] - 1  # spatial dims
        lap = torch.zeros_like(u)
        for i in range(D):
            g_i = grad[..., i : i + 1]
            g2 = torch.autograd.grad(
                g_i, coords,
                grad_outputs=torch.ones_like(g_i),
                create_graph=True, retain_graph=True,
            )[0]
            lap = lap + g2[..., i : i + 1]

        return u_t, u_tt, lap

    # ------------------------------------------------------------------ #
    def residual(
        self,
        u: torch.Tensor,
        coords: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the PDE residual.

        Args:
            u: Predicted field ``(B, N, 1)``.
            coords: Spatio-temporal coordinates ``(B, N, D+1)`` (requires grad).
            params: Optional PDE parameters (unused currently).

        Returns:
            Residual tensor ``(B, N, 1)``.
        """
        u_t, u_tt, lap = self._compute_derivatives(u, coords)
        res = self.a * u_tt + self.b * u_t - self.c * lap

        if self.use_correction and self.correction_net is not None:
            nn_in = torch.cat([u, coords], dim=-1)
            correction = self.correction_net(nn_in)
            res = res + correction
        return res

    def loss(
        self,
        u: torch.Tensor,
        coords: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return mean-squared residual (convenience for training)."""
        res = self.residual(u, coords, params)
        return torch.mean(res ** 2)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for :meth:`residual`."""
        if coords is None:
            raise ValueError("coords required for residual computation")
        return self.residual(u, coords, params)


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Analytic wave solution: u(x,t) = sin(pi*x)*cos(pi*t), c=1
    B, N = 1, 100
    x = torch.linspace(0, 1, N).unsqueeze(0).unsqueeze(-1)
    t = torch.linspace(0, 1, N).unsqueeze(0).unsqueeze(-1)
    coords = torch.cat([x, t], dim=-1).requires_grad_(True)  # (1, N, 2)
    u = torch.sin(math.pi * coords[..., 0:1]) * torch.cos(math.pi * coords[..., 1:2])

    block = HybridWaveResidualBlock(a=1.0, b=0.0, c=1.0, use_correction=False)
    res = block.residual(u, coords)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    xv = coords[..., 0].detach().squeeze().numpy()
    axes[0].plot(xv, u.detach().squeeze().numpy(), label="u")
    axes[0].set_title("Analytic wave u(x,t)")
    axes[0].legend()
    axes[1].plot(xv, res.detach().squeeze().numpy(), label="Residual")
    axes[1].set_title("PDE Residual (should ≈ 0)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("demo_residual.png", dpi=100)
    plt.close()
    print("Saved demo_residual.png")
