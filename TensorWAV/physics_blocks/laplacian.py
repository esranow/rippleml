"""
HybridLaplacianBlock — Hybrid Laplacian operator with learnable MLP correction.

Physics:
  - Point mode: analytical Laplacian via torch.autograd (trace of Hessian)
  - Grid mode: finite-difference stencil via depthwise conv (1D/2D/3D)

Learnable:
  - Small 2-layer MLP residual correction.
"""

import math
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Initialize last layer near zero so correction starts small
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridLaplacianBlock(nn.Module):
    """
    Hybrid Laplacian block combining a physics-based Laplacian with a
    learnable MLP correction.

    Args:
        mode: ``"point"`` (autograd) or ``"grid"`` (finite-difference).
        spatial_dim: Number of spatial dimensions (1, 2, or 3). Used in grid mode.
        dx: Grid spacing for FD stencil (grid mode only).
        correction_hidden: Hidden size of the correction MLP.
        correction_input_dim: Input feature dimension for the MLP (point mode).
        use_correction: Whether to add the learnable correction.
    """

    def __init__(
        self,
        mode: str = "point",
        spatial_dim: int = 1,
        dx: float = 1.0,
        correction_hidden: int = 32,
        correction_input_dim: int = 2,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.mode = mode
        self.spatial_dim = spatial_dim
        self.dx = dx
        self.use_correction = use_correction

        if mode == "grid":
            self._build_fd_kernel(spatial_dim)

        if use_correction:
            mlp_in = correction_input_dim if mode == "point" else spatial_dim
            self.correction_net = _CorrectionMLP(mlp_in, correction_hidden, 1)
        else:
            self.correction_net = None

    # ------------------------------------------------------------------ #
    # Finite-difference kernel construction
    # ------------------------------------------------------------------ #
    def _build_fd_kernel(self, dim: int) -> None:
        """Register a fixed FD Laplacian stencil as a non-learnable buffer."""
        if dim == 1:
            kernel = torch.tensor([[[1.0, -2.0, 1.0]]])          # (1,1,3)
        elif dim == 2:
            kernel = torch.zeros(1, 1, 3, 3)
            kernel[0, 0, 1, 0] = 1.0
            kernel[0, 0, 1, 2] = 1.0
            kernel[0, 0, 0, 1] = 1.0
            kernel[0, 0, 2, 1] = 1.0
            kernel[0, 0, 1, 1] = -4.0
        elif dim == 3:
            kernel = torch.zeros(1, 1, 3, 3, 3)
            kernel[0, 0, 1, 1, 0] = 1.0
            kernel[0, 0, 1, 1, 2] = 1.0
            kernel[0, 0, 1, 0, 1] = 1.0
            kernel[0, 0, 1, 2, 1] = 1.0
            kernel[0, 0, 0, 1, 1] = 1.0
            kernel[0, 0, 2, 1, 1] = 1.0
            kernel[0, 0, 1, 1, 1] = -6.0
        else:
            raise ValueError(f"spatial_dim must be 1, 2, or 3, got {dim}")
        self.register_buffer("fd_kernel", kernel)

    # ------------------------------------------------------------------ #
    # Physics computations
    # ------------------------------------------------------------------ #
    def _laplacian_point(
        self, u: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Autograd-based Laplacian: sum of diagonal Hessian entries."""
        # u: (B, N, 1), coords: (B, N, D) with requires_grad
        grad_u = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]  # (B, N, D)

        lap = torch.zeros_like(u)
        D = coords.shape[-1]
        for i in range(D):
            g_i = grad_u[..., i : i + 1]  # (B, N, 1)
            grad2 = torch.autograd.grad(
                g_i, coords,
                grad_outputs=torch.ones_like(g_i),
                create_graph=True, retain_graph=True,
            )[0]
            lap = lap + grad2[..., i : i + 1]
        return lap  # (B, N, 1)

    def _laplacian_grid(self, u: torch.Tensor) -> torch.Tensor:
        """FD Laplacian via depthwise convolution.  u: (B, C, *spatial)."""
        C = u.shape[1]
        # Expand kernel to depthwise (C groups)
        weight = self.fd_kernel.expand(C, 1, *self.fd_kernel.shape[2:])
        pad = 1
        if self.spatial_dim == 1:
            lap = F.conv1d(u, weight, padding=pad, groups=C)
        elif self.spatial_dim == 2:
            lap = F.conv2d(u, weight, padding=pad, groups=C)
        else:
            lap = F.conv3d(u, weight, padding=pad, groups=C)
        return lap / (self.dx ** 2)

    # ------------------------------------------------------------------ #
    # Correction prep
    # ------------------------------------------------------------------ #
    def _prep_for_nn(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor],
        params: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.mode == "point":
            if coords is not None:
                return torch.cat([u, coords], dim=-1)
            return u
        else:
            # Grid mode: flatten spatial dims for MLP
            B, C = u.shape[:2]
            spatial = u.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)
            return spatial

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hybrid Laplacian.

        Args:
            u: Field values.
                - Point mode: ``(B, N, 1)``; **coords** must be provided.
                - Grid mode: ``(B, C, *spatial)``.
            coords: Spatial coordinates ``(B, N, D)`` (point mode only, requires grad).
            params: Optional PDE parameters (unused by default).

        Returns:
            Laplacian + learnable correction, same shape as input.
        """
        if self.mode == "point":
            if coords is None:
                raise ValueError("coords required for point mode")
            physics = self._laplacian_point(u, coords)
        else:
            physics = self._laplacian_grid(u)

        if self.use_correction and self.correction_net is not None:
            nn_input = self._prep_for_nn(u, coords, params)
            correction = self.correction_net(nn_input)
            if self.mode == "grid":
                B, C = u.shape[:2]
                correction = correction.permute(0, 2, 1).reshape(u.shape)
            return physics + correction
        return physics


# ====================================================================== #
# Demo
# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Point mode demo: Laplacian of sin(pi*x) = -pi^2 sin(pi*x) ---
    B, N, D = 1, 200, 1
    x = torch.linspace(-1, 1, N).reshape(1, N, 1).requires_grad_(True)
    model = nn.Sequential(nn.Identity())  # u = sin(pi*x)
    u = torch.sin(math.pi * x)

    block = HybridLaplacianBlock(mode="point", spatial_dim=1, use_correction=False,
                                  correction_input_dim=2)
    lap = block(u, coords=x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    xv = x.detach().squeeze().numpy()
    axes[0].plot(xv, u.detach().squeeze().numpy(), label="u = sin(πx)")
    axes[0].set_title("Input u")
    axes[0].legend()
    axes[1].plot(xv, lap.detach().squeeze().numpy(), label="Computed ∇²u")
    axes[1].plot(xv, -(math.pi ** 2) * np.sin(math.pi * xv),
                 "--", label="Analytic −π²sin(πx)")
    axes[1].set_title("Laplacian")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("demo_laplacian.png", dpi=100)
    plt.close()
    print("Saved demo_laplacian.png")
