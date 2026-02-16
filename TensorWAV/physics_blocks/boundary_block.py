"""
BoundaryConditionBlock — Hard enforcement of Dirichlet / Neumann / periodic BCs
with a local NN correction near the boundary.

Physics:
  - Dirichlet: u(boundary) = g
  - Neumann: ∂u/∂n(boundary) = h
  - Periodic: u(left) = u(right)

Learnable:
  - Small MLP correction active only in a narrow boundary strip.
"""

import logging
from typing import Optional, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BoundaryConditionBlock(nn.Module):
    """
    Boundary condition enforcement block.

    Args:
        bc_type: ``"dirichlet"``, ``"neumann"``, or ``"periodic"``.
        bc_value: Boundary value for Dirichlet/Neumann (default 0).
        boundary_width: Fractional width of the boundary strip (0–1) where
                        the NN correction is active.
        domain: Tuple ``(low, high)`` for the spatial domain.
        correction_hidden: MLP hidden size.
        use_correction: Whether to apply NN correction near boundary.
    """

    def __init__(
        self,
        bc_type: str = "dirichlet",
        bc_value: float = 0.0,
        boundary_width: float = 0.1,
        domain: tuple = (-1.0, 1.0),
        correction_hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.bc_type = bc_type.lower()
        self.bc_value = bc_value
        self.boundary_width = boundary_width
        self.domain = domain
        self.use_correction = use_correction

        if use_correction:
            self.correction_net = nn.Sequential(
                nn.Linear(2, correction_hidden),
                nn.Tanh(),
                nn.Linear(correction_hidden, 1),
            )
            nn.init.zeros_(self.correction_net[-1].weight)
            nn.init.zeros_(self.correction_net[-1].bias)

    def _boundary_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """Smooth mask ~1 near boundary, ~0 in interior.  coords: (B,N,D)."""
        x = coords[..., 0]  # first spatial dim
        lo, hi = self.domain
        width = (hi - lo) * self.boundary_width
        mask_lo = torch.sigmoid(-(x - lo - width) / (width * 0.25 + 1e-8))
        mask_hi = torch.sigmoid((x - hi + width) / (width * 0.25 + 1e-8))
        return (mask_lo + mask_hi).unsqueeze(-1).clamp(0, 1)  # (B, N, 1)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply boundary condition enforcement.

        Args:
            u: ``(B, N, 1)`` field.
            coords: ``(B, N, D)`` spatial coordinates.
            params: Unused.

        Returns:
            BC-corrected field ``(B, N, 1)``.
        """
        if coords is None:
            raise ValueError("coords required for BoundaryConditionBlock")

        mask = self._boundary_mask(coords)

        if self.bc_type == "dirichlet":
            u_corrected = u * (1.0 - mask) + self.bc_value * mask
        elif self.bc_type == "neumann":
            u_corrected = u  # Neumann penalised via loss; pass through here
        elif self.bc_type == "periodic":
            # Blend left/right boundary values
            u_corrected = u * (1.0 - mask) + u.flip(dims=[1]) * mask
        else:
            raise ValueError(f"Unknown bc_type: {self.bc_type}")

        if self.use_correction and coords is not None:
            nn_in = torch.cat([u, coords[..., :1]], dim=-1)  # (B,N,2)
            correction = self.correction_net(nn_in) * mask
            u_corrected = u_corrected + correction

        return u_corrected


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    import math

    B, N = 1, 200
    x = torch.linspace(-1, 1, N).reshape(1, N, 1)
    u = torch.sin(math.pi * x)  # (1, N, 1), should be 0 at boundaries

    block = BoundaryConditionBlock(bc_type="dirichlet", bc_value=0.0,
                                    boundary_width=0.1, use_correction=False)
    u_bc = block(u, coords=x)

    fig, ax = plt.subplots(figsize=(6, 3))
    xv = x.squeeze().numpy()
    ax.plot(xv, u.squeeze().detach().numpy(), label="Original u")
    ax.plot(xv, u_bc.squeeze().detach().numpy(), "--", label="BC-enforced")
    ax.set_title("Dirichlet BC enforcement")
    ax.legend()
    plt.tight_layout()
    plt.savefig("demo_boundary.png", dpi=100)
    plt.close()
    print("Saved demo_boundary.png")
