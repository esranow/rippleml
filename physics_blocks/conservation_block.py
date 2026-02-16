"""
ConservationConstraintBlock — Enforce a conserved quantity via projection
plus a small learned correction.

Physics:
  - Compute integral quantity Q(u) (e.g., total mass, energy).
  - Project state so Q is preserved.

Learnable:
  - Small MLP that outputs a correction to reduce constraint violation.
"""

import logging
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ConservationConstraintBlock(nn.Module):
    """
    Conservation constraint block.

    Args:
        mode: ``"mass"`` (integral of u) or ``"energy"`` (integral of u²).
        correction_hidden: MLP hidden size.
        use_correction: Whether to apply learned correction.
    """

    def __init__(
        self,
        mode: str = "mass",
        correction_hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.use_correction = use_correction

        if use_correction:
            self.correction_net = nn.Sequential(
                nn.Linear(2, correction_hidden),
                nn.Tanh(),
                nn.Linear(correction_hidden, 1),
            )
            nn.init.zeros_(self.correction_net[-1].weight)
            nn.init.zeros_(self.correction_net[-1].bias)

    def _compute_quantity(self, u: torch.Tensor) -> torch.Tensor:
        """Compute conserved quantity.  u: (B, N, 1). Returns (B, 1)."""
        if self.mode == "mass":
            return u.mean(dim=1)  # (B, 1)
        elif self.mode == "energy":
            return (u ** 2).mean(dim=1)  # (B, 1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
        target_quantity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project u to enforce conservation and return constraint violation.

        Args:
            u: ``(B, N, 1)`` field.
            coords: ``(B, N, D)`` (used for correction MLP if available).
            target_quantity: ``(B, 1)`` target conserved value (defaults to current).

        Returns:
            ``(corrected_u, violation)`` where violation is a scalar loss.
        """
        Q_current = self._compute_quantity(u)
        if target_quantity is None:
            target_quantity = Q_current.detach()

        # Simple projection: shift u uniformly to match target
        if self.mode == "mass":
            shift = (target_quantity - Q_current).unsqueeze(1)  # (B, 1, 1)
            u_proj = u + shift
        elif self.mode == "energy":
            ratio = (target_quantity / (Q_current + 1e-8)).sqrt().unsqueeze(1)
            u_proj = u * ratio
        else:
            u_proj = u

        if self.use_correction and coords is not None:
            nn_in = torch.cat([u_proj, coords[..., :1]], dim=-1)
            correction = self.correction_net(nn_in)
            u_proj = u_proj + correction

        Q_new = self._compute_quantity(u_proj)
        violation = torch.mean((Q_new - target_quantity) ** 2)
        return u_proj, violation


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    import math

    B, N = 1, 100
    x = torch.linspace(0, 1, N).reshape(1, N, 1)
    u = torch.sin(math.pi * x)
    target_mass = u.mean(dim=1).detach()

    block = ConservationConstraintBlock(mode="mass", use_correction=False)
    u_perturbed = u + 0.5 * torch.randn_like(u)
    u_proj, viol = block(u_perturbed, coords=x, target_quantity=target_mass)

    fig, ax = plt.subplots(figsize=(6, 3))
    xv = x.squeeze().numpy()
    ax.plot(xv, u_perturbed.squeeze().detach().numpy(), label="Perturbed")
    ax.plot(xv, u_proj.squeeze().detach().numpy(), "--", label="Projected")
    ax.set_title(f"Conservation projection (violation={viol.item():.2e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("demo_conservation.png", dpi=100)
    plt.close()
    print("Saved demo_conservation.png")
