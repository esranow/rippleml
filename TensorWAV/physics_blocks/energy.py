"""
EnergyAwareBlock — Computes the total energy of a wave field and applies a
learnable gating correction to reduce energy drift.

Physics:
  E = 0.5 * (u_t² + c² |∇u|²) + V(u)

Learnable:
  Small gating network (MLP → sigmoid) outputs a correction coefficient that
  scales the state to penalise energy drift.

Returns: (corrected_state, energy_penalty)
"""

import math
import logging
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EnergyAwareBlock(nn.Module):
    """
    Energy-aware correction block for wave fields.

    Args:
        c: Wave speed (default 1.0).
        spatial_dim: Number of spatial dimensions.
        gate_hidden: Hidden size of the gating MLP.
        potential: Potential function name (``"none"`` or ``"quadratic"``).
    """

    def __init__(
        self,
        c: float = 1.0,
        spatial_dim: int = 1,
        gate_hidden: int = 32,
        potential: str = "none",
        **kwargs: Any,
    ):
        super().__init__()
        self.c = c
        self.spatial_dim = spatial_dim
        self.potential = potential

        # Gating network: takes scalar energy → correction coefficient in (0, 1)
        self.gate = nn.Sequential(
            nn.Linear(1, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    def compute_energy(
        self,
        u: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total energy density.

        Args:
            u: ``(B, N, 1)``
            coords: ``(B, N, D+1)`` (spatial + time, requires grad).

        Returns:
            energy: ``(B, 1)`` — mean energy over points.
        """
        grad = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]  # (B, N, D+1)

        u_t = grad[..., -1:]  # (B, N, 1)
        grad_spatial = grad[..., :-1]  # (B, N, D)

        kinetic = 0.5 * u_t ** 2
        gradient_energy = 0.5 * self.c ** 2 * torch.sum(grad_spatial ** 2, dim=-1, keepdim=True)

        V = torch.zeros_like(u)
        if self.potential == "quadratic":
            V = 0.5 * u ** 2

        energy_density = kinetic + gradient_energy + V  # (B, N, 1)
        energy = energy_density.mean(dim=1)  # (B, 1)
        return energy

    # ------------------------------------------------------------------ #
    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            u: ``(B, N, 1)`` field.
            coords: ``(B, N, D+1)`` spatio-temporal coordinates (requires grad).
            params: Unused.

        Returns:
            Tuple of ``(corrected_state, energy_penalty)`` where
            ``energy_penalty`` is a scalar suitable for adding to the loss.
        """
        if coords is None:
            raise ValueError("coords required for energy computation")

        energy = self.compute_energy(u, coords)  # (B, 1)
        gate_coeff = self.gate(energy)  # (B, 1)

        # Correct the state: scale u by gate (near 1 when energy is reasonable)
        corrected = u * gate_coeff.unsqueeze(1)  # broadcast (B,1,1)

        # Energy penalty: penalise deviation from initial energy magnitude
        energy_penalty = torch.mean(energy ** 2)
        return corrected, energy_penalty


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    B, N = 2, 100
    coords = torch.rand(B, N, 2, requires_grad=True)
    u = torch.sin(math.pi * coords[..., 0:1]) * torch.cos(math.pi * coords[..., 1:2])

    block = EnergyAwareBlock(c=1.0, spatial_dim=1)
    corrected, penalty = block(u, coords)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x = coords[0, :, 0].detach().numpy()
    axes[0].scatter(x, u[0].detach().squeeze().numpy(), s=5, label="Original u")
    axes[0].scatter(x, corrected[0].detach().squeeze().numpy(), s=5, label="Corrected")
    axes[0].set_title("State correction")
    axes[0].legend()
    axes[1].bar(["Energy penalty"], [penalty.item()])
    axes[1].set_title("Energy penalty")
    plt.tight_layout()
    plt.savefig("demo_energy.png", dpi=100)
    plt.close()
    print("Saved demo_energy.png")
