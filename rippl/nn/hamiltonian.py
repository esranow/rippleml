"""
HamiltonianBlock — Learns a Hamiltonian H(q, p) via a small MLP and computes
symplectic leapfrog updates.

Physics:
  dq/dt = ∂H/∂p,   dp/dt = −∂H/∂q   (Hamilton's equations)

Learnable:
  - MLP approximation of H(q, p).
  - Symplectic leapfrog integrator preserves structure.
"""

import logging
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HamiltonianBlock(nn.Module):
    """
    Hamiltonian neural network block with symplectic integration.

    Args:
        state_dim: Dimension of q (same for p, so phase space = 2 * state_dim).
        hidden: MLP hidden size.
    """

    def __init__(
        self,
        state_dim: int = 1,
        hidden: int = 32,
        **kwargs: Any,
    ):
        super().__init__()
        self.state_dim = state_dim

        # H(q, p) network
        self.h_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate H(q, p).

        Args:
            q, p: ``(B, D)``

        Returns:
            ``(B, 1)``
        """
        z = torch.cat([q, p], dim=-1)
        return self.h_net(z)

    def _grad_H(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (∂H/∂q, ∂H/∂p)."""
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.hamiltonian(q, p)
        dH = torch.autograd.grad(
            H.sum(), [q, p], create_graph=True, retain_graph=True,
        )
        return dH[0], dH[1]  # dH/dq, dH/dp

    def step(
        self,
        state: torch.Tensor,
        dt: float = 0.01,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Symplectic leapfrog step.

        Args:
            state: ``(B, 2*D)`` concatenated ``[q, p]``.
            dt: Time step.

        Returns:
            Updated state ``(B, 2*D)``.
        """
        D = self.state_dim
        q = state[..., :D]
        p = state[..., D:]

        # Half-step p
        dHdq, _ = self._grad_H(q, p)
        p_half = p - 0.5 * dt * dHdq

        # Full-step q
        _, dHdp = self._grad_H(q, p_half)
        q_new = q + dt * dHdp

        # Half-step p again
        dHdq_new, _ = self._grad_H(q_new, p_half)
        p_new = p_half - 0.5 * dt * dHdq_new

        return torch.cat([q_new, p_new], dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return Hamilton's equations: ``[dq/dt, dp/dt]``.

        Args:
            u: ``(B, 2*D)`` state ``[q, p]``.
        """
        D = self.state_dim
        q, p = u[..., :D], u[..., D:]
        dHdq, dHdp = self._grad_H(q, p)
        return torch.cat([dHdp, -dHdq], dim=-1)


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    block = HamiltonianBlock(state_dim=1, hidden=32)
    # Initialise H_net to approximate harmonic oscillator H = 0.5*(q²+p²)
    # (not trained here — just demo the API)
    state = torch.tensor([[1.0, 0.0]])
    dt = 0.05
    traj = [state.detach().numpy().copy()]
    for _ in range(200):
        state = block.step(state, dt)
        traj.append(state.detach().numpy().copy())
    traj = np.array(traj).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    t = np.arange(len(traj)) * dt
    axes[0].plot(t, traj[:, 0], label="q(t)")
    axes[0].plot(t, traj[:, 1], label="p(t)")
    axes[0].set_title("Hamiltonian trajectory")
    axes[0].legend()
    axes[1].plot(traj[:, 0], traj[:, 1])
    axes[1].set_title("Phase space")
    plt.tight_layout()
    plt.savefig("demo_hamiltonian.png", dpi=100)
    plt.close()
    print("Saved demo_hamiltonian.png")
