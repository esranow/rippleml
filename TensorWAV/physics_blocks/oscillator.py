"""
HybridOscillatorBlock — Harmonic + Duffing oscillator core with learnable
frequency offset, damping, and MLP residual.

Physics:
  u_tt + ω² u + α u³ = 0  (undamped Duffing)

Learnable:
  - ``nn.Parameter`` for ω offset and damping coefficient.
  - Tiny MLP residual applied in the step function.

APIs:
  - step(state, dt, params) — single RK2 integration step.
  - forward(u, coords, params) — continuous evaluation using autograd.
"""

import math
import logging
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class _ResidualMLP(nn.Module):
    def __init__(self, state_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridOscillatorBlock(nn.Module):
    """
    Hybrid Duffing/harmonic oscillator block.

    State vector ``s = [u, v]`` where ``v = du/dt``.

    Args:
        omega: Natural frequency (default 1.0).
        alpha: Cubic nonlinearity coefficient (0 gives harmonic oscillator).
        damping: Initial damping coefficient.
        hidden: MLP hidden size.
        use_correction: Whether to add the MLP residual.
    """

    def __init__(
        self,
        omega: float = 1.0,
        alpha: float = 0.0,
        damping: float = 0.0,
        hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.omega_base = omega
        self.alpha = alpha
        self.use_correction = use_correction

        # Learnable offsets
        self.omega_offset = nn.Parameter(torch.tensor(0.0))
        self.damping = nn.Parameter(torch.tensor(damping))

        if use_correction:
            self.correction_mlp = _ResidualMLP(2, hidden)
        else:
            self.correction_mlp = None

    @property
    def omega(self) -> torch.Tensor:
        return self.omega_base + self.omega_offset

    # ------------------------------------------------------------------ #
    def _dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Compute ds/dt = [v, -ω²u - αu³ - γv].

        Args:
            state: ``(..., 2)`` last dim is ``[u, v]``.
        """
        u = state[..., 0:1]
        v = state[..., 1:2]

        omega = self.omega
        a = -omega ** 2 * u - self.alpha * u ** 3 - self.damping * v
        dsdt = torch.cat([v, a], dim=-1)

        if self.use_correction and self.correction_mlp is not None:
            dsdt = dsdt + self.correction_mlp(state)
        return dsdt

    # ------------------------------------------------------------------ #
    def step(
        self,
        state: torch.Tensor,
        dt: float = 0.01,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single RK2 (midpoint) integration step.

        Args:
            state: ``(B, 2)`` or ``(B, N, 2)`` current ``[u, v]``.
            dt: Time step.
            params: Unused.

        Returns:
            Next state, same shape.
        """
        k1 = self._dynamics(state)
        k2 = self._dynamics(state + 0.5 * dt * k1)
        return state + dt * k2

    # ------------------------------------------------------------------ #
    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Continuous forward: given ``u`` as ``(B, 2)`` state ``[u, v]``,
        return the dynamics ``ds/dt``.
        """
        return self._dynamics(u)


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    block = HybridOscillatorBlock(omega=2.0, alpha=0.5, damping=0.05,
                                   use_correction=False)
    state = torch.tensor([[1.0, 0.0]])  # u=1, v=0
    dt = 0.01
    trajectory = [state.squeeze().numpy().copy()]
    for _ in range(500):
        state = block.step(state, dt)
        trajectory.append(state.detach().squeeze().numpy().copy())
    trajectory = np.array(trajectory)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    t = np.arange(len(trajectory)) * dt
    axes[0].plot(t, trajectory[:, 0], label="u(t)")
    axes[0].plot(t, trajectory[:, 1], label="v(t)")
    axes[0].set_title("Duffing oscillator trajectory")
    axes[0].legend()
    axes[1].plot(trajectory[:, 0], trajectory[:, 1])
    axes[1].set_title("Phase portrait")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("v")
    plt.tight_layout()
    plt.savefig("demo_oscillator.png", dpi=100)
    plt.close()
    print("Saved demo_oscillator.png")
