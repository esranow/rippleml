"""
HybridTimeStepperBlock — Physics integrator step + NN corrector for sub-step error.

Physics:
  - Forward Euler or RK2 integration of du/dt = f(u).

Learnable:
  - Small MLP corrector applied after the physics step to compensate truncation error.

APIs:
  - step(u, dt, rhs_fn, params) — one integration step with correction.
  - forward(u, coords, params) — wraps step using internal RHS.
"""

import logging
from typing import Optional, Any, Callable

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HybridTimeStepperBlock(nn.Module):
    """
    Hybrid time-stepper: physics integrator + NN corrector.

    Args:
        state_dim: Dimension of the state vector.
        method: ``"euler"`` or ``"rk2"``.
        correction_hidden: MLP hidden size.
        use_correction: Whether to add the NN corrector.
        default_dt: Default time step.
    """

    def __init__(
        self,
        state_dim: int = 1,
        method: str = "rk2",
        correction_hidden: int = 32,
        use_correction: bool = True,
        default_dt: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.method = method.lower()
        self.default_dt = default_dt
        self.use_correction = use_correction

        if use_correction:
            self.corrector = nn.Sequential(
                nn.Linear(state_dim, correction_hidden),
                nn.Tanh(),
                nn.Linear(correction_hidden, state_dim),
            )
            nn.init.zeros_(self.corrector[-1].weight)
            nn.init.zeros_(self.corrector[-1].bias)

        # Simple default RHS: -u (decay)
        self._default_rhs = lambda u: -u

    def step(
        self,
        u: torch.Tensor,
        dt: Optional[float] = None,
        rhs_fn: Optional[Callable] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One integration step.

        Args:
            u: ``(B, D)`` state.
            dt: Time step (uses default if None).
            rhs_fn: Callable f(u) → du/dt. Uses internal default if None.

        Returns:
            Updated state ``(B, D)``.
        """
        if dt is None:
            dt = self.default_dt
        f = rhs_fn if rhs_fn is not None else self._default_rhs

        if self.method == "euler":
            u_new = u + dt * f(u)
        elif self.method == "rk2":
            k1 = f(u)
            k2 = f(u + 0.5 * dt * k1)
            u_new = u + dt * k2
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.use_correction:
            u_new = u_new + self.corrector(u_new)

        return u_new

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single step using default RHS and dt."""
        return self.step(u, dt=self.default_dt)


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    block = HybridTimeStepperBlock(state_dim=1, method="rk2",
                                    use_correction=False, default_dt=0.1)
    # Simple exponential decay: du/dt = -u  → u(t) = exp(-t)
    u = torch.tensor([[1.0]])
    traj = [u.item()]
    for _ in range(50):
        u = block.step(u, dt=0.1, rhs_fn=lambda u: -u)
        traj.append(u.item())

    import math
    t = np.arange(len(traj)) * 0.1
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, traj, "o-", ms=3, label="RK2 stepper")
    ax.plot(t, np.exp(-t), "--", label="Exact exp(-t)")
    ax.set_title("HybridTimeStepperBlock")
    ax.legend()
    plt.tight_layout()
    plt.savefig("demo_hybrid_stepper.png", dpi=100)
    plt.close()
    print("Saved demo_hybrid_stepper.png")
