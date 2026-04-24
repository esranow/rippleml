"""
HybridWaveResidualBlock — Applies a learnable MLP correction on top of
the residual computed by ripple.physics.equation.Equation.

All PDE physics (u_tt, u_t, Lap) are computed by Equation; this block
adds only the learnable correction term.
"""

import logging
from typing import Optional, Any

import torch
import torch.nn as nn

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

    def residual(
        self,
        u: torch.Tensor,
        equation,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate PDE residual to Equation, add MLP correction.

        Args:
            u: Model output ``(N, 1)`` computed from inputs.
            equation: ripple.physics.equation.Equation instance.
            inputs: cat([x, t]) with requires_grad=True — same tensor model used.
        """
        res = equation.compute_residual(u, inputs)

        if self.use_correction and self.correction_net is not None:
            nn_in = torch.cat([u, inputs], dim=-1)
            res = res + self.correction_net(nn_in)
        return res

    def loss(
        self,
        u: torch.Tensor,
        equation,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean-squared residual."""
        return torch.mean(self.residual(u, equation, inputs) ** 2)

    def forward(
        self,
        u: torch.Tensor,
        equation=None,
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if equation is None or inputs is None:
            raise ValueError("equation and inputs required")
        return self.residual(u, equation, inputs)
