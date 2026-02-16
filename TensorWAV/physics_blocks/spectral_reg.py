"""
SpectralRegularizationBlock — Computes an FFT-based spectral loss penalty
to regularise high-frequency content.

Physics:
  - Compute power spectrum via FFT.
  - Penalise energy in modes above a threshold.

Returns: scalar spectral loss term.
"""

import math
import logging
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SpectralRegularizationBlock(nn.Module):
    """
    Spectral regularisation block returning a scalar loss term.

    Args:
        cutoff: Mode index above which energy is penalised.
        penalty_weight: Multiplier for the penalty.
        norm: ``"l2"`` or ``"l1"`` penalty on high-frequency amplitudes.
    """

    def __init__(
        self,
        cutoff: int = 16,
        penalty_weight: float = 1.0,
        norm: str = "l2",
        **kwargs: Any,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.penalty_weight = penalty_weight
        self.norm = norm

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute spectral regularisation penalty.

        Args:
            u: ``(B, N)`` or ``(B, C, N)`` real signal.

        Returns:
            Scalar penalty tensor.
        """
        spec = torch.fft.rfft(u, dim=-1)
        amp = torch.abs(spec)

        high_freq = amp[..., self.cutoff:]

        if self.norm == "l1":
            penalty = high_freq.abs().mean()
        else:
            penalty = (high_freq ** 2).mean()

        return self.penalty_weight * penalty


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    N = 128
    x = torch.linspace(0, 2 * math.pi, N)
    clean = torch.sin(x)
    noisy = clean + 0.5 * torch.sin(30 * x)

    block = SpectralRegularizationBlock(cutoff=8, penalty_weight=1.0, norm="l2")
    loss_clean = block(clean.unsqueeze(0))
    loss_noisy = block(noisy.unsqueeze(0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(x.numpy(), clean.numpy(), label="Clean")
    axes[0].plot(x.numpy(), noisy.numpy(), label="Noisy")
    axes[0].set_title("Signals")
    axes[0].legend()
    axes[1].bar(["Clean", "Noisy"], [loss_clean.item(), loss_noisy.item()])
    axes[1].set_title("Spectral reg. penalty")
    plt.tight_layout()
    plt.savefig("demo_spectral_reg.png", dpi=100)
    plt.close()
    print("Saved demo_spectral_reg.png")
