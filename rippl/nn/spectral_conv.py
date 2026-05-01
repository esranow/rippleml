"""
SpectralConvBlock — FNO-style spectral convolution with a local conv residual.

Physics:
  - Spectral multiplication in Fourier space (FNO kernel).

Learnable:
  - Complex weight matrix for spectral modes.
  - Local 1-D convolution bypass (residual path).
"""

import logging
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SpectralConvBlock(nn.Module):
    """
    FNO-style spectral convolution + local conv residual.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        modes: Number of Fourier modes to keep.
        use_local_conv: Whether to add the local conv residual path.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 16,
        use_local_conv: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.use_local_conv = use_local_conv

        # Complex weight for spectral convolution
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, 2)
        )

        if use_local_conv:
            self.local_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                                         padding=1)
        else:
            self.local_conv = None

    def _spectral_mul(
        self, x_ft: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Multiply spectral coefficients by complex weights.

        x_ft: (B, C_in, modes) complex;  weights: (C_in, C_out, modes, 2)
        """
        w = torch.view_as_complex(weights)  # (C_in, C_out, modes)
        return torch.einsum("bim,iom->bom", x_ft, w)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            u: ``(B, C, N)`` spatial signal.

        Returns:
            ``(B, C_out, N)``
        """
        N = u.shape[-1]
        x_ft = torch.fft.rfft(u, dim=-1)

        # Keep only first `modes` modes
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            u.shape[0], self.out_channels, x_ft.shape[-1],
            dtype=torch.cfloat, device=u.device,
        )
        out_ft[..., :m] = self._spectral_mul(x_ft[..., :m], self.weights[..., :m, :])

        out = torch.fft.irfft(out_ft, n=N, dim=-1)

        if self.use_local_conv and self.local_conv is not None:
            out = out + self.local_conv(u)

        return out


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    import math

    N = 64
    x = torch.linspace(0, 2 * math.pi, N)
    u = (torch.sin(x) + 0.3 * torch.sin(5 * x)).unsqueeze(0).unsqueeze(0)

    block = SpectralConvBlock(in_channels=1, out_channels=1, modes=8)
    with torch.no_grad():
        out = block(u)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x.numpy(), u.squeeze().numpy(), label="Input")
    ax.plot(x.numpy(), out.squeeze().numpy(), "--", label="SpectralConv out")
    ax.set_title("SpectralConvBlock")
    ax.legend()
    plt.tight_layout()
    plt.savefig("demo_spectral_conv.png", dpi=100)
    plt.close()
    print("Saved demo_spectral_conv.png")
