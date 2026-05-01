"""
SpectralHybridBlock — FFT-based spectral filtering with learnable frequency
weights and a tiny MLP on amplitudes.

Physics:
  - Forward FFT → apply physics filter (low-pass / band-pass cutoff)
  - Inverse FFT to reconstruct.

Learnable:
  - ``nn.Parameter`` frequency-wise weight vector.
  - Tiny MLP that operates on log-amplitudes and outputs per-frequency corrections.

APIs: forward, apply_filter, get_spectrum
"""

import math
import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class _AmplitudeMLP(nn.Module):
    """Tiny MLP on log-amplitudes."""

    def __init__(self, n_freqs: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freqs, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_freqs),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_amp: torch.Tensor) -> torch.Tensor:
        return self.net(log_amp)


class SpectralHybridBlock(nn.Module):
    """
    Hybrid spectral filtering block.

    Args:
        n_modes: Number of Fourier modes to keep.
        cutoff: Hard cutoff frequency index (modes above are zeroed before
                learnable weighting). Set ``None`` to keep all.
        hidden: MLP hidden size.
        use_correction: Whether to apply the learnable amplitude MLP.
    """

    def __init__(
        self,
        n_modes: int = 64,
        cutoff: Optional[int] = None,
        hidden: int = 32,
        use_correction: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.n_freqs = n_modes // 2 + 1  # rfft output length
        self.cutoff = cutoff if cutoff is not None else self.n_freqs
        self.use_correction = use_correction

        # Learnable per-frequency weight (initialised to 1 = identity)
        self.freq_weights = nn.Parameter(torch.ones(self.n_freqs))

        if use_correction:
            self.amp_mlp = _AmplitudeMLP(self.n_freqs, hidden)
        else:
            self.amp_mlp = None

    # ------------------------------------------------------------------ #
    def get_spectrum(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the real FFT spectrum of *u*.

        Args:
            u: ``(B, N)`` or ``(B, C, N)`` real-valued signal.

        Returns:
            Complex spectrum ``(B, [C], n_freqs)``.
        """
        return torch.fft.rfft(u, dim=-1)

    def apply_filter(
        self, spectrum: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply physics cutoff + learnable frequency weights to a spectrum.

        Args:
            spectrum: Complex tensor from :meth:`get_spectrum`.

        Returns:
            Filtered complex spectrum, same shape.
        """
        n = spectrum.shape[-1]
        w = self.freq_weights[:n].clone()
        # Hard cutoff
        if self.cutoff < n:
            w[self.cutoff :] = 0.0
        return spectrum * w

    # ------------------------------------------------------------------ #
    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: FFT → filter/correct → IFFT.

        Args:
            u: Real signal ``(B, N)`` or ``(B, C, N)``.
            coords: Unused (accepted for API compatibility).
            params: Unused.

        Returns:
            Filtered + corrected signal, same shape as *u*.
        """
        spectrum = self.get_spectrum(u)
        filtered = self.apply_filter(spectrum)

        if self.use_correction and self.amp_mlp is not None:
            amp = torch.abs(filtered)
            log_amp = torch.log1p(amp)
            # Pad/trim to match n_freqs for the MLP
            # If signal is shorter than n_modes, we need padding
            n = log_amp.shape[-1]
            if n < self.n_freqs:
                pad = torch.zeros(
                    *log_amp.shape[:-1], self.n_freqs - n,
                    device=u.device,
                )
                log_amp_padded = torch.cat([log_amp, pad], dim=-1)
            else:
                log_amp_padded = log_amp[..., : self.n_freqs]

            correction = self.amp_mlp(log_amp_padded)[..., :n]
            # Multiplicative correction (1 + small delta)
            filtered = filtered * (1.0 + correction)

        out = torch.fft.irfft(filtered, n=u.shape[-1], dim=-1)
        return out


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    N = 128
    x = torch.linspace(0, 2 * math.pi, N)
    signal = torch.sin(x) + 0.3 * torch.sin(10 * x) + 0.1 * torch.randn(N)
    signal = signal.unsqueeze(0)  # (1, N)

    block = SpectralHybridBlock(n_modes=N, cutoff=8, hidden=16, use_correction=True)
    with torch.no_grad():
        out = block(signal)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(x.numpy(), signal.squeeze().numpy(), label="Input")
    axes[0].plot(x.numpy(), out.squeeze().numpy(), "--", label="Filtered")
    axes[0].set_title("Signal & filtered output")
    axes[0].legend()

    spec = torch.abs(torch.fft.rfft(signal.squeeze()))
    axes[1].stem(spec.numpy()[:32], linefmt="C0-", markerfmt="C0o", basefmt="k-")
    axes[1].axvline(8, color="r", ls="--", label="cutoff")
    axes[1].set_title("Spectrum (first 32 modes)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("demo_spectral.png", dpi=100)
    plt.close()
    print("Saved demo_spectral.png")
