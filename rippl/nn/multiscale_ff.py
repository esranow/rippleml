"""
MultiScaleFourierFeatureBlock — Progressively increasing Fourier feature
frequencies with learnable gating.

Produces multi-scale positional embeddings that mitigate spectral bias by
exposing the network to a wide range of frequency components.
"""

import math
import logging
from typing import Optional, Any, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MultiScaleFourierFeatureBlock(nn.Module):
    """
    Multi-scale random Fourier feature block with learnable gating.

    Args:
        input_dim: Dimension of input coordinates.
        n_scales: Number of frequency scales.
        base_sigma: Lowest-scale standard deviation of random frequencies.
        scale_factor: Multiplicative factor between consecutive scales.
        features_per_scale: Number of Fourier features per scale.
    """

    def __init__(
        self,
        input_dim: int = 1,
        n_scales: int = 4,
        base_sigma: float = 1.0,
        scale_factor: float = 2.0,
        features_per_scale: int = 16,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_scales = n_scales
        self.features_per_scale = features_per_scale

        # Random frequency matrices (fixed)
        total = n_scales * features_per_scale
        B_matrices: List[torch.Tensor] = []
        sigma = base_sigma
        for _ in range(n_scales):
            B_matrices.append(torch.randn(input_dim, features_per_scale) * sigma)
            sigma *= scale_factor
        B_all = torch.cat(B_matrices, dim=1)  # (input_dim, total)
        self.register_buffer("B", B_all)

        # Learnable per-scale gate (initialised to uniform)
        self.gate_logits = nn.Parameter(torch.zeros(n_scales))

        self.output_dim = total * 2  # sin + cos

    @property
    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward(
        self,
        u: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-scale Fourier features.

        Args:
            u: Unused (API compatibility).
            coords: ``(B, N, D)`` input coordinates.

        Returns:
            ``(B, N, output_dim)`` Fourier features.
        """
        if coords is None:
            raise ValueError("coords required")

        # coords @ B → (B, N, total)
        proj = coords @ self.B  # (B, N, F*n_scales)

        features = torch.cat([torch.sin(2 * math.pi * proj),
                               torch.cos(2 * math.pi * proj)], dim=-1)

        # Apply per-scale gating
        gates = self.gates
        fpsc = self.features_per_scale
        for i in range(self.n_scales):
            lo = i * fpsc
            hi = (i + 1) * fpsc
            features[..., lo:hi] *= gates[i]
            features[..., lo + self.n_scales * fpsc : hi + self.n_scales * fpsc] *= gates[i]

        return features


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    block = MultiScaleFourierFeatureBlock(input_dim=1, n_scales=4,
                                           base_sigma=1.0, scale_factor=3.0,
                                           features_per_scale=16)
    N = 200
    coords = torch.linspace(-1, 1, N).reshape(1, N, 1)
    feats = block(coords=coords)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].imshow(feats[0].detach().numpy().T, aspect="auto",
                    extent=[-1, 1, 0, feats.shape[-1]])
    axes[0].set_title(f"Fourier features ({feats.shape[-1]})")
    axes[0].set_xlabel("x")
    axes[1].bar(range(4), block.gates.detach().numpy())
    axes[1].set_title("Scale gates")
    axes[1].set_xlabel("Scale")
    plt.tight_layout()
    plt.savefig("demo_multiscale_ff.png", dpi=100)
    plt.close()
    print("Saved demo_multiscale_ff.png")
