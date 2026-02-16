"""
PDEParameterEmbeddingBlock — Embed PDE parameters (c, damping γ, α, …) into
a latent vector and provide FiLM-style modulation.

Learnable:
  - Embedding MLP: maps parameter vector → latent.
  - FiLM head: produces per-feature (γ, β) for affine modulation of hidden
    features.

APIs:
  - forward(params) → latent embedding
  - modulate(features, params) → γ * features + β  (FiLM)
"""

import logging
from typing import Optional, Any, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class PDEParameterEmbeddingBlock(nn.Module):
    """
    Embeds a vector of PDE parameters and can modulate hidden features
    via FiLM (Feature-wise Linear Modulation).

    Args:
        param_dim: Dimension of the raw parameter vector.
        embed_dim: Dimension of the latent embedding.
        feature_dim: Dimension of the features to modulate (for FiLM head).
        hidden: Hidden size of embedding MLP.
    """

    def __init__(
        self,
        param_dim: int = 3,
        embed_dim: int = 16,
        feature_dim: int = 32,
        hidden: int = 32,
        **kwargs: Any,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim

        # Embedding MLP
        self.embed_net = nn.Sequential(
            nn.Linear(param_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

        # FiLM head: from embedding → (gamma, beta) for feature_dim
        self.film_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim * 2),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        u: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed PDE parameters.

        Args:
            u: Unused (API compatibility).
            coords: Unused.
            params: ``(B, param_dim)`` parameter vector.

        Returns:
            Latent embedding ``(B, embed_dim)``.
        """
        if params is None:
            raise ValueError("params required for PDEParameterEmbeddingBlock")
        return self.embed_net(params)

    # ------------------------------------------------------------------ #
    def modulate(
        self,
        features: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        FiLM modulation: ``γ * features + β``.

        Args:
            features: ``(B, *, feature_dim)`` hidden features.
            params: ``(B, param_dim)`` PDE parameters.

        Returns:
            Modulated features, same shape as *features*.
        """
        embedding = self.embed_net(params)  # (B, embed_dim)
        film = self.film_head(embedding)    # (B, feature_dim * 2)
        gamma, beta = film.chunk(2, dim=-1)  # each (B, feature_dim)

        # Expand gamma/beta for arbitrary middle dims
        while gamma.dim() < features.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * features + beta


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    block = PDEParameterEmbeddingBlock(param_dim=3, embed_dim=16, feature_dim=32)

    B = 4
    params = torch.randn(B, 3)
    embedding = block(params=params)

    features = torch.randn(B, 10, 32)
    modulated = block.modulate(features, params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].imshow(embedding.detach().numpy(), aspect="auto")
    axes[0].set_title(f"Embedding ({B}×16)")
    axes[0].set_ylabel("Batch")
    axes[1].plot(features[0, 0].detach().numpy(), label="Original")
    axes[1].plot(modulated[0, 0].detach().numpy(), "--", label="FiLM modulated")
    axes[1].set_title("FiLM modulation (sample)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("demo_embedding.png", dpi=100)
    plt.close()
    print("Saved demo_embedding.png")
