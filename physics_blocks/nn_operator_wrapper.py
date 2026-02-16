"""
OperatorWrapperBlock — Wraps a grid-based operator model and provides a
point-evaluation adaptor.

Converts between point-cloud ``(B, N, D)`` → grid ``(B, C, *spatial)`` and
back, enabling grid-native models (FNO, U-Net, etc.) to be used with
collocation-style PINN interfaces.
"""

import logging
from typing import Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class OperatorWrapperBlock(nn.Module):
    """
    Wraps a grid-based operator model with point-evaluation adaptor.

    Args:
        grid_model: An ``nn.Module`` that expects ``(B, C, *spatial)`` input.
        grid_shape: Tuple of spatial grid dimensions, e.g. ``(64,)`` or ``(32, 32)``.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        grid_model: Optional[nn.Module] = None,
        grid_shape: tuple = (64,),
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs: Any,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = len(grid_shape)

        # Default grid model: simple conv
        if grid_model is None:
            if self.spatial_dim == 1:
                self.grid_model = nn.Sequential(
                    nn.Conv1d(in_channels, 16, 3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(16, out_channels, 3, padding=1),
                )
            elif self.spatial_dim == 2:
                self.grid_model = nn.Sequential(
                    nn.Conv2d(in_channels, 16, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(16, out_channels, 3, padding=1),
                )
            else:
                self.grid_model = nn.Sequential(
                    nn.Conv3d(in_channels, 16, 3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(16, out_channels, 3, padding=1),
                )
        else:
            self.grid_model = grid_model

    def points_to_grid(
        self, u: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Scatter point values onto a regular grid via nearest-neighbor binning.

        Args:
            u: ``(B, N, C)`` point values.
            coords: ``(B, N, D)`` normalised [0,1] coordinates.

        Returns:
            ``(B, C, *grid_shape)`` grid tensor.
        """
        B, N, C = u.shape
        grid = torch.zeros(B, C, *self.grid_shape, device=u.device)
        counts = torch.zeros(B, 1, *self.grid_shape, device=u.device)

        # Bin coordinates
        indices = []
        for d in range(self.spatial_dim):
            idx = (coords[..., d].clamp(0, 1) * (self.grid_shape[d] - 1)).long()
            indices.append(idx)

        for b in range(B):
            if self.spatial_dim == 1:
                for n in range(N):
                    i = indices[0][b, n]
                    grid[b, :, i] += u[b, n]
                    counts[b, 0, i] += 1
            elif self.spatial_dim == 2:
                for n in range(N):
                    i, j = indices[0][b, n], indices[1][b, n]
                    grid[b, :, i, j] += u[b, n]
                    counts[b, 0, i, j] += 1

        counts = counts.clamp(min=1)
        return grid / counts

    def grid_to_points(
        self, grid: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample grid at point locations via bilinear interpolation.

        Args:
            grid: ``(B, C, *grid_shape)``
            coords: ``(B, N, D)`` normalised [0,1].

        Returns:
            ``(B, N, C)``
        """
        B, N, D = coords.shape
        # F.grid_sample expects coords in [-1, 1]
        grid_coords = 2.0 * coords - 1.0  # [0,1] → [-1,1]

        if self.spatial_dim == 1:
            # grid_sample needs 4D input; treat as (B, C, 1, N_grid)
            grid_4d = grid.unsqueeze(2)  # (B, C, 1, G)
            sample_pts = grid_coords.unsqueeze(2)  # (B, N, 1, 1) → need (B,1,N,2)
            # Create sample grid as (B, 1, N, 2) with y=0
            sample = torch.zeros(B, 1, N, 2, device=coords.device)
            sample[..., 0] = grid_coords[..., 0].unsqueeze(1)
            out = F.grid_sample(grid_4d, sample, align_corners=True, mode="bilinear")
            return out.squeeze(2).permute(0, 2, 1)  # (B, N, C)
        elif self.spatial_dim == 2:
            sample = grid_coords.unsqueeze(1)  # (B, 1, N, 2)
            out = F.grid_sample(grid, sample, align_corners=True, mode="bilinear")
            return out.squeeze(2).permute(0, 2, 1)  # (B, N, C)
        else:
            # 3D grid_sample
            sample = grid_coords.unsqueeze(1).unsqueeze(1)  # (B,1,1,N,3)
            out = F.grid_sample(grid, sample, align_corners=True, mode="bilinear")
            return out.squeeze(2).squeeze(2).permute(0, 2, 1)

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Point-cloud in → grid model → point-cloud out.

        Args:
            u: ``(B, N, C_in)`` point values.
            coords: ``(B, N, D)`` normalised [0,1] coordinates.

        Returns:
            ``(B, N, C_out)``
        """
        if coords is None:
            raise ValueError("coords required for point-to-grid conversion")

        grid_in = self.points_to_grid(u, coords)
        grid_out = self.grid_model(grid_in)
        return self.grid_to_points(grid_out, coords)


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    import math

    block = OperatorWrapperBlock(grid_shape=(64,), in_channels=1, out_channels=1)

    B, N = 1, 50
    coords = torch.rand(B, N, 1).sort(dim=1).values
    u = torch.sin(2 * math.pi * coords)

    with torch.no_grad():
        out = block(u, coords)

    fig, ax = plt.subplots(figsize=(6, 3))
    xv = coords.squeeze().numpy()
    ax.scatter(xv, u.squeeze().numpy(), s=10, label="Input points")
    ax.scatter(xv, out.squeeze().numpy(), s=10, marker="x", label="Output points")
    ax.set_title("OperatorWrapperBlock")
    ax.legend()
    plt.tight_layout()
    plt.savefig("demo_operator_wrapper.png", dpi=100)
    plt.close()
    print("Saved demo_operator_wrapper.png")
