import torch
import math
from typing import Optional, Union

def generate_sine_wave(
    grid: torch.Tensor, 
    frequency: float = 1.0, 
    phase: float = 0.0, 
    amplitude: float = 1.0
) -> torch.Tensor:
    """
    Generate a sine wave on the given grid.
    
    Args:
        grid (torch.Tensor): Spatial grid points. Shape (B, N, D) or (N, D).
        frequency (float): Frequency of the wave.
        phase (float): Phase shift.
        amplitude (float): Amplitude of the wave.
        
    Returns:
        torch.Tensor: Sine wave values. Shape matches grid shape excluding last dim implies (B, N, 1) usually.
                      Assuming we want scalar field u(x), so (..., 1).
    """
    # Assuming grid is (..., D), we compute sin(k * x + phi)
    # For D>1, this might be a plane wave or product.
    # Let's assume isotropic or based on first dim for simplicity if D>1, 
    # or sum of coords.
    # Standard: k dot x. Let's assume frequency applies to sum of coords or x[0].
    
    # We'll use sum of coordinates for ND case to create a diagonal wave 
    # or just use first coordinate if that makes more sense for "sine wave".
    # Given "1D and small 2D", let's sum coordinates to get a wave-like front.
    
    arg = torch.sum(grid, dim=-1, keepdim=True)
    return amplitude * torch.sin(2 * math.pi * frequency * arg + phase)

def generate_gaussian_bump(
    grid: torch.Tensor, 
    center: Union[float, torch.Tensor] = 0.0, 
    width: float = 0.1, 
    amplitude: float = 1.0
) -> torch.Tensor:
    """
    Generate a Gaussian bump.
    
    Args:
        grid (torch.Tensor): Spatial grid (..., D).
        center (float | torch.Tensor): Center position.
        width (float): Gaussian sigma/width.
        amplitude (float): Peak amplitude.
        
    Returns:
        torch.Tensor: Gaussian values (..., 1).
    """
    # squared distance
    diff = grid - center
    # Sum over spatial dims
    sq_dist = torch.sum(diff ** 2, dim=-1, keepdim=True)
    return amplitude * torch.exp(-0.5 * sq_dist / (width ** 2))
