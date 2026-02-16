import torch
from typing import Tuple, List, Union

def flatten_grid(x: torch.Tensor, spatial_dims: int = -1) -> torch.Tensor:
    """
    Flatten spatial dimensions of a tensor into a single dimension.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, d1, d2, ..., dn, C).
                          Or (B, C, d1, ..., dn) depending on convention.
                          NeuralWave core convention models mostly use (B, N, F).
                          But valid grid data might come as (B, H, W, F).
        spatial_dims (int): Number of spatial dimensions to collapse. 
                            If -1, collapses all dimensions between 0 (batch) and last (channel).
                            
    Returns:
        torch.Tensor: (B, N, C) where N = product of spatial dims.
    """
    # Assume input is (B, ..., C)
    # We want to keep B and C, flatten everything else into N.
    
    shape = x.shape
    if len(shape) < 3:
        # Already (B, N, C) or (B, N)
        return x
        
    B = shape[0]
    C = shape[-1]
    
    # Check if we align with (B, ..., C)
    # The prompt implies "supports dim=1,2,3".
    # Often FNO takes (B, x, y, c) -> (B, x*y, c)
    
    return x.view(B, -1, C)

def unflatten_grid(x: torch.Tensor, original_shape: Union[Tuple[int, ...], torch.Size]) -> torch.Tensor:
    """
    Restore spatial structure from flattened tensor.
    
    Args:
        x (torch.Tensor): (B, N, C)
        original_shape (Tuple): Original shape (B, d1, ..., dn, C)
        
    Returns:
        torch.Tensor: Reshaped tensor.
    """
    return x.view(original_shape)
