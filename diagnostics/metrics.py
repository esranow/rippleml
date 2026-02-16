import torch
import math

def l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute L2 error (Euclidean distance) between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor.
        target (torch.Tensor): Target tensor.
        
    Returns:
        float: L2 error.
    """
    return torch.norm(pred - target).item()

def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Compute Relative L2 error: ||pred - target|| / ||target||.
    
    Args:
        pred (torch.Tensor): Prediction.
        target (torch.Tensor): Target.
        epsilon (float): Small value to avoid division by zero.
        
    Returns:
        float: Relative L2 error.
    """
    diff_norm = torch.norm(pred - target)
    target_norm = torch.norm(target)
    return (diff_norm / (target_norm + epsilon)).item()
