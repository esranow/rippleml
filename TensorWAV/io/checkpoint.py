import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

def save_checkpoint(
    model: nn.Module, 
    optimizer: Optional[optim.Optimizer], 
    epoch: int, 
    path: Union[str, Path],
    extra_data: Dict[str, Any] = {}
) -> None:
    """
    Save model and optimizer state to a file.
    
    Args:
        model (nn.Module): The model to save.
        optimizer (Optional[optim.Optimizer]): The optimizer to save.
        epoch (int): Current epoch.
        path (Union[str, Path]): Path to save the checkpoint.
        extra_data (Dict[str, Any]): Additional data to save.
    """
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        **extra_data
    }
    
    if optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
        
    torch.save(state, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(
    model: nn.Module, 
    optimizer: Optional[optim.Optimizer], 
    path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load model and optimizer state from a file.
    
    Args:
        model (nn.Module): Model to load weights into.
        optimizer (Optional[optim.Optimizer]): Optimizer to load state into.
        path (Union[str, Path]): Path to the checkpoint file.
        
    Returns:
        Dict[str, Any]: The full checkpoint dictionary (including epoch, etc).
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location="cpu") # Load to CPU first to avoid CUDA errors if on CPU machine
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    logging.info(f"Loaded checkpoint from {path} (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    return checkpoint
