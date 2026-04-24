import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Callable, Union, Tuple
from ripple.operators.grid_utils import flatten_grid

class OperatorTrainer:
    """
    Trainer specific for Operator Learning tasks (mapping function to function).
    Focuses on Data Loss (MSE/L2) between prediction and target.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Args:
            model (nn.Module): The operator model (e.g. FNO).
            optimizer (optim.Optimizer): Optimizer.
            loss_fn (Callable): Loss function. Defaults to relative L2 or MSE.
            device (str): Device.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model.to(self.device)
        
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Perform a single optimization step.
        
        Args:
            batch: Tuple (inputs, targets).
                   Inputs: (B, ...) spatial grid
                   Targets: (B, ...) spatial grid
                   
        Returns:
            float: Loss value.
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Ensure data is flattened if model requires (B, N, C)
        # Our FNO implementation expects (B, N, C).
        # We assume dataset yields (B, ..., C).
        if inputs.dim() > 3:
            inputs = flatten_grid(inputs)
        if targets.dim() > 3:
            targets = flatten_grid(targets)
            
        self.model.train()
        self.optimizer.zero_grad()
        
        preds = self.model(inputs)
        
        loss = self.loss_fn(preds, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
