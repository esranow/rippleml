import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Dict
from ripple.physics.operators import Gradient

class BoundaryCondition:
    """
    Base class for Boundary Conditions.
    """
    def __init__(self, apply_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            apply_fn: Function mapping (u, x) -> loss_term
        """
        self.apply_fn = apply_fn

    def __call__(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.apply_fn(u, x)

class DirichletBC(BoundaryCondition):
    """
    Dirichlet Boundary Condition: u(x) = g(x) on boundary.
    Loss = MSE(u(x) - g(x))
    """
    def __init__(self, value_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Args:
            value_fn (Callable): Function g(x) returning target values.
        """
        def apply(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            target = value_fn(x)
            return torch.mean((u - target) ** 2)
            
        super().__init__(apply)

class NeumannBC(BoundaryCondition):
    """
    Neumann Boundary Condition: du/dn(x) = h(x) on boundary.
    Assumes normal derivative or just derivative for 1D.
    For high-dim, assumes simplified normal or requires normal vector input.
    Currently implements simple derivative check magnitude or user-provided logic.
    Refined implementation: uses gradient w.r.t input.
    """
    def __init__(self, derivative_fn: Callable[[torch.Tensor], torch.Tensor], normal_idx: int = 0):
        """
        Args:
            derivative_fn (Callable): Function h(x) returning target derivative values.
            normal_idx (int): dimension index along which to check derivative (simplification).
        """
        _grad_op = Gradient()

        def apply(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            d_u = _grad_op.compute(u, {"inputs": x})[..., normal_idx:normal_idx+1]
            target = derivative_fn(x)
            return torch.mean((d_u - target) ** 2)

        super().__init__(apply)

class PeriodicBC(BoundaryCondition):
    """
    Periodic Boundary Condition: u(x_left) = u(x_right).
    Pass in pairs of points or assume inputs are concatenated [x_left, x_right].
    """
    def __init__(self):
        """
        Simple periodic enforcing u(x1) approx u(x2).
        Likely used by splitting batch into two halves corresponding to boundaries.
        """
        def apply(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Assume u contains [u_left; u_right] stacked in batch dim
            # This is a strong assumption on data loader, but standard for these classes 
            # unless we take two distinct tensors.
            # Let's assume the user handles the batch preparation where
            # half the batch is left, half is right.
            batch_size = u.shape[0]
            if batch_size % 2 != 0:
                # Fallback or error
                return torch.tensor(0.0, device=u.device)
                
            mid = batch_size // 2
            u_left = u[:mid]
            u_right = u[mid:]
            return torch.mean((u_left - u_right) ** 2)
            
        super().__init__(apply)
