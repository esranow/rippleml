"""
rippl.core.equation_system — Manages multiple coupled equations.
"""
from __future__ import annotations
import torch
from typing import Any, Dict, List, Optional
from rippl.core.equation import Equation

class EquationSystem:
    """
    Groups multiple Equation objects.
    
    equations: List of Equation instances.
    weights: Optional list of weights for each equation's loss contribution.
    """
    def __init__(self, equations: List[Equation], weights: Optional[List[float]] = None):
        self.equations = equations
        self.weights = weights if weights is not None else [1.0] * len(equations)
        
    def compute_residuals(self, fields: Dict[str, torch.Tensor], coords: torch.Tensor, spatial_dims: int = None) -> List[torch.Tensor]: # fields: {name: (N, 1)}, coords: (N, D)
        """Compute residuals for all equations in the system simultaneously."""
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
            
        # 1. Collect all required derivatives from ALL equations
        all_requests = []
        for eq in self.equations:
            for coeff, op in eq.terms:
                sig = op.signature()
                all_requests.extend(sig.get("requires_derived", []))
        
        # 2. Precompute derivatives once
        derived = {}
        if all_requests:
            from rippl.physics.derivatives import compute_all_derivatives
            derived = compute_all_derivatives(fields, coords, list(set(all_requests)))
            
        residuals = []
        params = {"inputs": coords, "fields": fields, "derived": derived}
        if spatial_dims is not None:
            params["spatial_dims"] = spatial_dims
        
        for eq in self.equations:
            dummy_field = next(iter(fields.values()))
            residuals.append(eq.residual(dummy_field, params))
            
        return residuals

    def compute_loss(self, fields: Dict[str, torch.Tensor], coords: torch.Tensor, spatial_dims: int = None) -> torch.Tensor: # fields: {name: (N, 1)}, coords: (N, D)
        """Calculate the scalar weighted MSE loss across all system residuals."""
        residuals = self.compute_residuals(fields, coords, spatial_dims=spatial_dims)
        loss = torch.tensor(0.0, device=coords.device)
        for res, weight in zip(residuals, self.weights):
            loss = loss + weight * torch.mean(res**2)
        return loss
