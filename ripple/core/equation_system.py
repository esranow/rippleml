"""
ripple.core.equation_system — Manages multiple coupled equations.
"""
from __future__ import annotations
import torch
from typing import Any, Dict, List, Optional
from ripple.physics.equation import Equation

class EquationSystem:
    """
    Groups multiple Equation objects.
    
    equations: List of Equation instances.
    weights: Optional list of weights for each equation's loss contribution.
    """
    def __init__(self, equations: List[Equation], weights: Optional[List[float]] = None):
        self.equations = equations
        self.weights = weights if weights is not None else [1.0] * len(equations)
        
    def compute_residuals(self, fields: Dict[str, torch.Tensor], coords: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes residual for each equation.
        
        fields: Dict of field tensors {name: tensor}.
        coords: Coordinate tensor used to compute fields.
        """
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
            
        residuals = []
        params = {"inputs": coords}
        
        for eq in self.equations:
            # Each equation acts on a specific field determined by its operators
            # But Equation.residual expects a single 'field' argument.
            # In multi-field, the operators handle the field lookup?
            # Wait, looking at Equation.residual:
            # out = out + coeff * op.compute(field, params)
            # This 'field' is passed from the top.
            
            # If Equation was designed for single field, we need to adapt it.
            # However, for Phase 3, we can assume each Equation corresponds to one residual term.
            # But which field should we pass to Equation.residual?
            
            # Let's look at the operators. Each operator has a 'field' property.
            # So Equation.residual should probably be updated to take the fields dict
            # OR EquationSystem handles the routing.
            
            # Actually, if I update Equation.residual to ignore its 'field' argument 
            # and let operators pull from params["fields"], it would be cleaner.
            
            # But the prompt says EquationSystem.compute_residuals(fields, coords).
            # I will pass 'fields' in params.
            params["fields"] = fields
            
            # We still need to pass A field to Equation.residual because it's required.
            # We'll pass the first field or a dummy, but operators will use params["fields"].
            dummy_field = next(iter(fields.values()))
            residuals.append(eq.residual(dummy_field, params))
            
        return residuals

    def compute_loss(self, fields: Dict[str, torch.Tensor], coords: torch.Tensor) -> torch.Tensor:
        """Weighted MSE sum of all residuals."""
        residuals = self.compute_residuals(fields, coords)
        loss = torch.tensor(0.0, device=coords.device)
        for res, weight in zip(residuals, self.weights):
            loss = loss + weight * torch.mean(res**2)
        return loss
