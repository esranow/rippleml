"""
rippl.core.equation — Generic PDE residual from a list of (coeff, Operator) terms.
"""
from __future__ import annotations
import torch
from typing import Any, Dict, List, Tuple

from rippl.physics.operators import Operator

class Equation:
    """
    Residual = sum_i( coeff_i * operator_i.compute(field, params) ) - forcing

    terms: list of (coefficient: float, operator: Operator)
    forcing: optional callable(params) -> tensor
    """

    def __init__(
        self,
        terms: List[Union[Operator, Tuple[float, Operator]]],
        forcing=None,
    ):
        # Standardize to List[Tuple[float, Operator]]
        self.terms = []
        for item in terms:
            if isinstance(item, tuple):
                self.terms.append(item)
            else:
                self.terms.append((1.0, item))
        self.forcing = forcing  # callable(params) -> tensor | None

    def residual(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor: # field: (N, 1)
        """Sum terms and subtract forcing to compute the pointwise residual."""
        out = torch.zeros_like(field)
        for coeff, op in self.terms:
            out = out + coeff * op.compute(field, params)
        if self.forcing is not None:
            out = out - self.forcing(params)
        return out

    def compute_residual(self, u: torch.Tensor, inputs: torch.Tensor, spatial_dims: int = None) -> torch.Tensor: # u: (N, 1), inputs: (N, D)
        """Orchestrate derivative precomputation and residual evaluation."""
        if not inputs.requires_grad:
            inputs = inputs.requires_grad_(True)
        
        # 1. Collect all required derivatives from operators
        all_requests = []
        for coeff, op in self.terms:
            sig = op.signature()
            all_requests.extend(sig.get("requires_derived", []))
        
        # 2. Precompute derivatives if any requested
        derived = {}
        if all_requests:
            from rippl.physics.derivatives import compute_all_derivatives
            # In simple Equation.compute_residual, fields is just {"u": u}
            fields = {"u": u}
            derived = compute_all_derivatives(fields, inputs, list(set(all_requests)))
            
        params = {"inputs": inputs, "derived": derived}
        if spatial_dims is not None:
            params["spatial_dims"] = spatial_dims
        return self.residual(u, params)
