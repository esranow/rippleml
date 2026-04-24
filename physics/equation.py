"""
ripple.physics.equation — Generic PDE residual from a list of (coeff, Operator) terms.
No PDEs are hardcoded; callers compose them.
"""
from __future__ import annotations
import torch
from typing import Any, Dict, List, Tuple

from ripple.physics.operators import Operator


class Equation:
    """
    Residual = sum_i( coeff_i * operator_i.compute(field, params) ) - forcing

    terms: list of (coefficient: float, operator: Operator)
    forcing: optional callable(params) -> tensor
    """

    def __init__(
        self,
        terms: List[Tuple[float, Operator]],
        forcing=None,
    ):
        self.terms = terms
        self.forcing = forcing  # callable(params) -> tensor | None

    def residual(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Returns residual tensor (same shape as field)."""
        out = torch.zeros_like(field)
        for coeff, op in self.terms:
            out = out + coeff * op.compute(field, params)
        if self.forcing is not None:
            out = out - self.forcing(params)
        return out

    def compute_residual(self, u: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Single entry-point for Experiment and all callers.

        Args:
            u: Model output computed from inputs (autograd graph connected).
            inputs: cat([x, t], dim=-1) with requires_grad=True — the SAME
                    tensor the model used so operators can differentiate u w.r.t. it.
        """
        if not inputs.requires_grad:
            inputs = inputs.requires_grad_(True)
        return self.residual(u, {"inputs": inputs})
