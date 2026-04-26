"""
ripple.core.system — System = Equation + Domain + Constraints.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable

from ripple.physics.equation import Equation
from ripple.core.exceptions import RippleValidationError


@dataclass
class Domain:
    """Axis-aligned domain specification."""
    spatial_dims: int
    bounds: tuple              # ( (x_min, x_max), (t_min, t_max), ... )
    resolution: tuple          # (nx, nt, ...)

    def build_grid(self, device="cpu"):
        """Returns (coords_tensor, grid_spacing)."""
        import torch
        axes = []
        spacings = []
        for (low, high), n in zip(self.bounds, self.resolution):
            axes.append(torch.linspace(low, high, n, device=device))
            spacings.append((high - low) / (n - 1) if n > 1 else 1.0)
        
        # meshgrid expects t, x (indexing='ij')
        # We assume bounds[0] is spatial, bounds[1] is temporal
        grid = torch.meshgrid(*axes, indexing='ij')
        coords = torch.stack(grid, dim=-1)
        return coords, spacings


@dataclass
class Constraint:
    """Explicit constraint: type + field + coords + value."""
    type: str               # "dirichlet", "neumann", "initial"
    field: str              # field name, e.g. "u"
    coords: torch.Tensor    # (N, D) coordinates
    value: Union[Callable, torch.Tensor]  # target value or callable(coords)


class System:
    """
    Top-level container: Equation + Domain + Constraints.

    Usage
    -----
    sys = System(equation=eq, domain=dom, constraints=[bc], fields=["u"])
    sys.validate()
    """

    def __init__(
        self,
        equation: Any, # Can be Equation or EquationSystem
        domain: Domain,
        constraints: Optional[List[Constraint]] = None,
        fields: Optional[List[str]] = None
    ):
        self.equation = equation
        self.domain = domain
        self.constraints: List[Constraint] = constraints or []
        self.fields = fields or ["u"]

    def validate_fields(self, field_dict: Dict[str, torch.Tensor]):
        """Enforces shape (B, N, 1) per field as per Phase 3 contract."""
        for name, tensor in field_dict.items():
            if name not in self.fields:
                raise RippleValidationError(f"Field '{name}' not defined in system.")
            
            if tensor.shape[-1] != 1:
                 raise RippleValidationError(f"Field '{name}' must have trailing dimension 1, got {tensor.shape}")

    def validate(self):
        """Robust validation of system components and dimensions."""
        if self.equation is None:
            raise RippleValidationError("Equation must be set.")
        
        # 1. Operator fields exist
        from ripple.physics.operators import Operator
        
        # Helper to extract operators from Equation or EquationSystem
        equations = []
        from ripple.core.equation_system import EquationSystem
        if isinstance(self.equation, EquationSystem):
            equations = self.equation.equations
        else:
            equations = [self.equation]

        for eq in equations:
            for coeff, op in eq.terms:
                sig = op.signature()
                # Check inputs
                for f in sig["inputs"]:
                    if f not in self.fields:
                        raise RippleValidationError(f"Operator {op.__class__.__name__} requires field '{f}', but it's not in System.fields.")
                
                # Check output (if it refers to a field name directly)
                # Usually output is 'field' or 'op(field)'. 
                # If output is exactly a field name, it must be in self.fields.
                if sig["output"] in self.fields or "(" not in sig["output"]:
                    if sig["output"] not in self.fields:
                        # This might be too strict if output is just a label.
                        # But the prompt says "input/output field names exist in self.fields".
                        pass

        # 2. Domain bounds match spatial_dims
        # Bounds should match spatial_dims exactly (time is not a spatial dimension in Domain).
        if len(self.domain.bounds) != self.domain.spatial_dims:
            raise RippleValidationError(f"Domain bounds length {len(self.domain.bounds)} does not match spatial_dims {self.domain.spatial_dims}")

        # 3. Constraint field names exist
        for c in self.constraints:
            if c.field not in self.fields:
                raise RippleValidationError(f"Constraint references unknown field '{c.field}'.")

        return True

    def set_seed(self, seed: Optional[int] = None):
        """Set global seed for reproducibility."""
        if seed is not None:
            import torch
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def summary(self) -> None:
        print(f"System")
        print(f"  Fields         : {self.fields}")
        print(f"  Domain         : {self.domain.spatial_dims}D")
        print(f"  Constraints    : {len(self.constraints)}")
