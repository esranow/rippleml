"""
rippl.core.system — System = Equation + Domain + Constraints.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable

from rippl.core.equation import Equation
from rippl.core.exceptions import RipplValidationError


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

    def generate_loader(self, batch_size: int = 2048):
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        # Sobol sampling over domain bounds including time if present
        sobol = torch.quasirandom.SobolEngine(len(self.bounds), scramble=True)
        n_points = max(batch_size * 10, 50000)
        pts = sobol.draw(n_points)
        # Scale to actual bounds
        for i, (lo, hi) in enumerate(self.bounds):
            pts[:, i] = pts[:, i] * (hi - lo) + lo
        dataset = TensorDataset(pts)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@dataclass
class Constraint:
    """Explicit constraint: type + field + coords + value."""
    type: str               # "dirichlet", "neumann", "initial"
    field: str              # field name, e.g. "u"
    coords: torch.Tensor    # (N, D) coordinates
    value: Union[Callable, torch.Tensor]  # target value or callable(coords)


@dataclass
class NeumannConstraint:
    field: str
    coords: torch.Tensor
    normal_direction: int
    value: Union[Callable, torch.Tensor]

class MovingBoundaryConstraint(Constraint):
    """
    Constraint that re-evaluates boundary location each epoch.
    """
    def __init__(self, field: str, boundary_fn: Callable[[int, Optional[torch.nn.Module]], torch.Tensor],
                 value: Union[Callable, torch.Tensor], type: str = "dirichlet"):
        # Initialize with dummy coords, will be updated in first epoch
        super().__init__(type=type, field=field, coords=torch.empty(0), value=value)
        self.boundary_fn = boundary_fn

    def update(self, epoch: int, model: Optional[torch.nn.Module] = None):
        """Re-evaluate boundary coordinates."""
        self.coords = self.boundary_fn(epoch, model)


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
        constraints: Optional[List[Union[Constraint, NeumannConstraint]]] = None,
        fields: Optional[List[str]] = None,
        particular_solution: Optional[Callable] = None,
        scales: Optional['ReferenceScales'] = None
    ):
        self.equation = equation
        self.domain = domain
        self.constraints: List[Union[Constraint, NeumannConstraint]] = constraints or []
        self.fields = fields or ["u"]
        self.particular_solution = particular_solution
        self.scales = scales

    def validate_fields(self, field_dict: Dict[str, torch.Tensor]) -> None:
        """Verify that the provided field tensors match the system specification."""
        for name, tensor in field_dict.items():
            if name not in self.fields:
                raise RipplValidationError(f"Field '{name}' not defined in system.")
            
            if tensor.shape[-1] != 1:
                 raise RipplValidationError(f"Field '{name}' must have trailing dimension 1, got {tensor.shape}")

    def validate(self) -> bool:
        """Perform a full integrity check on the system components."""
        if self.equation is None:
            raise RipplValidationError("Equation must be set.")
        
        # 1. Operator fields exist
        from rippl.physics.operators import Operator
        
        # Helper to extract operators from Equation or EquationSystem
        equations = []
        from rippl.core.equation_system import EquationSystem
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
                        raise RipplValidationError(f"Operator {op.__class__.__name__} requires field '{f}', but it's not in System.fields.")
                
                # Check output (if it refers to a field name directly)
                # Usually output is 'field' or 'op(field)'. 
                # If output is exactly a field name, it must be in self.fields.
                if "(" not in sig["output"] and sig["output"] not in self.fields:
                    raise RipplValidationError(f"Operator {op.__class__.__name__} output field '{sig['output']}' not in System.fields.")

                # Feature 1: Large coefficient warning
                if self.scales is None:
                    import warnings
                    from rippl.core.exceptions import PhysicsModelWarning
                    if abs(coeff) > 1e3 or (abs(coeff) < 1e-3 and abs(coeff) > 0):
                        warnings.warn(
                            f"Large coefficient ({coeff}) detected. Consider setting System(scales=ReferenceScales(...)) "
                            "to avoid gradient starvation.",
                            PhysicsModelWarning
                        )

        # 2. Domain bounds match spatial_dims
        # Bounds should match spatial_dims exactly (time is not a spatial dimension in Domain).
        if len(self.domain.bounds) != self.domain.spatial_dims:
            raise RipplValidationError(f"Domain bounds length {len(self.domain.bounds)} does not match spatial_dims {self.domain.spatial_dims}")

        # 3. Constraint field names exist
        for c in self.constraints:
            if c.field not in self.fields:
                raise RipplValidationError(f"Constraint references unknown field '{c.field}'.")

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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'System':
        """
        Create a System instance from a configuration dictionary.

        Args:
            config: Configuration dictionary containing geometry, physics, and model sections.

        Returns:
            An initialized System instance.
        """
        # 1. Geometry (Domain)
        geom = config["geometry"]
        domain = Domain(
            spatial_dims=geom["spatial_dims"],
            bounds=tuple(tuple(b) for b in geom["bounds"]),
            resolution=tuple(geom["resolution"])
        )
        
        # 2. Physics (Equation)
        phys = config["physics"]
        fields = phys.get("fields", ["u"])
        
        from rippl.core.equation import Equation
        from rippl.core.equation_system import EquationSystem
        from rippl.core.config import get_operator_class
        
        eq_data = phys["equation"]
        if isinstance(eq_data, list):
            # Single equation
            terms = []
            for item in eq_data:
                coeff, op_name = item[0], item[1]
                op_config = item[2] if len(item) > 2 else {}
                op_cls = get_operator_class(op_name)
                terms.append((coeff, op_cls(**op_config)))
            equation = Equation(terms)
        else:
            # Multi-field equation system
            eqs = []
            for field_name, terms_data in eq_data.items():
                terms = []
                for item in terms_data:
                    coeff, op_name = item[0], item[1]
                    op_config = item[2] if len(item) > 2 else {}
                    op_cls = get_operator_class(op_name)
                    terms.append((coeff, op_cls(**op_config)))
                eqs.append(Equation(terms))
            equation = EquationSystem(eqs)
            
        # 3. Constraints
        constraints = []
        for c_data in phys.get("constraints", []):
            coords = torch.tensor(c_data["coords"], dtype=torch.float32)
            val = c_data["value"]
            if isinstance(val, (list, float, int)):
                val = torch.tensor(val, dtype=torch.float32)
            constraints.append(Constraint(
                type=c_data["type"],
                field=c_data["field"],
                coords=coords,
                value=val
            ))
            
        return cls(equation=equation, domain=domain, constraints=constraints, fields=fields)

    def to_config(self) -> Dict[str, Any]:
        """
        Serialize the System state to a configuration dictionary.

        Returns:
            A dictionary representation of the system.
        """
        return {
            "geometry": {
                "spatial_dims": self.domain.spatial_dims,
                "bounds": [list(b) for b in self.domain.bounds],
                "resolution": list(self.domain.resolution)
            },
            "physics": {
                "fields": self.fields,
                "equation": self._serialize_equation(),
                "constraints": [self._serialize_constraint(c) for c in self.constraints]
            }
        }

    def _serialize_equation(self) -> Any:
        from rippl.core.equation import Equation
        from rippl.core.equation_system import EquationSystem
        
        def _ser_eq(eq):
            terms = []
            for coeff, op in eq.terms:
                op_name = op.__class__.__name__.lower()
                # Try to extract config from operator attributes
                op_config = {}
                for k, v in op.__dict__.items():
                    if k not in ["field", "spatial_dims"] and not k.startswith("_"):
                        if isinstance(v, (int, float, str, bool)):
                            op_config[k] = v
                terms.append([coeff, op_name, op_config])
            return terms

        if isinstance(self.equation, EquationSystem):
            return {f: _ser_eq(eq) for f, eq in zip(self.fields, self.equation.equations)}
        return _ser_eq(self.equation)

    def _serialize_constraint(self, constraint: Any) -> Dict[str, Any]:
        res = {
            "type": constraint.type,
            "field": constraint.field,
            "coords": constraint.coords.tolist()
        }
        if isinstance(constraint.value, torch.Tensor):
            res["value"] = constraint.value.tolist()
        elif not callable(constraint.value):
            res["value"] = constraint.value
        else:
            res["value"] = "callable" # Placeholder
        return res
