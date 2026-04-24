"""
ripple.core.system — System = Equation + Domain + Constraints.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ripple.physics.equation import Equation


@dataclass
class Domain:
    """Axis-aligned domain specification."""
    spatial_dims: int          # 1 or 2
    x_range: tuple = (0.0, 1.0)
    t_range: tuple = (0.0, 1.0)
    y_range: Optional[tuple] = None  # only for 2-D
    bounds: Optional[tuple] = None
    resolution: Optional[int] = None


class Constraint:
    def __init__(self, fn, weight=1.0, type: str = "custom"):
        self.fn = fn
        self.weight = weight
        self.type = type


class System:
    """
    Top-level container: Equation + Domain + Constraints.

    Usage
    -----
    sys = System(equation=eq, domain=dom, constraints=[bc])
    sys.validate()
    sys.summary()
    """

    def __init__(
        self,
        equation: Equation,
        domain: Domain,
        constraints: Optional[List[Constraint]] = None,
    ):
        self.equation = equation
        self.domain = domain
        self.constraints: List[Constraint] = constraints or []

    def validate(self):
        """Robust validation of system components and dimensions."""
        assert self.equation is not None, "equation must be set"
        assert hasattr(self.domain, "spatial_dims"), "domain must have spatial_dims"
        assert len(self.equation.terms) > 0, "equation has no terms"

        from ripple.physics.operators import Operator, TimeDerivative
        has_t1 = False
        has_t2 = False
        for term in self.equation.terms:
            assert isinstance(term, tuple) and len(term) == 2, "term must be (coeff, operator)"
            assert isinstance(term[1], Operator), "operator must be an Operator instance"
            if isinstance(term[1], TimeDerivative):
                if term[1].order == 1: has_t1 = True
                if term[1].order == 2: has_t2 = True

        c_types = [c.type for c in self.constraints]
        for c in self.constraints:
            assert callable(c.fn), "constraint fn must be callable"
            assert c.type in ("boundary", "initial", "custom"), f"invalid constraint type: {c.type}"

        if has_t2 and self.constraints:
            assert "initial" in c_types, "TimeDerivative(order=2) requires at least one 'initial' constraint"
        if has_t1 and self.constraints:
            assert "initial" in c_types or "boundary" in c_types, "TimeDerivative(order=1) requires initial OR boundary constraint"

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
        print(f"  Equation terms : {len(self.equation.terms)}")
        print(f"  Domain         : {self.domain.spatial_dims}D  "
              f"x in {self.domain.x_range}  t in {self.domain.t_range}")
        print(f"  Constraints    : {[c.name for c in self.constraints]}")
