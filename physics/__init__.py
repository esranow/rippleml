"""
ripple: Physics Package
Defines PDE specifications, residuals, and boundary conditions.
"""
from ripple.physics.pde import PDESpec
from ripple.physics.residuals import build_residual_fn
from ripple.physics.boundary import BoundaryCondition, DirichletBC, NeumannBC, PeriodicBC

__all__ = [
    "PDESpec",
    "build_residual_fn",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "PeriodicBC",
]
