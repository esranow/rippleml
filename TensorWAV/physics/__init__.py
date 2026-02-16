"""
NeuralWave Core: Physics Package
Defines PDE specifications, residuals, and boundary conditions.
"""
from TensorWAV.physics.pde import PDESpec
from TensorWAV.physics.residuals import build_residual_fn
from TensorWAV.physics.boundary import BoundaryCondition, DirichletBC, NeumannBC, PeriodicBC

__all__ = [
    "PDESpec",
    "build_residual_fn",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "PeriodicBC",
]
