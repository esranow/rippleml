"""
ripple.physics.operators — Operator base + concrete implementations.
Reuses autograd patterns from ripple.physics.residuals.
"""
from __future__ import annotations
import torch
from typing import Any, Dict


class Operator:
    """Abstract base: compute(field, params) -> tensor."""

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete operators
# ---------------------------------------------------------------------------

class Laplacian(Operator):
    """
    Autograd Laplacian: sum of d²u/dx_i² over spatial dims.
    inputs convention: (..., D) where last dim = time → spatial dims = [:-1].
    """

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        inputs: torch.Tensor = params["inputs"]  # requires_grad=True
        spatial_dim = inputs.shape[-1] - 1  # exclude time

        # First-order spatial grads
        grads = torch.autograd.grad(
            field, inputs,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True,
        )[0]  # (..., D)

        laplacian = torch.zeros_like(field)
        for i in range(spatial_dim):
            gi = grads[..., i : i + 1]
            ggi = torch.autograd.grad(
                gi, inputs,
                grad_outputs=torch.ones_like(gi),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if ggi is not None:
                laplacian = laplacian + ggi[..., i : i + 1]
        return laplacian


class Gradient(Operator):
    """Full gradient ∂u/∂x_i for each spatial dim."""

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        inputs: torch.Tensor = params["inputs"]
        grads = torch.autograd.grad(
            field, inputs,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grads[..., :-1]  # spatial only


class TimeDerivative(Operator):
    """
    nth-order time derivative.  order=1 → u_t, order=2 → u_tt.
    inputs: (..., D), last dim is time.
    """

    def __init__(self, order: int = 1):
        assert order in (1, 2), "Only order 1 or 2 supported."
        self.order = order

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        inputs: torch.Tensor = params["inputs"]
        u = field

        for _ in range(self.order):
            g = torch.autograd.grad(
                u.sum(), inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            u = g[..., -1:] if g is not None else torch.zeros_like(field)
        return u


class Diffusion(Operator):
    """Diffusion operator: alpha * Laplacian(u)"""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.laplacian = Laplacian()

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return self.alpha * self.laplacian.compute(field, params)

class Advection(Operator):
    """Advection operator: v * Gradient(u)"""
    def __init__(self, v: float):
        self.v = v
        self.gradient = Gradient()

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        grad = self.gradient.compute(field, params)
        # assuming 1D for v
        return self.v * grad[..., 0:1]

class Source(Operator):
    """Source term: fn(field, params)"""
    def __init__(self, fn):
        self.fn = fn

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return self.fn(field, params)

class Nonlinear(Operator):
    """Nonlinear operator: fn(field)"""
    def __init__(self, fn):
        self.fn = fn

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return self.fn(field, params)

