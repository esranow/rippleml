"""
ripple.physics.operators — Operator base + concrete implementations.
Reuses autograd patterns from ripple.physics.residuals.
"""
from __future__ import annotations
import torch
from typing import Any, Dict, List


class Operator:
    """Abstract base: compute(field, params) -> tensor."""

    def __init__(self, field: str = "u"):
        self.field = field

    def _get_field(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        if "fields" in params and self.field in params["fields"]:
            return params["fields"][self.field]
        return field

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def signature(self) -> Dict[str, Any]:
        """Returns metadata about the operator."""
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 0,
            "type": "generic"
        }


# ---------------------------------------------------------------------------
# Concrete operators
# ---------------------------------------------------------------------------

class Laplacian(Operator):
    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        inputs: torch.Tensor = params["inputs"]
        spatial_dim = inputs.shape[-1] - 1

        grads = torch.autograd.grad(
            u, inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        laplacian = torch.zeros_like(u)
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

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"laplacian({self.field})",
            "order": 2, 
            "type": "spatial"
        })
        return sig


class Gradient(Operator):
    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        inputs: torch.Tensor = params["inputs"]
        grads = torch.autograd.grad(
            u, inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grads[..., :-1]

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"grad({self.field})",
            "order": 1, 
            "type": "spatial"
        })
        return sig


class Divergence(Operator):
    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        inputs: torch.Tensor = params["inputs"]
        spatial_dim = inputs.shape[-1] - 1
        
        div = torch.zeros(u.shape[:-1] + (1,), device=u.device)
        for i in range(spatial_dim):
            vi = u[..., i:i+1]
            gi = torch.autograd.grad(
                vi, inputs,
                grad_outputs=torch.ones_like(vi),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if gi is not None:
                div = div + gi[..., i:i+1]
        return div

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"div({self.field})",
            "order": 1, 
            "type": "spatial"
        })
        return sig


class TimeDerivative(Operator):
    def __init__(self, order: int = 1, field: str = "u"):
        super().__init__(field=field)
        self.order = order

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        inputs: torch.Tensor = params["inputs"]

        for _ in range(self.order):
            g = torch.autograd.grad(
                u.sum(), inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            u = g[..., -1:] if g is not None else torch.zeros_like(u)
        return u

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"dt^{self.order}({self.field})",
            "order": self.order,
            "type": "temporal"
        })
        return sig


class Diffusion(Operator):
    def __init__(self, alpha: float, field: str = "u"):
        super().__init__(field=field)
        self.alpha = alpha
        self.laplacian = Laplacian(field=field)

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return self.alpha * self.laplacian.compute(field, params)

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"diffusion({self.field})",
            "order": 2, 
            "type": "spatial"
        })
        return sig

class Advection(Operator):
    def __init__(self, v: float, field: str = "u"):
        super().__init__(field=field)
        self.v = v
        self.gradient = Gradient(field=field)

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        grad = self.gradient.compute(u, params)
        return self.v * grad[..., 0:1]

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"advection({self.field})",
            "order": 1, 
            "type": "spatial"
        })
        return sig

class Source(Operator):
    def __init__(self, fn, field: str = "u"):
        super().__init__(field=field)
        self.fn = fn

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        return self.fn(u, params)

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"source({self.field})",
            "order": 0, 
            "type": "source"
        })
        return sig

class Nonlinear(Operator):
    def __init__(self, fn, field: str = "u"):
        super().__init__(field=field)
        self.fn = fn

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        u = self._get_field(field, params)
        return self.fn(u, params)

    def signature(self) -> Dict[str, Any]:
        sig = super().signature()
        sig.update({
            "output": f"nonlinear({self.field})",
            "order": 0, 
            "type": "nonlinear"
        })
        return sig
