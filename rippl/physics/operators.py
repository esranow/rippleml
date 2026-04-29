"""
rippl.physics.operators — Operator base + concrete implementations.
Reuses autograd patterns from rippl.physics.residuals.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from rippl.core.config import register_operator

class Operator:
    """Abstract base: forward(fields, coords, derived) -> tensor."""

    def __init__(self, field: str = "u"):
        self.field = field

    def forward(self, fields: Dict[str, torch.Tensor], coords: torch.Tensor, derived: Dict[str, torch.Tensor] = None) -> torch.Tensor: # fields: {name: (N, 1)}, coords: (N, D), derived: {name: (N, 1)}
        """Low-level tensor operation implementing the physics of the operator."""
        raise NotImplementedError

    def compute(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Backward compatibility for compute()."""
        fields = params.get("fields", {self.field: field})
        coords = params["inputs"]
        derived = params.get("derived", {}).copy()
        if "spatial_dims" in params:
            derived["spatial_dims"] = params["spatial_dims"]
        
        # Ensure all required derivatives are present in derived dict
        sig = self.signature()
        reqs = sig.get("requires_derived", [])
        missing = [r for r in reqs if r not in derived]
        if missing:
            from rippl.physics.derivatives import compute_all_derivatives
            derived.update(compute_all_derivatives(fields, coords, missing))
            
        return self.forward(fields, coords, derived)

    def signature(self) -> Dict[str, Any]:
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 0,
            "type": "generic",
            "requires_derived": []
        }


# ---------------------------------------------------------------------------
# Linear Operators (Updated to new contract)
# ---------------------------------------------------------------------------

@register_operator("laplacian")
class Laplacian(Operator):
    def __init__(self, field="u", spatial_dims=None):
        # spatial_dims: explicit int — number of spatial dims
        # if None, inferred from Domain at equation build time
        # NEVER default to total input dims
        super().__init__(field=field)
        self.spatial_dims = spatial_dims

    def signature(self) -> Dict[str, Any]:
        # We need second order derivatives for each spatial dimension
        # If spatial_dims is None, we default to 1 for signature purposes
        # but Equation will handle the actual precomputation.
        n = self.spatial_dims or 1 
        reqs = []
        for d in range(n):
            suffix = 'xy'[d] if d < 2 else str(d)
            reqs.append(f"{self.field}_{suffix}{suffix}")
        return {
            "inputs": [self.field],
            "output": f"laplacian({self.field})",
            "order": 2, 
            "type": "spatial",
            "requires_derived": reqs
        }

    def forward(self, fields, coords, derived=None):
        # only sum over dims 0..spatial_dims-1
        # NEVER include time dim (last dim)
        u = fields[self.field]
        result = torch.zeros_like(u)
        n_spatial = self.spatial_dims or (coords.shape[-1] - 1)
        for d in range(n_spatial):  # explicitly exclude time
            suffix = 'xy'[d] if d < 2 else str(d)
            key = f"{self.field}_{suffix}{suffix}"
            result = result + derived[key]
        return result


@register_operator("gradient")
class Gradient(Operator):
    def __init__(self, field="u", spatial_dims=None):
        super().__init__(field=field)
        self.spatial_dims = spatial_dims

    def signature(self) -> Dict[str, Any]:
        n = self.spatial_dims or 1
        reqs = []
        for d in range(n):
            suffix = 'xy'[d] if d < 2 else str(d)
            reqs.append(f"{self.field}_{suffix}")
        return {
            "inputs": [self.field],
            "output": f"grad({self.field})",
            "order": 1, 
            "type": "spatial",
            "requires_derived": reqs
        }

    def forward(self, fields, coords, derived=None):
        u = fields[self.field]
        n_spatial = self.spatial_dims or (coords.shape[-1] - 1)
        grads = []
        for d in range(n_spatial):
            suffix = 'xy'[d] if d < 2 else str(d)
            key = f"{self.field}_{suffix}"
            grads.append(derived[key])
        return torch.cat(grads, dim=-1)


@register_operator("divergence")
class Divergence(Operator):
    def signature(self) -> Dict[str, Any]:
        return {
            "inputs": [self.field],
            "output": f"div({self.field})",
            "order": 1, 
            "type": "spatial",
            "requires_derived": []
        }

    def forward(self, fields, coords, derived=None):
        u = fields[self.field]
        inputs = coords
        spatial_dim = inputs.shape[-1]
        if derived and "spatial_dims" in derived:
            spatial_dim = derived["spatial_dims"]
        elif derived is not None and any("_t" in k for k in derived.keys()):
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


@register_operator("timederivative")
class TimeDerivative(Operator):
    def __init__(self, order: int = 1, field: str = "u"):
        super().__init__(field=field)
        self.order = order

    def signature(self) -> Dict[str, Any]:
        return {
            "inputs": [self.field],
            "output": f"dt^{self.order}({self.field})",
            "order": self.order,
            "type": "temporal",
            "requires_derived": []
        }

    def forward(self, fields, coords, derived=None):
        u = fields[self.field]
        inputs = coords
        for _ in range(self.order):
            g = torch.autograd.grad(
                u.sum(), inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            u = g[..., -1:] if g is not None else torch.zeros_like(u)
        return u


@register_operator("diffusion")
class Diffusion(Operator):
    def __init__(self, alpha: float, field: str = "u"):
        super().__init__(field=field)
        self.alpha = alpha
        self.laplacian = Laplacian(field=field)

    def signature(self) -> Dict[str, Any]:
        sig = self.laplacian.signature()
        return {
            "inputs": [self.field],
            "output": f"diffusion({self.field})",
            "order": 2, 
            "type": "spatial",
            "requires_derived": sig["requires_derived"]
        }

    def forward(self, fields, coords, derived=None):
        return self.alpha * self.laplacian.forward(fields, coords, derived)


@register_operator("advection")
class Advection(Operator):
    def __init__(self, v: float, field: str = "u"):
        super().__init__(field=field)
        self.v = v
        self.gradient = Gradient(field=field)

    def signature(self) -> Dict[str, Any]:
        sig = self.gradient.signature()
        return {
            "inputs": [self.field],
            "output": f"advection({self.field})",
            "order": 1, 
            "type": "spatial",
            "requires_derived": sig["requires_derived"]
        }

    def forward(self, fields, coords, derived=None):
        grad = self.gradient.forward(fields, coords, derived)
        return self.v * grad[..., 0:1]


@register_operator("source")
class Source(Operator):
    def __init__(self, fn, field: str = "u"):
        super().__init__(field=field)
        self.fn = fn

    def signature(self) -> Dict[str, Any]:
        return {
            "inputs": [self.field],
            "output": f"source({self.field})",
            "order": 0, 
            "type": "source",
            "requires_derived": []
        }

    def forward(self, fields, coords, derived=None):
        u = fields[self.field]
        params = {"inputs": coords, "fields": fields, "derived": derived}
        return self.fn(u, params)


@register_operator("nonlinear")
class Nonlinear(Operator):
    def __init__(self, fn, field: str = "u"):
        super().__init__(field=field)
        self.fn = fn

    def signature(self) -> Dict[str, Any]:
        return {
            "inputs": [self.field],
            "output": f"nonlinear({self.field})",
            "order": 0, 
            "type": "nonlinear",
            "requires_derived": []
        }

    def forward(self, fields, coords, derived=None):
        u = fields[self.field]
        params = {"inputs": coords, "fields": fields, "derived": derived}
        return self.fn(u, params)


# ---------------------------------------------------------------------------
# Part B: Nonlinear Operators
# ---------------------------------------------------------------------------

@register_operator("burgers_advection")
class BurgersAdvection(Operator):
    # u * ∂u/∂x — requires u_x precomputed
    def __init__(self, field="u", spatial_dim=0):
        super().__init__(field=field)
        self.spatial_dim = spatial_dim
        self._dim_name = ["x", "y", "z"][spatial_dim]
    
    def signature(self):
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 1,
            "type": "burgers_advection",
            "requires_derived": [f"{self.field}_{self._dim_name}"]
        }
    
    def forward(self, fields, coords, derived=None):
        u = fields[self.field]
        u_deriv = derived[f"{self.field}_{self._dim_name}"]
        return u * u_deriv


@register_operator("nonlinear_advection")
class NonlinearAdvection(Operator):
    # (u·∇)u — for vector fields, u advects itself
    # used in Navier-Stokes momentum
    def __init__(self, field_u="u", field_v="v"):
        super().__init__(field=field_u)
        self.field_u = field_u
        self.field_v = field_v
    
    def signature(self):
        return {
            "inputs": [self.field_u, self.field_v],
            "output": self.field_u,
            "order": 1,
            "type": "nonlinear_advection",
            "requires_derived": [
                f"{self.field_u}_x", f"{self.field_u}_y",
                f"{self.field_v}_x", f"{self.field_v}_y"
            ]
        }
    
    def forward(self, fields, coords, derived=None):
        u = fields[self.field_u]
        v = fields[self.field_v]
        u_x = derived[f"{self.field_u}_x"]
        u_y = derived[f"{self.field_u}_y"]
        v_x = derived[f"{self.field_v}_x"]
        v_y = derived[f"{self.field_v}_y"]
        # returns (conv_u, conv_v) as stacked tensor
        conv_u = u * u_x + v * u_y
        conv_v = u * v_x + v * v_y
        return torch.cat([conv_u, conv_v], dim=-1)


@register_operator("pressure_gradient")
class PressureGradient(Operator):
    # ∂p/∂x or ∂p/∂y
    def __init__(self, field_p="p", direction=0):
        super().__init__(field=field_p)
        self.field_p = field_p
        self.direction = direction
        self._dim_name = ["x", "y", "z"][direction]
    
    def signature(self):
        return {
            "inputs": [self.field_p],
            "output": self.field_p,
            "order": 1,
            "type": f"pressure_gradient_{self._dim_name}",
            "requires_derived": [f"{self.field_p}_{self._dim_name}"]
        }
    
    def forward(self, fields, coords, derived=None):
        return derived[f"{self.field_p}_{self._dim_name}"]


@register_operator("velocity_divergence")
class VelocityDivergence(Operator):
    # ∇·u = u_x + v_y — incompressibility constraint
    def __init__(self, field_u="u", field_v="v"):
        super().__init__(field=field_u)
        self.field_u = field_u
        self.field_v = field_v
    
    def signature(self):
        return {
            "inputs": [self.field_u, self.field_v],
            "output": self.field_u,
            "order": 1,
            "type": "velocity_divergence",
            "requires_derived": [
                f"{self.field_u}_x", f"{self.field_v}_y"
            ]
        }
    
    def forward(self, fields, coords, derived=None):
        return derived[f"{self.field_u}_x"] + derived[f"{self.field_v}_y"]


# ---------------------------------------------------------------------------
# Structural Mechanics Operators
# ---------------------------------------------------------------------------

@register_operator("strain_tensor")
class StrainTensor(Operator):
    # ε = 0.5*(∇u + ∇uᵀ)
    # for 2D: εxx=ux_x, εyy=uy_y, εxy=0.5*(ux_y + uy_x)
    def __init__(self, field_ux="ux", field_uy="uy"):
        super().__init__(field=field_ux)
        self.field_ux = field_ux
        self.field_uy = field_uy
    
    def signature(self):
        return {
            "inputs": [self.field_ux, self.field_uy],
            "output": "strain",
            "order": 1,
            "type": "strain_tensor",
            "requires_derived": [
                f"{self.field_ux}_x", f"{self.field_ux}_y",
                f"{self.field_uy}_x", f"{self.field_uy}_y"
            ]
        }
    
    def forward(self, fields, coords, derived=None):
        exx = derived[f"{self.field_ux}_x"]
        eyy = derived[f"{self.field_uy}_y"]
        exy = 0.5 * (derived[f"{self.field_ux}_y"] + 
                     derived[f"{self.field_uy}_x"])
        return torch.cat([exx, eyy, exy], dim=-1)  # (N, 3)

@register_operator("stress_tensor")
class StressTensor(Operator):
    # σ = λ*tr(ε)*I + 2μ*ε — linear elasticity constitutive law
    # λ, μ: Lamé parameters
    # σxx = (λ+2μ)*εxx + λ*εyy
    # σyy = λ*εxx + (λ+2μ)*εyy
    # σxy = 2μ*εxy
    def __init__(self, lame_lambda=1.0, lame_mu=1.0):
        super().__init__(field="strain")
        self.lam = lame_lambda
        self.mu = lame_mu
    
    def signature(self):
        return {
            "inputs": ["strain"],
            "output": "stress",
            "order": 0,
            "type": "stress_tensor",
            "requires_derived": []
        }
    
    def forward(self, fields, coords, derived=None):
        strain = fields["strain"]  # (N, 3): [exx, eyy, exy]
        exx, eyy, exy = strain[..., 0:1], strain[..., 1:2], strain[..., 2:3]
        sxx = (self.lam + 2*self.mu)*exx + self.lam*eyy
        syy = self.lam*exx + (self.lam + 2*self.mu)*eyy
        sxy = 2*self.mu*exy
        return torch.cat([sxx, syy, sxy], dim=-1)  # (N, 3)

@register_operator("elastic_equilibrium")
class ElasticEquilibrium(Operator):
    # ∇·σ + f = 0
    # div(σ)_x = σxx_x + σxy_y
    # div(σ)_y = σxy_x + σyy_y
    def __init__(self, field_ux="ux", field_uy="uy",
                 lame_lambda=1.0, lame_mu=1.0,
                 body_force_x=0.0, body_force_y=0.0):
        super().__init__(field=field_ux)
        self.lam = lame_lambda
        self.mu = lame_mu
        self.fx = body_force_x
        self.fy = body_force_y
        self.field_ux = field_ux
        self.field_uy = field_uy
    
    def signature(self):
        return {
            "inputs": [self.field_ux, self.field_uy],
            "output": self.field_ux,
            "order": 2,
            "type": "elastic_equilibrium",
            "requires_derived": [
                f"{self.field_ux}_xx", f"{self.field_ux}_yy",
                f"{self.field_ux}_xy", f"{self.field_uy}_xx",
                f"{self.field_uy}_yy", f"{self.field_uy}_xy"
            ]
        }
    
    def forward(self, fields, coords, derived=None):
        ux_xx = derived[f"{self.field_ux}_xx"]
        ux_yy = derived[f"{self.field_ux}_yy"]
        ux_xy = derived[f"{self.field_ux}_xy"]
        uy_xx = derived[f"{self.field_uy}_xx"]
        uy_yy = derived[f"{self.field_uy}_yy"]
        uy_xy = derived[f"{self.field_uy}_xy"]
        
        res_x = (self.lam+2*self.mu)*ux_xx + self.mu*ux_yy + \
                (self.lam+self.mu)*uy_xy + self.fx
        res_y = (self.lam+self.mu)*ux_xy + self.mu*uy_xx + \
                (self.lam+2*self.mu)*uy_yy + self.fy
        return torch.cat([res_x, res_y], dim=-1)


# ---------------------------------------------------------------------------
# Quantum Mechanics (Schrödinger) Operators
# ---------------------------------------------------------------------------

@register_operator("schrodinger_kinetic")
class SchrodingerKinetic(Operator):
    # -ℏ²/2m * ∇²ψ — kinetic energy operator
    # split into real/imag: operates on both components
    def __init__(self, hbar=1.0, mass=1.0,
                 field_real="psi_real", field_imag="psi_imag"):
        super().__init__(field=field_real)
        self.coeff = -(hbar**2) / (2*mass)
        self.field_real = field_real
        self.field_imag = field_imag
    
    def signature(self):
        return {
            "inputs": [self.field_real, self.field_imag],
            "output": self.field_real,
            "order": 2,
            "type": "schrodinger_kinetic",
            "requires_derived": [
                f"{self.field_real}_xx", f"{self.field_imag}_xx"
            ]
        }
    
    def forward(self, fields, coords, derived=None):
        kin_real = self.coeff * derived[f"{self.field_real}_xx"]
        kin_imag = self.coeff * derived[f"{self.field_imag}_xx"]
        return torch.cat([kin_real, kin_imag], dim=-1)

@register_operator("potential_term")
class PotentialTerm(Operator):
    # V(x)*ψ — potential energy
    def __init__(self, potential_fn, field_real="psi_real",
                 field_imag="psi_imag"):
        # potential_fn: callable V(coords) → (N,1)
        super().__init__(field=field_real)
        self.V = potential_fn
        self.field_real = field_real
        self.field_imag = field_imag
    
    def signature(self):
        return {
            "inputs": [self.field_real, self.field_imag],
            "output": self.field_real,
            "order": 0,
            "type": "potential_term",
            "requires_derived": []
        }
    
    def forward(self, fields, coords, derived=None):
        V = self.V(coords)
        pot_real = V * fields[self.field_real]
        pot_imag = V * fields[self.field_imag]
        return torch.cat([pot_real, pot_imag], dim=-1)

@register_operator("schrodinger_time")
class SchrodingerTimeEvolution(Operator):
    # iℏ ∂ψ/∂t — left hand side of TDSE
    # real part: -ℏ * ψ_imag_t
    # imag part: +ℏ * ψ_real_t
    def __init__(self, hbar=1.0, field_real="psi_real",
                 field_imag="psi_imag"):
        super().__init__(field=field_real)
        self.hbar = hbar
        self.field_real = field_real
        self.field_imag = field_imag
    
    def signature(self):
        return {
            "inputs": [self.field_real, self.field_imag],
            "output": self.field_real,
            "order": 1,
            "type": "schrodinger_time",
            "requires_derived": [
                f"{self.field_real}_t", f"{self.field_imag}_t"
            ]
        }
    
    def forward(self, fields, coords, derived=None):
        lhs_real = -self.hbar * derived[f"{self.field_imag}_t"]
        lhs_imag = self.hbar * derived[f"{self.field_real}_t"]
        return torch.cat([lhs_real, lhs_imag], dim=-1)


@register_operator("artificial_viscosity")
class ArtificialViscosity(Operator):
    """
    Dynamic diffusion injected where |∇u| exceeds threshold.
    Prevents NaN in near-shock regions without custom CUDA kernels.
    Uses open-source PyTorch ops only.
    
    u_visc = epsilon(x) * Δu
    epsilon(x) = epsilon_max * r(x) / (max(r) + 1e-8)
    where r(x) = local residual magnitude
    
    Only activates where |∇u| > gradient_threshold.
    """
    def __init__(self, field: str = "u",
                 epsilon_max: float = 0.1,
                 gradient_threshold: float = 10.0,
                 spatial_dims: int = None):
        super().__init__(field=field)
        self.epsilon_max = epsilon_max
        self.gradient_threshold = gradient_threshold
        self.spatial_dims = spatial_dims

    def signature(self) -> dict:
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 2,
            "type": "artificial_viscosity",
            "requires_derived": []  # computes its own derivatives internally
        }

    def forward(self, fields: dict, coords: torch.Tensor,
                derived: dict = None) -> torch.Tensor:
        u = fields[self.field]
        n_spatial = self.spatial_dims or (coords.shape[-1] - 1)

        # Compute |∇u| — gradient magnitude
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
        du = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        grad_mag = du[:, :n_spatial].pow(2).sum(dim=-1, keepdim=True).sqrt()

        # Laplacian for diffusion term
        lap = torch.zeros_like(u)
        for d in range(n_spatial):
            u_d = du[:, d:d+1]
            u_dd = torch.autograd.grad(u_d.sum(), coords, create_graph=True)[0][:, d:d+1]
            lap = lap + u_dd

        # Dynamic coefficient — only where gradient exceeds threshold
        mask = (grad_mag > self.gradient_threshold).float()
        r = (grad_mag * mask)
        epsilon = self.epsilon_max * r / (r.max() + 1e-8)


        return epsilon * lap


@register_operator("level_set")
class LevelSetOperator(Operator):
    """
    Tracks interface via Level Set Equation: ∂φ/∂t + v·∇φ = 0.
    Supports constant velocity, field-based velocity, or callable velocity.
    """
    def __init__(self, velocity: Union[str, torch.Tensor, Callable] = None, field: str = "phi", spatial_dims: int = None):
        super().__init__(field=field)
        self.velocity = velocity
        self.spatial_dims = spatial_dims
        self.gradient = Gradient(field=field, spatial_dims=spatial_dims)

    def signature(self) -> Dict[str, Any]:
        sig = self.gradient.signature()
        reqs = sig["requires_derived"] + [f"{self.field}_t"]
        return {
            "inputs": [self.field],
            "output": f"level_set({self.field})",
            "order": 1,
            "type": "interface_tracking",
            "requires_derived": reqs
        }

    def forward(self, fields: Dict[str, torch.Tensor], coords: torch.Tensor, derived: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        phi_t = derived[f"{self.field}_t"]
        grad_phi = self.gradient.forward(fields, coords, derived)
        
        # Determine velocity v
        if isinstance(self.velocity, str):
            v = fields[self.velocity]
        elif callable(self.velocity):
            # Callable expects (fields, coords, derived)
            v = self.velocity(fields, coords, derived)
        elif isinstance(self.velocity, torch.Tensor):
            v = self.velocity
        else:
            v = torch.zeros_like(grad_phi) # Default to zero velocity
            
        # Advection term: v · ∇φ
        # v: (N, D), grad_phi: (N, D)
        # Ensure dimensions match
        if v.shape[-1] > grad_phi.shape[-1]:
            v = v[..., :grad_phi.shape[-1]]
            
        advection = torch.sum(v * grad_phi, dim=-1, keepdim=True)
        return phi_t + advection
