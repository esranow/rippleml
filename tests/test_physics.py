import pytest
import torch
import numpy as np
from TensorWAV.physics.pde import PDESpec
from TensorWAV.physics.residuals import build_residual_fn
from TensorWAV.physics.boundary import DirichletBC, NeumannBC, PeriodicBC

# Set seeds
torch.manual_seed(42)

def analytic_wave_1d(x_t: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Analytic solution to 1D wave equation: u_tt - c^2 u_xx = 0
    u(x, t) = sin(x - c*t)
    """
    x = x_t[..., 0:1]
    t = x_t[..., 1:2]
    return torch.sin(x - c * t)

def test_residual_calculation():
    """
    Test residual calculation for 1D wave equation.
    PDE: u_tt - c^2 * u_xx = 0
    Solution: u = sin(x - ct)
    """
    # PDE Coeffs: a=1 (u_tt), c=-c_val^2 (u_xx term moved to RHS? No, usually a*u_tt + c*Laplacian = 0)
    # Wave Eq: u_tt - v^2 u_xx = 0 => u_tt + (-v^2) u_xx = 0
    # So a=1, c=-v^2.
    
    v = 2.0
    pde = PDESpec(a=1.0, b=0.0, c=-(v**2))
    
    residual_fn = build_residual_fn(pde)
    
    # Create grid
    batch_size = 100
    # Inputs: (x, t)
    inputs = torch.rand(batch_size, 2, requires_grad=True) * 2 * np.pi
    
    # Compute analytic u
    u = analytic_wave_1d(inputs, c=v)
    
    # Compute residual
    # Expected: (1)*(-v^2 sin) + (-v^2)*(-sin) = -v^2 sin + v^2 sin = 0
    
    res = residual_fn(u, inputs)
    
    # Check if residual is close to zero
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-5)

def test_dirichlet_bc():
    """
    Test Dirichlet BC loss calculation.
    """
    def exact_fn(x):
        return torch.zeros_like(x[..., 0:1]) # target is 0
        
    bc = DirichletBC(value_fn=exact_fn)
    
    inputs = torch.randn(10, 2)
    u_pred = torch.ones(10, 1) * 0.5 # Error = 0.5
    
    loss = bc(u_pred, inputs)
    
    # MSE of (0.5 - 0)^2 = 0.25
    assert torch.isclose(loss, torch.tensor(0.25))

def test_neumann_bc():
    """
    Test Neumann BC loss calculation.
    """
    # Target derivative = 1.0
    def deriv_fn(x):
        return torch.ones_like(x[..., 0:1])
        
    bc = NeumannBC(derivative_fn=deriv_fn, normal_idx=0)
    
    # Inputs
    inputs = torch.randn(10, 2, requires_grad=True)
    # Define u = 2 * x (so du/dx = 2)
    u = 2 * inputs[..., 0:1]
    
    loss = bc(u, inputs)
    
    # Expected grad = 2. Target = 1. Error = (2-1)^2 = 1.
    assert torch.isclose(loss, torch.tensor(1.0))

def test_periodic_bc():
    """
    Test Periodic BC.
    """
    bc = PeriodicBC()
    
    # u_left = 1, u_right = 0.8
    u = torch.tensor([[1.0], [0.8]]) 
    inputs = torch.randn(2, 2) # Dummy
    
    loss = bc(u, inputs)
    
    # MSE (1.0 - 0.8)^2 = 0.04
    assert torch.isclose(loss, torch.tensor(0.04), atol=1e-6)
