import pytest
import torch
import math
from TensorWAV.datasets.generators import generate_sine_wave, generate_gaussian_bump
from TensorWAV.solvers.fd_solver import solve_wave_fd_1d, solve_wave_fd_2d
from TensorWAV.solvers.spectral_solver import solve_periodic_spectral_1d

torch.manual_seed(42)

def test_generators_shapes():
    B, N, D = 4, 32, 1
    grid = torch.linspace(0, 1, N).view(1, N, 1).repeat(B, 1, 1)
    
    sine = generate_sine_wave(grid)
    assert sine.shape == (B, N, 1)
    assert not torch.isnan(sine).any()
    
    bump = generate_gaussian_bump(grid)
    assert bump.shape == (B, N, 1)
    assert not torch.isnan(bump).any()

def test_fd_solver_1d_stability():
    """
    Test 1D FD solver produces finite values and correct shape.
    """
    B, N = 2, 50
    dx = 0.1
    dt = 0.05
    c = 1.0
    # C = (1*0.05/0.1)^2 = 0.25 < 1 (Stable)
    
    u0 = torch.zeros(B, N, 1)
    v0 = torch.zeros(B, N, 1)
    # Add a bump to u0
    u0[:, 20:30, :] = 1.0
    
    steps = 10
    u_hist = solve_wave_fd_1d(u0, v0, c, dt, dx, steps)
    
    assert u_hist.shape == (B, steps + 1, N, 1)
    assert not torch.isnan(u_hist).any()
    # Check max value doesn't explode
    assert torch.max(torch.abs(u_hist)) < 10.0

def test_fd_solver_2d_shape():
    """
    Test 2D FD solver shape.
    """
    B, H, W = 1, 10, 10
    u0 = torch.zeros(B, H, W, 1)
    v0 = torch.zeros(B, H, W, 1)
    
    sol = solve_wave_fd_2d(u0, v0, 1.0, 0.01, 0.1, 0.1, 5)
    
    assert sol.shape == (B, 6, H, W, 1)
    assert not torch.isnan(sol).any()

def test_spectral_solver_advection():
    """
    Test spectral solver propagates a wave.
    Advection: u_t + u_x = 0. u(x,t) = u(x-t).
    Periodic domain [0, 2pi].
    """
    N = 64
    x = torch.linspace(0, 2*math.pi, N+1)[:-1] # Periodic grid (drop last point)
    
    grid = x.view(1, N, 1)
    u0 = torch.sin(grid) # sin(x)
    
    # Advect for t = 2pi => should return to start
    # a=1.0
    steps = 10
    dt = 2*math.pi / steps
    
    u_hist = solve_periodic_spectral_1d(u0, a=1.0, dt=dt, steps=steps, L=2*math.pi)
    
    # Final step roughly equals initial
    u_final = u_hist[:, -1, :, :]
    
    # Spectral is exact in time here, so it should be very close up to float prec
    assert torch.allclose(u0, u_final, atol=1e-5)
