import pytest
import torch
import math
from TensorWAV.diagnostics.metrics import l2_error, relative_l2_error
from TensorWAV.diagnostics.energy import wave_energy
from TensorWAV.diagnostics.spectral import spectral_error

torch.manual_seed(42)

def test_metrics():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 2.0])
    assert l2_error(a, b) == 0.0
    
    c = torch.tensor([1.0, 4.0])
    # diff [0, 2], norm = 2
    assert l2_error(a, c) == 2.0
    
    # Rel error: 2.0 / sqrt(1+16) = 2 / 4.123
    rel = relative_l2_error(a, c)
    assert rel > 0

def test_wave_energy_conservation():
    """
    Test energy conservation for analytical standing wave.
    u = sin(x) cos(ct)
    v = -c sin(x) sin(ct)
    """
    # Domain [0, 2pi], c=1
    N = 100
    dx = 2 * math.pi / N
    x = torch.linspace(0, 2*math.pi, N)
    c = 1.0
    
    # t = 0
    # u = sin(x), v = 0
    # E = integral 0.5 c^2 cos^2(x) dx = 0.5 * pi
    # (average of cos^2 is 0.5 over 2pi -> pi)
    
    u0 = torch.sin(x).view(1, N, 1)
    v0 = torch.zeros(1, N, 1)
    
    E0 = wave_energy(u0, v0, c, dx).item()
    
    # t = pi/2c (cos=0, sin=1)
    # u = 0, v = -c sin(x)
    # E = integral 0.5 (-c sin)^2 = 0.5 c^2 sin^2 dx = 0.5 * pi
    u1 = torch.zeros(1, N, 1)
    v1 = (-c * torch.sin(x)).view(1, N, 1)
    
    E1 = wave_energy(u1, v1, c, dx).item()
    
    # Check conservation (approx due to discretization)
    assert abs(E0 - E1) < 0.1

def test_spectral_metrics():
    # Signal with specific freq
    x = torch.linspace(0, 1, 64)
    s1 = torch.sin(2 * math.pi * 5 * x).view(1, 64, 1)
    
    assert spectral_error(s1, s1) < 1e-6
    
    s2 = torch.sin(2 * math.pi * 6 * x).view(1, 64, 1)
    # Different locally and spectrally
    assert spectral_error(s1, s2) > 0.1
