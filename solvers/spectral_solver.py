import torch
import torch.fft
import math
from typing import Optional

def solve_periodic_spectral_1d(
    u0: torch.Tensor, 
    a: float, 
    dt: float, 
    steps: int, 
    L: float = 2.0 * math.pi
) -> torch.Tensor:
    """
    Solve linear advection u_t + a u_x = 0 using Spectral method.
    Periodic BC is implied by Fourier basis.
    
    Args:
        u0 (torch.Tensor): Initial condition (B, N, 1).
        a (float): Advection speed.
        dt (float): Time step.
        steps (int): Number of steps.
        L (float): Domain length (default 2pi).
        
    Returns:
        torch.Tensor: (B, steps+1, N, 1).
    """
    B, N, _ = u0.shape
    device = u0.device
    
    # Wavenumbers k
    # 2*pi*k / L
    k = torch.fft.fftfreq(N, d=L/(2*math.pi*N)).to(device) * (2*math.pi/L * N) 
    # Actually just standard fftfreq * 2pi/dx ?
    # d = L/N. 1/d = N/L.
    # fftfreq returns f cycles/unit. k = 2pi * f.
    # fftfreq(n, d) -> f
    # k = 2 * pi * f
    k = 2 * math.pi * torch.fft.fftfreq(N, d=L/N).to(device)
    
    # Shape k to broadcast: (1, N) from (N,)
    # u0 is (B, N, 1). We usually operate on N dim.
    # Let's align k for broadcasting.
    k_tens = k.view(1, N, 1)
    
    # FFT of u0
    # u0 map (B, N, 1) -> (B, N, 1) complex
    u_hat = torch.fft.fft(u0, dim=1)
    
    history = [u0]
    
    # Exact solution in frequency domain for u_t = -a u_x
    # u_hat(t) = u_hat(0) * exp(-i * a * k * t)
    
    times = torch.arange(1, steps + 1, device=device) * dt
    
    for t in times:
        # Compute exact evolution for this time
        # This is more stable/accurate than stepping for linear spectral
        phase = -1j * a * k_tens * t
        u_hat_t = u_hat * torch.exp(phase)
        
        u_curr = torch.fft.ifft(u_hat_t, dim=1).real
        history.append(u_curr)
        
    return torch.stack(history, dim=1)
