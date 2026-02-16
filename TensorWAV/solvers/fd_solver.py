import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def solve_wave_fd_1d(
    u0: torch.Tensor, 
    v0: torch.Tensor, 
    c: float, 
    dt: float, 
    dx: float, 
    steps: int
) -> torch.Tensor:
    """
    Solve 1D Wave Equation u_tt = c^2 u_xx using Finite Difference.
    
    Args:
        u0 (torch.Tensor): Initial position (B, N, 1).
        v0 (torch.Tensor): Initial velocity (B, N, 1).
        c (float): Wave speed.
        dt (float): Time step.
        dx (float): Spatial step.
        steps (int): Number of time steps.
        
    Returns:
        torch.Tensor: Trajectory (B, steps+1, N, 1).
    """
    # Courant number
    C = (c * dt / dx) ** 2
    
    # Needs 3 time levels: u_prev, u_curr, u_next
    # Initial step: Taylor expansion
    # u(t+dt) = u(t) + dt*u_t(t) + 0.5*dt^2*u_tt(t)
    # u_tt = c^2 u_xx
    
    B, N, _ = u0.shape
    device = u0.device
    
    # Store history
    history = [u0]
    
    # Laplacian kernel [1, -2, 1]
    kernel = torch.tensor([[[1.0, -2.0, 1.0]]], device=device).view(1, 1, 3) 
    
    def laplacian(u_in):
        # u: (B, N, 1) -> (B, 1, N) for conv1d
        u_p = u_in.permute(0, 2, 1)
        # Pad for boundary (Dirichlet zero or replicating? Let's use replication or zero)
        # Using replicate for simple reflection or zero for fixed.
        # "1D ... solver" -> Zero BC is standard standard unless specified. 
        # Using Zero BC (pad with 0)
        u_pad = F.pad(u_p, (1, 1), mode='constant', value=0.0)
        u_xx = F.conv1d(u_pad, kernel) # (B, 1, N)
        return u_xx.permute(0, 2, 1) # (B, N, 1)

    # First step
    u_prev = u0
    # u1 = u0 + dt * v0 + 0.5 * C * (u(x+dx) - 2u + u(x-dx)) 
    # C in diff eq usually applied as (c*dt)^2 * u_xx. Here our laplacian is just [1, -2, 1] * 1/dx^2?
    # Kernel [1,-2,1] corresponds to dx^2 * u_xx.
    # So C * (kernel_output) = (c*dt/dx)^2 * (u(x+1)+...) = c^2 dt^2 * (u_xx approximation)
    
    # Correct scaling: u_xx approx = conv(u, [1,-2,1]) / dx^2
    # u(t+dt) = u + dt*v + 0.5 * c^2 * dt^2 * u_xx
    #         = u + dt*v + 0.5 * (c*dt/dx)^2 * conv(u, [1,-2,1])
    
    lap0 = laplacian(u0)
    u_curr = u0 + dt * v0 + 0.5 * C * lap0
    history.append(u_curr)
    
    for _ in range(steps - 1):
        lap = laplacian(u_curr)
        # Scheme: u_next = 2*u_curr - u_prev + C * lap
        u_next = 2 * u_curr - u_prev + C * lap
        
        history.append(u_next)
        u_prev = u_curr
        u_curr = u_next
        
    return torch.stack(history, dim=1)

def solve_wave_fd_2d(
    u0: torch.Tensor, 
    v0: torch.Tensor, 
    c: float, 
    dt: float, 
    dx: float, 
    dy: float,
    steps: int
) -> torch.Tensor:
    """
    Solve 2D Wave Equation using FD.
    
    Args:
        u0 (torch.Tensor): (B, H, W, 1).
        v0 (torch.Tensor): (B, H, W, 1).
        c, dt, dx, dy: Parameters.
        steps: Time steps.
        
    Returns:
        torch.Tensor: (B, steps+1, H, W, 1).
    """
    Cx = (c * dt / dx) ** 2
    Cy = (c * dt / dy) ** 2
    
    B, H, W, _ = u0.shape
    device = u0.device
    
    history = [u0]
    
    # Kernels
    kx = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]]], device=device)
    ky = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]]], device=device)
    
    def laplacian_2d(u_in):
        # u: (B, H, W, 1) -> (B, 1, H, W)
        u_p = u_in.permute(0, 3, 1, 2)
        u_pad = F.pad(u_p, (1, 1, 1, 1), mode='constant', value=0.0)
        
        # We apply x and y parts separately to scale by Cx, Cy
        # actually safer to just sum them if dx=dy and use one kernel, but spec allows dx!=dy
        res_x = F.conv2d(u_pad, kx)
        res_y = F.conv2d(u_pad, ky)
        return res_x, res_y

    u_prev = u0
    
    # First step
    lx, ly = laplacian_2d(u0)
    # u1 = u0 + dt*v0 + 0.5 * (Cx*lx + Cy*ly)
    # Convert result back to (B, H, W, 1)
    lap_term = (Cx * lx + Cy * ly).permute(0, 2, 3, 1)
    
    u_curr = u0 + dt * v0 + 0.5 * lap_term
    history.append(u_curr)
    
    for _ in range(steps - 1):
        lx, ly = laplacian_2d(u_curr)
        lap_term = (Cx * lx + Cy * ly).permute(0, 2, 3, 1)
        
        u_next = 2 * u_curr - u_prev + lap_term
        
        history.append(u_next)
        u_prev = u_curr
        u_curr = u_next
        
    return torch.stack(history, dim=1)
