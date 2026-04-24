import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def solve_diffusion_fd_1d(
    u0: torch.Tensor,
    steps: int,
    alpha: float,
    dt: float,
    dx: float
) -> torch.Tensor:
    """
    Solve 1D Diffusion Equation u_t = alpha * u_xx using Finite Difference.
    """
    if alpha * dt / (dx ** 2) > 0.5:
        raise ValueError("CFL violation: alpha*dt/dx^2 > 0.5")
    
    device = u0.device
    history = [u0]
    kernel = torch.tensor([[[1.0, -2.0, 1.0]]], device=device).view(1, 1, 3)
    
    def laplacian(u_in):
        u_p = u_in.permute(0, 2, 1)
        u_pad = F.pad(u_p, (1, 1), mode='constant', value=0.0)
        u_xx = F.conv1d(u_pad, kernel)
        return u_xx.permute(0, 2, 1)

    u_curr = u0
    coeff = alpha * dt / (dx ** 2)
    
    for _ in range(steps):
        lap = laplacian(u_curr)
        u_next = u_curr + coeff * lap
        history.append(u_next)
        u_curr = u_next
        
    return torch.stack(history, dim=1)

def solve_advection_fd_1d(
    u0: torch.Tensor,
    steps: int,
    v: float,
    dt: float,
    dx: float
) -> torch.Tensor:
    """Solve 1D Advection u_t = -v * u_x using Upwind."""
    if abs(v) * dt / dx > 1.0:
        raise ValueError("CFL violation for advection")
        
    history = [u0]
    u_curr = u0
    
    for _ in range(steps):
        # Forward or backward diff depending on v
        u_p = u_curr.permute(0, 2, 1)
        if v > 0:
            u_pad = F.pad(u_p, (1, 0), mode='constant', value=0.0)
            kernel = torch.tensor([[[-1.0, 1.0]]], device=u0.device).view(1, 1, 2)
        else:
            u_pad = F.pad(u_p, (0, 1), mode='constant', value=0.0)
            kernel = torch.tensor([[[-1.0, 1.0]]], device=u0.device).view(1, 1, 2)
        
        grad = F.conv1d(u_pad, kernel).permute(0, 2, 1)
        
        # for v > 0: u[i] - u[i-1] -> kernel applied is u[i] - u[i-1]
        # the sign depends on upwind direction.
        # Minimal upwind: 
        if v > 0:
            # u_new = u - v dt/dx (u[i] - u[i-1])
            u_next = u_curr - (v * dt / dx) * grad
        else:
            # u_new = u - v dt/dx (u[i+1] - u[i]) -> grad with pad (0,1) gives u[i+1]-u[i] 
            # wait, kernel [-1, 1] on u[i], u[i+1] gives u[i+1] - u[i].
            u_next = u_curr - (v * dt / dx) * grad
            
        history.append(u_next)
        u_curr = u_next
        
    return torch.stack(history, dim=1)

def solve_advdiff_fd_1d(
    u0: torch.Tensor,
    steps: int,
    alpha: float,
    v: float,
    dt: float,
    dx: float
) -> torch.Tensor:
    """Solve Advection-Diffusion."""
    if alpha * dt / (dx ** 2) > 0.5:
        raise ValueError("CFL violation: alpha*dt/dx^2 > 0.5")
    if abs(v) * dt / dx > 1.0:
        raise ValueError("CFL violation (advection)")
        
    history = [u0]
    u_curr = u0
    lap_kernel = torch.tensor([[[1.0, -2.0, 1.0]]], device=u0.device).view(1, 1, 3)
    
    for _ in range(steps):
        u_p = u_curr.permute(0, 2, 1)
        lap_pad = F.pad(u_p, (1, 1), mode='constant', value=0.0)
        lap = F.conv1d(lap_pad, lap_kernel).permute(0, 2, 1)
        
        if v > 0:
            grad_pad = F.pad(u_p, (1, 0), mode='constant', value=0.0)
            grad_kernel = torch.tensor([[[-1.0, 1.0]]], device=u0.device).view(1, 1, 2)
        else:
            grad_pad = F.pad(u_p, (0, 1), mode='constant', value=0.0)
            grad_kernel = torch.tensor([[[-1.0, 1.0]]], device=u0.device).view(1, 1, 2)
            
        grad = F.conv1d(grad_pad, grad_kernel).permute(0, 2, 1)
        
        u_next = u_curr + (alpha * dt / (dx**2)) * lap - (v * dt / dx) * grad
        history.append(u_next)
        u_curr = u_next
        
    return torch.stack(history, dim=1)

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
    if C > 1.0:
        raise ValueError(f"CFL violation: (c*dt/dx)^2 = {C:.4f} > 1.0 — reduce dt or increase dx")
    
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
    if Cx + Cy > 1.0:
        raise ValueError(f"CFL violation: Cx+Cy = {Cx+Cy:.4f} > 1.0 — reduce dt, dx, or dy")
    
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
def solve_reaction_diffusion_fd_1d(
    u0: torch.Tensor,
    steps: int,
    alpha: float,
    equation,
    dt: float,
    dx: float
) -> torch.Tensor:
    """Solve 1D Reaction-Diffusion u_t = alpha*u_xx + f(u,x,t) using FD."""
    device = u0.device
    N = u0.shape[1]
    # Build spatial coords (N, 1)
    x = torch.linspace(0, (N-1)*dx, N, device=device).view(N, 1)
    
    history = [u0]
    u_curr = u0
    
    kernel = torch.tensor([[[1.0, -2.0, 1.0]]], device=device).view(1, 1, 3)
    
    def laplacian(u_in):
        u_p = u_in.permute(0, 2, 1)
        u_pad = F.pad(u_p, (1, 1), mode='constant', value=0.0)
        return F.conv1d(u_pad, kernel).permute(0, 2, 1)

    # Extract non-temporal, non-diffusion terms from equation
    from ripple.physics.operators import TimeDerivative, Diffusion
    non_pde_terms = [(c, op) for c, op in equation.terms 
                     if not isinstance(op, (TimeDerivative, Diffusion))]

    for i in range(steps):
        t = i * dt
        # inputs = cat([x, t]) -> (N, 2)
        t_vec = torch.full((N, 1), t, device=device)
        inputs = torch.cat([x, t_vec], dim=-1)
        params = {"inputs": inputs, "dx": dx, "dt": dt}
        
        # Physics parts
        lap = laplacian(u_curr)
        f_val = torch.zeros_like(u_curr)
        for coeff, op in non_pde_terms:
            f_val = f_val + coeff * op.compute(u_curr, params)
            
        u_next = u_curr + dt * (alpha * lap / (dx**2) + f_val)
        history.append(u_next)
        u_curr = u_next
        
    return torch.stack(history, dim=1)
def solve_damped_wave_fd_1d(
    u0: torch.Tensor,
    v0: torch.Tensor,
    beta: float,
    c: float,
    dt: float,
    dx: float,
    steps: int
) -> torch.Tensor:
    """Solve 1D Damped Wave Equation u_tt + beta*u_t = c^2*u_xx."""
    device = u0.device
    C = (c * dt / dx) ** 2
    if C > 1.0:
        raise ValueError(f"CFL violation: {C:.4f} > 1.0")
        
    history = [u0]
    B, N, _ = u0.shape
    kernel = torch.tensor([[[1.0, -2.0, 1.0]]], device=device).view(1, 1, 3)
    
    def laplacian(u_in):
        u_p = u_in.permute(0, 2, 1)
        u_pad = F.pad(u_p, (1, 1), mode='constant', value=0.0)
        return F.conv1d(u_pad, kernel).permute(0, 2, 1)

    # Initial step u1
    lap0 = laplacian(u0)
    # u1 = u0 + dt*v0 + 0.5*dt^2*(c^2*lap0 - beta*v0)
    u_curr = u0 + dt * v0 + 0.5 * (dt**2) * ( (c**2) * lap0 / (dx**2) - beta * v0)
    history.append(u_curr)
    u_prev = u0
    
    # Pre-calculate factors
    f1 = 1.0 / (1.0 + beta * dt / 2.0)
    f2 = 2.0
    f3 = 1.0 - beta * dt / 2.0
    
    for _ in range(steps - 1):
        lap = laplacian(u_curr)
        u_next = f1 * ( (c**2 * dt**2 / dx**2) * lap + f2 * u_curr - f3 * u_prev )
        history.append(u_next)
        u_prev = u_curr
        u_curr = u_next
        
    return torch.stack(history, dim=1)
