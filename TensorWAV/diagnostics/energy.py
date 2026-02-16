import torch

def wave_energy(
    u: torch.Tensor, 
    v: torch.Tensor, 
    c: float, 
    dx: float,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute the energy functional for the wave equation:
    E = 0.5 * integral( u_t^2 + c^2 * |grad u|^2 ) dx
    
    Args:
        u (torch.Tensor): Displacement field (B, N, 1).
        v (torch.Tensor): Velocity field u_t (B, N, 1).
        c (float): Wave speed.
        dx (float): Spatial step size.
        dim (int): Spatial dimensionality (implied 1D for now given shape).
        
    Returns:
        torch.Tensor: Energy per batch item (B, 1).
    """
    # Kinetic Energy: 0.5 * v^2
    kinetic = 0.5 * (v ** 2)
    
    # Potential Energy: 0.5 * c^2 * (u_x)^2
    # Compute u_x via finite difference for estimation analysis
    # Centered difference: (u[i+1] - u[i-1]) / 2dx
    
    # Pad to handle boundary
    u_vals = u.squeeze(-1) # (B, N)
    u_pad = torch.nn.functional.pad(u_vals, (1, 1), mode='replicate')
    
    # Gradient
    u_x = (u_pad[:, 2:] - u_pad[:, :-2]) / (2 * dx)
    u_x = u_x.unsqueeze(-1) # Back to (B, N, 1)
    
    potential = 0.5 * (c ** 2) * (u_x ** 2)
    
    # Integrate (Riemann sum)
    # Energy density = Kinetic + Potential
    density = kinetic + potential
    
    # Integral approx sum * dx
    energy = torch.sum(density, dim=1, keepdim=True) * dx
    
    return energy
