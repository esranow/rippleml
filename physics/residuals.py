import torch
import torch.nn as nn
from typing import Callable, Tuple, List, Optional
from ripple.physics.pde import PDESpec



def build_residual_fn(pde: PDESpec) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Builds a residual function for the given PDE specification.
    
    Args:
        pde (PDESpec): The PDE specification.
        
    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: 
            Function that takes (model_output, inputs) and returns the residual.
            Inputs are assumed to be (B, D) where D = spatial_dims + 1 (time).
            Last column is time t.
            Output u is (B, 1).
    """
    
    def residual_fn(u: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual.
        
        Residual = a*u_tt + b*u_t + c*Laplacian(u) + f(u) - g(x,t)
        
        Args:
            u (torch.Tensor): Output of the neural network, shape (B, 1).
            inputs (torch.Tensor): Inputs to the network, shape (B, D_spatial + 1).
                                   Assumes last dimension is time t.
                                   First D-1 dimensions are spatial x.
                                   REQUIRES GRADIENT.
        
        Returns:
            torch.Tensor: The residual tensor, shape (B, 1).
        """
        if not inputs.requires_grad:
            raise ValueError("Inputs tensor must usually require grad for physics loss.")
            
        # First derivatives (gradient w.r.t all inputs [x, t])
        grads = torch.autograd.grad(
            u, inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0] # Shape (B, D)
        
        # Split spatial and time derivatives
        # Last index is time
        # grads[..., :-1] are spatial derivatives (u_x, u_y, ...)
        # grads[..., -1] is time derivative u_t
        
        u_t = grads[..., -1:]
        u_spatial = grads[..., :-1]
        
        # Second derivatives
        # We need u_tt and Laplacian(u)
        
        # To get second derivatives, we differentiate the first derivatives again
        # But we need specific components.
        
        # u_tt: differentiate u_t w.r.t. time input only
        grad_u_t = torch.autograd.grad(
            u_t.sum(), inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if grad_u_t is None:
            u_tt = torch.zeros_like(u)
        else:
            u_tt = grad_u_t[..., -1:]
        
        # Laplacian(u): sum of u_xx, u_yy, etc.
        # This requires iterating over spatial dimensions if done naively via autograd
        # or creating a loop.
        
        spatial_dim = u_spatial.shape[-1]
        laplacian_val = torch.zeros_like(u)
        
        for i in range(spatial_dim):
            # Gradient of the i-th spatial component w.r.t inputs
            # We need the i-th component of the spatial gradient
            # Let's be careful with slicing.
            
            grad_component = u_spatial[..., i:i+1] # (B, 1)
            
            grad_grad = torch.autograd.grad(
                grad_component, inputs,
                grad_outputs=torch.ones_like(grad_component),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_grad is not None:
                # The i-th component of this new gradient corresponds to d(u_xi)/dxi = u_xixi
                laplacian_val = laplacian_val + grad_grad[..., i:i+1]
            
        # Construct terms
        # Wave eq: u_tt - c^2 * Lap(u) = 0  →  residual = a*u_tt + b*u_t - c*Lap(u)
        term_utt = pde.a * u_tt
        term_ut = pde.b * u_t
        term_lap = -pde.c * laplacian_val
        
        # Nonlinear term f(u)
        term_nonlinear = torch.zeros_like(u)
        if pde.nonlinear_type == 'linear':
            term_nonlinear = u
        elif pde.nonlinear_type is not None:
             # Basic implementation for now, or raise error if unsupported
            pass 

        # Forcing term g(x, t)
        term_forcing = torch.zeros_like(u)
        if pde.forcing is not None:
            term_forcing = pde.forcing(inputs)
            
        residual = term_utt + term_ut + term_lap + term_nonlinear - term_forcing  # = 0 at solution
        
        return residual

    return residual_fn
