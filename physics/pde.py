import dataclasses
from typing import Callable, Optional, Union
import torch

@dataclasses.dataclass
class PDESpec:
    """
    Specification for a Partial Differential Equation (PDE).
    
    Represents the equation:
    a * u_tt + b * u_t + c * Laplacian(u) + f(u) - g(x, t) = 0 (residual term)
    
    Attributes:
        a (float): Coefficient for the second time derivative term (u_tt).
        b (float): Coefficient for the first time derivative term (u_t).
        c (float): Coefficient for the spatial Laplacian term (Laplacian(u)).
        nonlinear_type (Optional[str]): Type of nonlinearity f(u). Currently supports 'linear' or None.
        forcing (Optional[Callable[[torch.Tensor], torch.Tensor]]): Forcing function g(x, t).
    """
    a: float = 1.0  # u_tt coefficient
    b: float = 0.0  # u_t coefficient
    c: float = 1.0  # Laplacian(u) coefficient. Residual form: a*u_tt + b*u_t - c*Lap(u) = 0 → wave eq: a=1,b=0,c=c²
    nonlinear_type: Optional[str] = None
    forcing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
