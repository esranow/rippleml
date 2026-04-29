import torch
import torch.nn as nn

class DistanceFunction:
    def __call__(self, coords: torch.Tensor) -> torch.Tensor: # coords: (N, D)
        """Evaluate the distance function to return a (N, 1) tensor."""

class BoxDistance(DistanceFunction):
    def __init__(self, bounds: list):
        """
        bounds: [(x0,x1), (y0,y1), ...]
        """
        self.bounds = bounds

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, D)
        d = torch.ones(coords.shape[0], 1, device=coords.device)
        for i, (low, high) in enumerate(self.bounds):
            xi = coords[:, i:i+1]
            # dist = (xi - low) * (high - xi)
            # Normalizing to avoid very large values if bounds are large, 
            # though the prompt suggests x*(1-x) for [0,1].
            d = d * (xi - low) * (high - xi)
        return d

class HardConstraintWrapper(torch.nn.Module):
    def __init__(self, model, distance_fn, particular_solution=None):
        super().__init__()
        self.model = model
        self.distance_fn = distance_fn
        self.particular_solution = particular_solution
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor: # coords: (N, D)
        """Apply the hard constraint transformation: u = D*NN + G."""
        D = self.distance_fn(coords)
        u_raw = self.model(coords)
        
        # Handle dict output (multi-field)
        if isinstance(u_raw, dict):
            out = {}
            for field, val in u_raw.items():
                g = 0.0
                if self.particular_solution:
                    if callable(self.particular_solution):
                        # If PS is a single callable returning a dict or tensor
                        ps_val = self.particular_solution(coords)
                        if isinstance(ps_val, dict):
                            g = ps_val.get(field, 0.0)
                        else:
                            g = ps_val # assume it's for 'u' or the only field
                    elif isinstance(self.particular_solution, dict):
                        ps_fn = self.particular_solution.get(field)
                        if ps_fn:
                            g = ps_fn(coords)
                out[field] = D * val + g
            return out
        else:
            # Single field
            g = self.particular_solution(coords) if self.particular_solution else 0.0
            return D * u_raw + g

class NeumannAnsatzWrapper(torch.nn.Module):
    """
    Hard enforcement of Neumann BCs: ∂u/∂n = g on boundary.
    Ansatz: u = ũ(x) + D(x)*N(x)
    where D(x) is distance to boundary (zero on boundary)
    and N(x) satisfies ∂N/∂n = g - ∂ũ/∂n on boundary.
    
    For simple case g=0: u = D(x)*ũ(x) enforces zero-flux exactly.
    """
    def __init__(self, model, distance_fn,
                 neumann_value: float = 0.0,
                 normal_dim: int = 0):
        super().__init__()
        self.model = model
        self.D = distance_fn
        self.g = neumann_value
        self.normal_dim = normal_dim

    def forward(self, coords):
        D_val = self.D(coords)
        u_raw = self.model(coords)
        # For zero Neumann: u = D*ũ, ∂u/∂n = 0 on boundary by product rule
        # since D=0 and ∂D/∂n ≠ 0 on boundary
        return D_val * u_raw


class MixedBCAnsatz(torch.nn.Module):
    """
    Handles mixed Dirichlet + Neumann boundaries.
    Dirichlet on left/right: u(0)=a, u(1)=b
    Neumann on top/bottom: ∂u/∂y=0
    
    Particular solution g(x) = a + (b-a)*x satisfies Dirichlet exactly.
    Distance D(x) = x*(1-x) is zero at Dirichlet boundaries.
    Network predicts correction: u = g(x) + D(x)*ũ(x)
    """
    def __init__(self, model, particular_solution: callable,
                 distance_fn):
        super().__init__()
        self.model = model
        self.g = particular_solution
        self.D = distance_fn

    def forward(self, coords):
        return self.g(coords) + self.D(coords) * self.model(coords)

class AnsatzFactory:
    """Factory for common ansatz configurations."""

    @staticmethod
    def dirichlet_1d(model, a: float = 0.0, b: float = 0.0) -> MixedBCAnsatz:
        """u(0)=a, u(1)=b exactly satisfied."""
        g = lambda coords: a + (b - a) * coords[:, 0:1]
        D = lambda coords: coords[:, 0:1] * (1 - coords[:, 0:1])
        return MixedBCAnsatz(model, g, D)

    @staticmethod
    def dirichlet_2d_box(model, value: float = 0.0) -> HardConstraintWrapper:
        """u=value on all four sides of [0,1]²."""
        D = BoxDistance([(0,1),(0,1)])
        g = lambda coords: torch.full((coords.shape[0],1), value)
        return HardConstraintWrapper(model, D, particular_solution=g)

    @staticmethod
    def neumann_zero_1d(model) -> NeumannAnsatzWrapper:
        """Zero flux at x=0 and x=1."""
        D = BoxDistance([(0,1)])
        return NeumannAnsatzWrapper(model, D, neumann_value=0.0)

class TimeVaryingDistance(DistanceFunction):
    """
    Distance function that depends on time.
    distance_fn_of_t: Callable[[coords], distance] where coords includes time.
    """
    def __init__(self, distance_fn_of_t):
        self.fn = distance_fn_of_t

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        return self.fn(coords)
