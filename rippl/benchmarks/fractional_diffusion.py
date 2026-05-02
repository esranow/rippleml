"""
rippl.benchmarks.fractional_diffusion — Fractional Subdiffusion.
Equation: D^0.5_t u = Laplacian(u)
"""
import torch
from rippl.core.system import System, Domain, Constraint
from rippl.physics.fractional import FractionalSystem
from rippl.nn.multi_field_mlp import MultiFieldMLP
from rippl.core.experiment import Experiment

def main():
    # 1. Equation
    # Subdiffusion alpha=0.5
    eq = FractionalSystem.subdiffusion(alpha=0.5, spatial_dims=1, diffusivity=1.0)
    
    # 2. Domain
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(50,))
    
    # 3. Constraints
    # u(x,0) = sin(pi*x)
    # u(0,t) = 0, u(1,t) = 0
    c_ic = Constraint(type="initial", field="u", coords=None, value=lambda x: torch.sin(torch.pi * x[:, 0:1]))
    c_bc0 = Constraint(type="dirichlet", field="u", coords=None, value=0.0)
    
    # 4. System
    sys = System(equation=eq, domain=domain, fields=["u"], constraints=[c_ic])
    
    # 5. Model
    model = MultiFieldMLP(in_dim=2, fields=["u"], hidden=32, layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    exp = Experiment(sys, model, opt)
    
    print("Fractional Subdiffusion System built.")
    # res = exp.train(coords, epochs=100)

if __name__ == "__main__":
    main()
