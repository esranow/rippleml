"""
rippl.benchmarks.csg_laplace_annulus — Solving Laplace Equation on CSG Annulus.
"""
import torch
import numpy as np
from rippl.core.system import System, Constraint
from rippl.physics.operators import Laplacian
from rippl.core.equation import Equation
from rippl.geometry.csg import Annulus, CSGDomain
from rippl.nn.multi_field_mlp import MultiFieldMLP
from rippl.core.experiment import Experiment

def main():
    # 1. Geometry & Domain
    # Outer circle (R=1.0) minus Inner circle (R=0.5)
    shape = Annulus(center=(0.0, 0.0), r_inner=0.5, r_outer=1.0)
    # CSGDomain automatically handles bounding box and sampling
    domain = CSGDomain(shape, spatial_dims=2)
    
    # 2. Physics: Laplace Equation (∇²u = 0)
    eq = Equation([(1.0, Laplacian(field="u"))])
    
    # 3. Boundary Constraints
    # We sample the inner and outer boundaries separately for Dirichlet conditions
    n_boundary = 400
    
    # Inner boundary: R=0.5
    theta_in = torch.rand(n_boundary) * 2 * np.pi
    coords_in = torch.stack([0.5 * torch.cos(theta_in), 0.5 * torch.sin(theta_in)], dim=-1)
    c_inner = Constraint(type="dirichlet", field="u", coords=coords_in, value=0.0)
    
    # Outer boundary: R=1.0
    theta_out = torch.rand(n_boundary) * 2 * np.pi
    coords_out = torch.stack([1.0 * torch.cos(theta_out), 1.0 * torch.sin(theta_out)], dim=-1)
    c_outer = Constraint(type="dirichlet", field="u", coords=coords_out, value=1.0)
    
    # 4. System Definition
    sys = System(
        equation=eq, 
        domain=domain, 
        fields=["u"], 
        constraints=[c_inner, c_outer]
    )
    
    # 5. Model & Training Setup
    model = MultiFieldMLP(in_dim=2, fields=["u"], hidden=64, layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    exp = Experiment(sys, model, opt)
    
    print("CSG Laplace Annulus benchmark initialized.")
    print(f"Domain volume estimate: {domain.get_sampler().estimate_volume():.4f}")
    
    # Ready for training
    # exp.train(coords=domain.to_collocation_points(2000, has_time=False), epochs=2000)

if __name__ == "__main__":
    main()
