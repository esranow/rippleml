"""
rippl.benchmarks.eikonal_2d — Eikonal Equation in 2D.
Equation: |∇V| = 1
"""
import torch
from rippl.core.system import System, Domain, Constraint
from rippl.physics.hamilton_jacobi import HJSystem
from rippl.nn.multi_field_mlp import MultiFieldMLP
from rippl.core.experiment import Experiment

def main():
    # 1. Equation
    eq = HJSystem.eikonal(spatial_dims=2)
    
    # 2. Domain
    domain = Domain(spatial_dims=2, bounds=((0, 1), (0, 1)), resolution=(20, 20))
    
    # 3. Constraints
    # V(0.5, 0.5) = 0
    c_source = Constraint(type="dirichlet", field="V", coords=torch.tensor([[0.5, 0.5]]), value=0.0)
    
    # 4. System
    sys = System(equation=eq, domain=domain, fields=["V"], constraints=[c_source])
    
    # 5. Model
    model = MultiFieldMLP(in_dim=2, fields=["V"], hidden=64, layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    exp = Experiment(sys, model, opt)
    
    print("Eikonal 2D System built.")

if __name__ == "__main__":
    main()
