"""
rippl.benchmarks.turing_patterns — Gray-Scott Turing Patterns.
"""
import torch
from rippl.core.system import System, Domain, Constraint
from rippl.physics.reaction_diffusion import TuringSystem
from rippl.nn.multi_field_mlp import MultiFieldMLP
from rippl.core.experiment import Experiment

def main():
    # 1. Equation
    ts = TuringSystem(**TuringSystem.SPOTS)
    eq_system = ts.build_equation_system()
    
    # 2. Domain
    domain = Domain(spatial_dims=2, bounds=((0, 1), (0, 1)), resolution=(40, 40))
    
    # 3. Constraints
    # IC: u=1, v=0 except center
    def ic_u(x):
        return torch.ones(x.shape[0], 1)
    def ic_v(x):
        v = torch.zeros(x.shape[0], 1)
        center_mask = (torch.abs(x[:, 0:1] - 0.5) < 0.1) & (torch.abs(x[:, 1:2] - 0.5) < 0.1)
        v[center_mask] = 1.0
        return v
        
    c_u = Constraint(type="initial", field="u", coords=None, value=ic_u)
    c_v = Constraint(type="initial", field="v", coords=None, value=ic_v)
    
    # 4. System
    sys = System(equation=eq_system, domain=domain, fields=["u", "v"], constraints=[c_u, c_v])
    
    # 5. Model
    model = MultiFieldMLP(in_dim=2, fields=["u", "v"], hidden=64, layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    exp = Experiment(sys, model, opt)
    
    print("Turing Pattern System built.")

if __name__ == "__main__":
    main()
