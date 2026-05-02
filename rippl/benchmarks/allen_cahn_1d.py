"""
rippl.benchmarks.allen_cahn_1d — 1D Allen-Cahn equation.
"""
import torch
from rippl.core.system import System, Domain, Constraint
from rippl.physics.phase_field import PhaseFieldSystem
from rippl.nn.multi_field_mlp import MultiFieldMLP
from rippl.core.experiment import Experiment

def main():
    # 1. Equation
    eps = 0.1
    eq = PhaseFieldSystem.allen_cahn(M=1.0, epsilon=eps)
    
    # 2. Domain
    domain = Domain(spatial_dims=1, bounds=((-1, 1),), resolution=(100,))
    
    # 3. Constraints
    # IC: phi(x,0) = 0.5*(1 + tanh(x/eps))
    c_ic = Constraint(type="initial", field="phi", coords=None, 
                      value=lambda x: 0.5 * (1 + torch.tanh(x[:, 0:1] / eps)))
    
    # 4. System
    sys = System(equation=eq, domain=domain, fields=["phi"], constraints=[c_ic])
    
    # 5. Model
    model = MultiFieldMLP(in_dim=2, fields=["phi"], hidden=32, layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    exp = Experiment(sys, model, opt)
    
    print("Allen-Cahn 1D System built.")

if __name__ == "__main__":
    main()
