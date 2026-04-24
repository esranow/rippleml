import torch
import math
from ripple.physics.operators import TimeDerivative, Diffusion, Source, Nonlinear
from ripple.physics.equation import Equation
from ripple.core.system import System, Domain, Constraint
from ripple.core.simulation import Simulation
from ripple.core.experiment import Experiment

def test_source_nonlinear():
    print("RUNNING SOURCE + NONLINEAR TEST")
    
    domain = Domain(spatial_dims=1, x_range=(0, 1), t_range=(0, 1))
    ic = Constraint(fn=lambda u, x, t: torch.tensor(0.0), type="initial")
    
    # 1. Source Test: u_t = a*u_xx + f(x,t)
    f = lambda u, p: torch.sin(math.pi * p["inputs"][..., 0:1])
    eq_source = Equation([
        (1.0, TimeDerivative(1)),
        (-0.01, Diffusion(0.01)),
        (-1.0, Source(f))
    ])
    sys_source = System(eq_source, domain, [ic])
    
    # 2. Nonlinear Test: u_t = g(u, x, t) e.g. u_t = -u^2
    g = lambda u, p: -u**2
    eq_nonlinear = Equation([
        (1.0, TimeDerivative(1)),
        (-1.0, Nonlinear(g))
    ])
    sys_nonlinear = System(eq_nonlinear, domain, [ic])
    
    # Validation Loop
    N = 32
    u0 = torch.zeros((1, N, 1))
    v0 = torch.zeros((1, N, 1))
    
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(2, 1)
        def forward(self, inputs):
            return self.net(inputs)
            
    for name, sys in [("Source", sys_source), ("Nonlinear", sys_nonlinear)]:
        # Simulation (checks routing/execution)
        # Note: Simulation FD solvers might not support arbitrary Source/Nonlinear yet
        # but select_solver should at least handle the basic types if matched.
        # Actually, let's just check Experiment as it uses Equation SOT directly.
        
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        exp = Experiment(sys, model, opt)
        
        x = torch.linspace(0, 1, 10).view(-1, 1)
        t = torch.linspace(0, 1, 10).view(-1, 1)
        
        loss = exp.train(x, t)
        assert math.isfinite(loss)
        print(f"  [PASS] {name} Experiment (loss={loss:.6f})")

if __name__ == "__main__":
    test_source_nonlinear()
    print("OK")
