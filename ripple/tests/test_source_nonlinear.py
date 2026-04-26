import torch
import math
from ripple.physics.operators import TimeDerivative, Diffusion, Source, Nonlinear
from ripple.physics.equation import Equation
from ripple.core.system import System, Domain, Constraint
from ripple.core.experiment import Experiment

def test_source_nonlinear():
    print("RUNNING SOURCE + NONLINEAR TEST")
    
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(32,))
    ic = Constraint(type="initial", field="u", coords=torch.zeros((1, 2)), value=0.0)
    
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
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(2, 1)
        def forward(self, inputs):
            return self.net(inputs)
            
    for name, sys in [("Source", sys_source), ("Nonlinear", sys_nonlinear)]:
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        exp = Experiment(sys, model, opt)
        
        coords = torch.rand(4, 10, 2)
        
        train_res = exp.train(coords)
        loss = train_res["loss"]
        assert math.isfinite(loss)
        print(f"  [PASS] {name} Experiment (loss={loss:.6f})")

if __name__ == "__main__":
    test_source_nonlinear()
    print("OK")
