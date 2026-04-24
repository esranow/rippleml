import torch
import math
from ripple.physics.operators import TimeDerivative, Advection, Diffusion
from ripple.physics.equation import Equation
from ripple.core.system import System, Domain, Constraint
from ripple.core.simulation import Simulation
from ripple.core.experiment import Experiment

def test_advdiff_combined():
    print("RUNNING ADVDIFF COMBINED TEST")
    
    # 1. Build System
    v = 0.5
    a = 0.01
    eq = Equation([
        (1.0, TimeDerivative(1)),
        (v, Advection(v)),
        (-a, Diffusion(a))
    ])
    
    domain = Domain(spatial_dims=1, x_range=(0, 1), t_range=(0, 1))
    # Dummy initial constraint
    ic = Constraint(fn=lambda u, x, t: torch.zeros(1), type="initial")
    sys = System(eq, domain, [ic])
    
    # 2. Simulation
    N = 64
    u0 = torch.exp(-100 * (torch.linspace(0, 1, N) - 0.5)**2).view(1, N, 1)
    v0 = torch.zeros_like(u0) # ignored for advdiff
    
    sim = Simulation(sys, dt=0.001, dx=1/N)
    traj = sim.run(u0, v0, steps=10)
    
    assert traj.shape == (1, 11, N, 1)
    assert not torch.isnan(traj).any()
    print("  [PASS] Simulation")
    
    # 3. Experiment
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        def forward(self, inputs):
            return self.net(inputs)
            
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    exp = Experiment(sys, model, opt)
    
    x = torch.linspace(0, 1, 10).view(-1, 1)
    t = torch.linspace(0, 1, 10).view(-1, 1)
    
    loss = exp.train(x, t)
    assert math.isfinite(loss)
    print(f"  [PASS] Experiment (loss={loss:.6f})")

if __name__ == "__main__":
    test_advdiff_combined()
    print("OK")
