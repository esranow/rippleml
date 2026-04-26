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
    
    # 2. Simulation
    N = 64
    dt = 0.01
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(N,))
    # Dummy initial constraint (field, coords, value)
    ic = Constraint(type="initial", field="u", coords=torch.zeros((1, 2)), value=0.0) 
    sys = System(eq, domain, [ic])
    
    u0 = torch.exp(-100 * (torch.linspace(0, 1, N) - 0.5)**2).view(1, N, 1)
    v0 = torch.zeros_like(u0) # ignored for advdiff
    
    sim = Simulation(sys)
    out = sim.run(u0, v0, steps=100, dt=dt)
    traj = out["field"]
    
    assert traj.shape == (1, 101, N, 1)
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
    
    # Coords: (Batch, N, 2)
    coords = torch.rand(4, 10, 2)
    
    res = exp.train(coords)
    loss = res["loss"]
    assert math.isfinite(loss)
    print(f"  [PASS] Experiment (loss={loss:.6f})")

if __name__ == "__main__":
    test_advdiff_combined()
    print("OK")
