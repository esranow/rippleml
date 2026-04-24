
import torch
import torch.nn as nn
from ripple.core.system import System, Domain, Constraint
from ripple.physics.equation import Equation
from ripple.physics.operators import TimeDerivative, Laplacian, Diffusion, Advection
from ripple.core.simulation import Simulation, run_system

def test_systems():
    # 1D Domain
    dom = Domain(spatial_dims=1, x_range=(0.0, 1.0), t_range=(0.0, 1.0))
    
    # 1. WAVE
    c = 1.0
    c2 = c**2
    wave_eq = Equation([
        (1.0, TimeDerivative(order=2)),
        (-c2, Laplacian())
    ])
    wave_sys = System(wave_eq, dom, constraints=[Constraint(lambda u,x,t: u, type="initial")])
    
    # 2. DIFFUSION
    alpha = 0.01
    diff_eq = Equation([
        (1.0, TimeDerivative(order=1)),
        (-1.0, Diffusion(alpha))
    ])
    diff_sys = System(diff_eq, dom, constraints=[Constraint(lambda u,x,t: u, type="initial")])
    
    # 3. ADVECTION
    v = 0.5
    adv_eq = Equation([
        (1.0, TimeDerivative(order=1)),
        (1.0, Advection(v))
    ])
    adv_sys = System(adv_eq, dom, constraints=[Constraint(lambda u,x,t: u, type="initial")])
    
    systems = {
        "wave": wave_sys,
        "diffusion": diff_sys,
        "advection": adv_sys
    }
    
    N = 50
    u0 = torch.sin(torch.pi * torch.linspace(0, 1, N)).view(1, N, 1)
    v0 = torch.zeros_like(u0)
    
    results = {}
    
    for name, sys in systems.items():
        print(f"Testing {name}...")
        try:
            # Simulation
            sim = Simulation(sys, c=c, dt=0.001, dx=1.0/N)
            traj = sim.run(u0, v0, steps=5)
            
            # Experiment
            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))
                def forward(self, x): return self.net(x)
            
            model = MLP()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            x = torch.linspace(0, 1, 10).view(-1, 1)
            t = torch.linspace(0, 1, 10).view(-1, 1)
            
            from ripple.core.experiment import Experiment
            exp = Experiment(sys, model, opt)
            loss = exp.train(x, t)
            
            results[name] = {
                "status": "PASS",
                "sim_shape": traj.shape,
                "exp_loss": f"{loss:.6f}"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {
                "status": "FAIL",
                "error": str(e)
            }
            
    for name, res in results.items():
        print(f"{name.upper()}: {res['status']}")
        if res['status'] == "PASS":
            print(f"  Shape: {res['sim_shape']}")
            print(f"  Loss:  {res['exp_loss']}")
        else:
            print(f"  Error: {res['error']}")

if __name__ == "__main__":
    test_systems()
