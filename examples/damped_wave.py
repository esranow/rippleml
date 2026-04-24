import torch
from ripple.physics.operators import TimeDerivative, Laplacian
from ripple.physics.equation import Equation
from ripple.core.system import System, Domain, Constraint
from ripple.core.simulation import Simulation, run_system

def run_damped_wave():
    print("Damped Wave Demo")
    beta = 0.5
    c = 1.0
    eq = Equation([
        (1.0, TimeDerivative(2)),
        (beta, TimeDerivative(1)),
        (-c**2, Laplacian())
    ])
    
    domain = Domain(spatial_dims=1, x_range=(0, 1), t_range=(0, 2))
    ic = Constraint(fn=lambda u, x, t: torch.sin(3.14159 * x), type="initial")
    sys = System(eq, domain, [ic])
    
    N = 100
    x = torch.linspace(0, 1, N)
    u0 = torch.sin(3.14159 * x).view(1, N, 1)
    v0 = torch.zeros_like(u0)
    
    print("Running simulation...")
    traj = run_system(sys, mode="sim", u0=u0, v0=v0, steps=100, dt=0.01, dx=0.01)
    
    print(f"Done. Trajectory shape: {traj.shape}")
    Simulation.visualize(traj, title="Damped Wave")

if __name__ == "__main__":
    run_damped_wave()
