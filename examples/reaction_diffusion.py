import torch
from ripple.physics.operators import TimeDerivative, Diffusion, Nonlinear
from ripple.physics.equation import Equation
from ripple.core.system import System, Domain, Constraint
from ripple.core.simulation import Simulation, run_system

def run_rd_demo():
    print("Reaction-Diffusion Demo (Allen-Cahn variant)")
    
    # u_t = 0.001*u_xx + 5.0*(u - u^3)
    alpha = 0.001
    f = lambda u, inputs: 5.0 * (u - u**3)
    
    eq = Equation([
        (1.0, TimeDerivative(1)),
        (-alpha, Diffusion(alpha)),
        (-1.0, Nonlinear(f))
    ])
    
    domain = Domain(spatial_dims=1, x_range=(-1, 1), t_range=(0, 1))
    ic = Constraint(fn=lambda u, x, t: torch.zeros(1), type="initial")
    sys = System(eq, domain, [ic])
    
    # Simulation
    N = 100
    x = torch.linspace(-1, 1, N)
    # Initial state: random perturbations
    u0 = (0.2 * torch.rand(1, N, 1) - 0.1)
    
    print("Running simulation...")
    traj = run_system(
        sys, mode="sim", 
        u0=u0, steps=100, 
        dt=0.01, dx=2.0/N
    )
    
    print(f"Simulation done. Trajectory shape: {traj.shape}")
    
    # Visualize
    try:
        Simulation.visualize(traj, title="Reaction-Diffusion (Allen-Cahn)")
    except Exception as e:
        print(f"Visualization skipped: {e}")

if __name__ == "__main__":
    run_rd_demo()
