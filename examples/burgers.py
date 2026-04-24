import torch
from ripple.physics.operators import TimeDerivative, Nonlinear
from ripple.physics.equation import Equation
from ripple.core.system import System, Domain, Constraint
from ripple.core.simulation import Simulation, run_system

def run_burgers():
    print("Inviscid Burgers Demo")
    
    # u_t + u*u_x = 0
    # We define g(u, params) as the nonlinear term u*u_x
    def g(u, params):
        # u: (B, N, 1)
        dx = params.get("dx", 0.01)
        # Centered difference for u_x
        u_left = torch.roll(u, 1, dims=1)
        u_right = torch.roll(u, -1, dims=1)
        # Note: simplistic periodic boundaries for roll
        u_x = (u_right - u_left) / (2 * dx)
        return u * u_x
        
    eq = Equation([
        (1.0, TimeDerivative(1)),
        (1.0, Nonlinear(g))
    ])
    
    domain = Domain(spatial_dims=1, x_range=(0, 1), t_range=(0, 0.5))
    ic = Constraint(fn=lambda u, x, t: torch.exp(-100 * (x - 0.5)**2), type="initial")
    sys = System(eq, domain, [ic])
    
    N = 100
    x = torch.linspace(0, 1, N)
    u0 = torch.exp(-100 * (x - 0.5)**2).view(1, N, 1)
    
    print("Running simulation...")
    # dt=0.001 to keep it stable
    traj = run_system(sys, mode="sim", u0=u0, steps=100, dt=0.001, dx=0.01)
    
    print(f"Done. Trajectory shape: {traj.shape}")
    Simulation.visualize(traj, title="Inviscid Burgers (Centered FD)")

if __name__ == "__main__":
    run_burgers()
