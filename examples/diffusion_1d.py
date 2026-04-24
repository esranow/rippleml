import torch
from ripple.core.system import System
from ripple.core.equation import Equation
from ripple.core.domain import Domain
from ripple.physics.operators import TimeDerivative, Diffusion
from ripple.core.simulation import Simulation
from ripple.core.system import Constraint

def main():
    # Gaussian initial condition
    N = 100
    x = torch.linspace(-5, 5, N).view(1, N, 1)
    u0 = torch.exp(-x**2)
    v0 = torch.zeros_like(u0)  # Unused in diffusion, but needed for run() API
    
    domain = Domain(spatial_dims=1)
    
    alpha = 0.5
    eq = Equation([
        (1.0, TimeDerivative(order=1)),
        (-1.0, Diffusion(alpha=alpha))
    ])
    
    def zero_bc(u, x, t):
        return (u[:,0]**2 + u[:,-1]**2).mean()

    sys = System(domain=domain, equation=eq, constraints=[Constraint(zero_bc)])
    
    dx = 10.0 / N
    dt = 0.4 * (dx**2) / alpha  # ensure stability (<= 0.5)
    
    sim = Simulation(system=sys, c=alpha, dt=dt, dx=dx)
    
    # Run simulation
    traj = sim.run(u0, v0, steps=200)
    
    # Visualize smoothing over time
    Simulation.visualize(traj, title="1D Diffusion (Smoothing of Gaussian)", interval=50)

if __name__ == "__main__":
    main()
