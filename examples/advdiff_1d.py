import torch
from ripple.core.system import System
from ripple.core.equation import Equation
from ripple.core.domain import Domain
from ripple.physics.operators import TimeDerivative, Diffusion, Advection
from ripple.core.simulation import Simulation

def main():
    N = 100
    x = torch.linspace(-5, 5, N).view(1, N, 1)
    u0 = torch.exp(-x**2)
    v0 = torch.zeros_like(u0)
    
    domain = Domain(spatial_dims=1)
    v = 0.5
    alpha = 0.1
    eq = Equation([
        (1.0, TimeDerivative(order=1)),
        (1.0, Advection(v=v)),
        (-1.0, Diffusion(alpha=alpha))
    ])
    
    sys = System(domain=domain, equation=eq)
    
    dx = 10.0 / N
    dt = min(dx / v * 0.9, 0.4 * (dx**2) / alpha)
    
    sim = Simulation(system=sys, dt=dt, dx=dx)
    traj = sim.run(u0, v0, steps=100)
    
    Simulation.visualize(traj, title="1D Advection-Diffusion", interval=50)

if __name__ == "__main__":
    main()
