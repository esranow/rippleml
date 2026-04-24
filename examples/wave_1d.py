"""
examples/wave_1d.py
Gaussian pulse propagation demo using ripple System + Simulation.

Run:
    python examples/wave_1d.py
"""
import torch
import math

from ripple.physics.operators import Laplacian, TimeDerivative
from ripple.physics.equation import Equation
from ripple.core import System, Domain
from ripple.core.simulation import Simulation

# ── domain ────────────────────────────────────────────────────────────
N   = 128       # spatial grid points
L   = 10.0      # domain length  [0, L]
dx  = L / N
dt  = 0.05      # satisfies CFL: c*dt/dx = 0.5 < 1
c   = 1.0
steps = 40

# ── Gaussian initial condition u0 = exp(-k*(x - x0)^2) ───────────────
k  = 5.0        # sharpness
x0 = L / 2.0   # centred
x  = torch.linspace(0, L, N)
u0 = torch.exp(-k * (x - x0) ** 2).view(1, N, 1)   # (B=1, N, 1)
v0 = torch.zeros_like(u0)

# ── System ────────────────────────────────────────────────────────────
eq  = Equation(terms=[(1.0, TimeDerivative(2)), (-c**2, Laplacian())])
dom = Domain(spatial_dims=1, x_range=(0.0, L), t_range=(0.0, steps * dt))
sys = System(eq, dom)

# ── run ───────────────────────────────────────────────────────────────
sim  = Simulation(sys, c=c, dt=dt, dx=dx)
out  = sim.run(u0, v0, steps=steps)
traj = out["field"]
print(f"Trajectory: {tuple(traj.shape)}")

# ── visualize ─────────────────────────────────────────────────────────
Simulation.visualize(traj, title="1-D Gaussian Wave Propagation")
