import torch
import torch.nn as nn
import numpy as np
from rippl.core.system import System, Domain, MovingBoundaryConstraint
from rippl.core.equation import Equation
from rippl.physics.operators import TimeDerivative, Laplacian
from rippl.core.experiment import Experiment
from rippl.physics.boundary import ParametricBoundary
from scipy.special import erf

def solve_stefan():
    # Parameters
    alpha = 1.0
    lam = 0.5
    T0 = 1.0
    Tm = 0.0
    
    # 1. Domain
    # We solve in x in [0, 1], t in [0, 1]
    # But the physics is only active for x < s(t)
    # For simplicity, we can solve on the whole domain and use MovingBoundaryConstraint
    # to enforce Tm at the front.
    domain = Domain(spatial_dims=1, bounds=((0, 1), (0.01, 1)), resolution=(50, 50))
    
    # 2. Equation: Heat Equation T_t - alpha * T_xx = 0
    eq = Equation(terms=[
        (1.0, TimeDerivative(field="T")),
        (-alpha, Laplacian(field="T", spatial_dims=1))
    ])
    
    # 3. Constraints
    constraints = []
    
    # Fixed Dirichlet at x=0
    # We sample points (0, t)
    t_vals = torch.linspace(0.01, 1.0, 50).reshape(-1, 1)
    x0_coords = torch.cat([torch.zeros_like(t_vals), t_vals], dim=-1)
    constraints.append(MovingBoundaryConstraint(
        field="T",
        boundary_fn=lambda e, m: x0_coords,
        value=lambda x: torch.full((x.shape[0], 1), T0)
    ))
    
    # Moving Boundary at x = s(t) = 2*lam*sqrt(alpha*t)
    def front_coords(epoch, model):
        t = torch.linspace(0.01, 1.0, 50).reshape(-1, 1)
        x_front = 2 * lam * np.sqrt(alpha) * torch.sqrt(t)
        return torch.cat([x_front, t], dim=-1)
        
    constraints.append(MovingBoundaryConstraint(
        field="T",
        boundary_fn=front_coords,
        value=Tm
    ))
    
    # 4. System
    sys = System(equation=eq, domain=domain, constraints=constraints, fields=["T"])
    
    # 5. Model and Optimizer
    model = nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 6. Experiment
    exp = Experiment(sys, model, opt)
    
    # 7. Train
    print("Training Stefan Problem...")
    coords, _ = domain.build_grid()
    coords = coords.reshape(-1, 2)
    exp.train(coords, epochs=500)
    
    # 8. Verify
    # Analytic solution at t=0.5
    t_test = 0.5
    x_test = torch.linspace(0, 2*lam*np.sqrt(alpha*t_test), 100).reshape(-1, 1)
    coords_test = torch.cat([x_test, torch.full_like(x_test, t_test)], dim=-1)
    
    T_pred = model(coords_test).detach().numpy()
    
    def analytic(x, t):
        return T0 * (1 - erf(x / (2 * np.sqrt(alpha * t))) / erf(lam))
        
    T_true = analytic(x_test.numpy(), t_test)
    
    mse = np.mean((T_pred - T_true)**2)
    print(f"MSE at t={t_test}: {mse:.2e}")
    
    if mse < 1e-2:
        print("Stefan Problem Benchmark PASSED")
    else:
        print("Stefan Problem Benchmark FAILED (High MSE)")

if __name__ == "__main__":
    solve_stefan()
