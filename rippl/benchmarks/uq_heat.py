"""
rippl.benchmarks.uq_heat — Heat Equation with MC Dropout Uncertainty.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from rippl.core.system import System, Domain, Constraint
from rippl.physics.operators import TimeDerivative, Laplacian
from rippl.core.equation import Equation
from rippl.training.uq import ProbabilisticExperiment
from rippl.nn.multi_field_mlp import MultiFieldMLP

def main():
    # 1. Physics Definition
    alpha = 0.01
    # u_t - alpha * u_xx = 0
    eq = Equation([(1.0, TimeDerivative(field="u")), (-alpha, Laplacian(field="u"))])
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(100,))
    
    # IC: u(x,0) = sin(pi*x)
    def ic_fn(x):
        return torch.sin(np.pi * x[:, 0:1])
        
    ic = Constraint(type="initial", field="u", coords=None, value=ic_fn)
    
    # BC: u(0,t) = u(1,t) = 0
    def bc_coords_fn(domain):
        # x=0 and x=1
        t = torch.linspace(0, 1, 50).reshape(-1, 1)
        x0 = torch.zeros_like(t)
        x1 = torch.ones_like(t)
        return torch.cat([torch.cat([x0, t], dim=-1), torch.cat([x1, t], dim=-1)], dim=0)

    bc = Constraint(type="dirichlet", field="u", coords=bc_coords_fn(domain), value=0.0)
    
    sys = System(equation=eq, domain=domain, fields=["u"], constraints=[ic, bc])
    
    # 2. Model & Optimizer
    # We use a standard MLP which will be wrapped for MCDropout
    model = MultiFieldMLP(in_dim=2, fields=["u"], hidden=64, layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Probabilistic Experiment (MC Dropout)
    print("Initializing Probabilistic Experiment (MC Dropout)...")
    exp = ProbabilisticExperiment(
        sys, model, method="mc_dropout", 
        dropout_rate=0.1, 
        opt=opt,
        adaptive_loss=True
    )
    
    # 4. Training
    coords, _ = domain.build_grid()
    coords = coords.reshape(-1, 2)
    print("Starting training...")
    exp.train(coords=coords, epochs=2000)
    
    # 5. Inference with UQ
    print("Running UQ inference...")
    test_x = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_fixed = 0.5
    test_coords = torch.cat([test_x, torch.full_like(test_x, t_fixed)], dim=-1)
    
    uq_res = exp.predict_with_uncertainty(test_coords, n_samples=50)
    mean = uq_res["mean"].detach().cpu().numpy().flatten()
    std = uq_res["std"].detach().cpu().numpy().flatten()
    x_plot = test_x.numpy().flatten()
    
    # 6. Analytic Comparison
    # u(x,t) = exp(-alpha * pi^2 * t) * sin(pi*x)
    analytic = np.exp(-alpha * np.pi**2 * t_fixed) * np.sin(np.pi * x_plot)
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, analytic, 'k--', label="Analytic", alpha=0.8)
    plt.plot(x_plot, mean, 'b-', label="PINN Mean")
    plt.fill_between(x_plot, mean - 2*std, mean + 2*std, color='blue', alpha=0.2, label="95% CI (±2 std)")
    
    plt.xlabel("x")
    plt.ylabel("u(x, t=0.5)")
    plt.title(f"Uncertainty Quantification: Heat Equation (t={t_fixed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Report
    report = exp.uncertainty_report(test_coords)
    print(f"UQ Report: {report}")
    
    plt.savefig("uq_heat_benchmark.png")
    print("Benchmark complete. Plot saved to uq_heat_benchmark.png")

if __name__ == "__main__":
    main()
