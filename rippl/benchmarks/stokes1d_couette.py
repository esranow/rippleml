"""
rippl.benchmarks.stokes1d_couette — 1D Simplified Stokes/Couette Flow.
Fields: u, p (velocity, pressure).
Analytic solution: u(x) = x, p(x) = 1 - x.
Equations:
  1. Laplacian(u) = 0
  2. Grad(u) + Grad(p) = 0
"""
import torch
from rippl.physics.operators import Laplacian, Gradient
from rippl.physics.equation import Equation
from rippl.core.equation_system import EquationSystem
from rippl.core.system import System, Domain, Constraint
from rippl.models.multi_field_mlp import MultiFieldMLP
from rippl.core.experiment import Experiment

def main():
    # 1. Operators & Equations
    eq_momentum = Equation(terms=[
        (1.0, Laplacian(field="u"))
    ])
    
    eq_continuity = Equation(terms=[
        (1.0, Gradient(field="u")),
        (10.0, Gradient(field="p"))
    ])
    
    eq_system = EquationSystem(equations=[eq_momentum, eq_continuity])

    # 2. Domain
    # 1D spatial domain [0, 1].
    domain = Domain(
        spatial_dims=1,
        bounds=((0, 1),),
        resolution=(50,)
    )

    # 3. Constraints (Analytic BCs)
    # u(0)=0, u(1)=1
    # p(0)=1, p(1)=0
    c_u0 = Constraint(type="dirichlet", field="u", coords=torch.tensor([[0.0]]), value=0.0)
    c_u1 = Constraint(type="dirichlet", field="u", coords=torch.tensor([[1.0]]), value=1.0)
    # Pressure gauge: p(0) = 1.0 — breaks pressure uniqueness degeneracy
    x_gauge = torch.zeros(1, 1)
    p_gauge_target = 1.0
    c_p_gauge = Constraint(type="dirichlet", field="p", coords=x_gauge, value=p_gauge_target)
    
    # 4. System
    sys = System(
        equation=eq_system,
        domain=domain,
        fields=["u", "p"],
        constraints=[c_u0, c_u1, c_p_gauge]
    )
    sys.validate()

    # 5. Model & Training Setup
    # MultiFieldMLP expects (N, in_dim) inputs.
    model = MultiFieldMLP(fields=["u", "p"], in_dim=1, hidden=20, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # grad_clip=1.0
    
    exp = Experiment(sys, model, opt)
    
    # Grid for training
    coords, _ = domain.build_grid()
    coords = coords.reshape(-1, 1)
    
    print("Stokes 1D Couette System built and validated.")
    print(f"Fields: {sys.fields}")
    
    # Dry run
    res = exp.train(coords, epochs=10)
    print(f"Initial training results: {res['loss']:.6f}")
    
    # Verify analytic (optional check)
    with torch.no_grad():
        test_coords = torch.tensor([[0.5]])
        pred = model(test_coords)
        print(f"Prediction at x=0.5: u={pred['u'].item():.4f}, p={pred['p'].item():.4f}")
        print(f"Target at x=0.5: u=0.5, p=0.5")

if __name__ == "__main__":
    main()
