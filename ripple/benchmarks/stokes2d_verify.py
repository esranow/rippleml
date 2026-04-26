"""
ripple.benchmarks.stokes2d_verify — 2D Steady Stokes PINN.
Fields: u, v, p (x-velocity, y-velocity, pressure).
"""
import torch
from ripple.physics.operators import Laplacian, Gradient, Divergence
from ripple.physics.equation import Equation
from ripple.core.equation_system import EquationSystem
from ripple.core.system import System, Domain, Constraint
from ripple.models.multi_field_mlp import MultiFieldMLP
from ripple.core.experiment import Experiment

def main():
    # 1. Operators & Equations
    # Momentum X: -Laplacian(u) + Gradient_x(p) = 0
    # Momentum Y: -Laplacian(v) + Gradient_y(p) = 0
    # Continuity: Divergence(u, v) = 0
    
    # We need a custom operator to extract x and y components of Gradient
    class GradX(Gradient):
        def compute(self, field, params):
            return super().compute(field, params)[..., 0:1]
            
    class GradY(Gradient):
        def compute(self, field, params):
            return super().compute(field, params)[..., 1:2]
            
    # Continuity operator acting on a vector field (u, v)
    # Divergence in ripple currently expects a single field (which could be a vector)
    # But our MultiFieldMLP returns separate scalars 'u' and 'v'.
    # We need an operator that takes both.
    
    class MultiFieldDivergence(Divergence):
        """Div(u, v) = u_x + v_y"""
        def compute(self, field, params):
            u = params["fields"]["u"]
            v = params["fields"]["v"]
            inputs = params["inputs"]
            
            u_x = torch.autograd.grad(u.sum(), inputs, create_graph=True)[0][..., 0:1]
            v_y = torch.autograd.grad(v.sum(), inputs, create_graph=True)[0][..., 1:2]
            return u_x + v_y

    eq_x = Equation(terms=[
        (-1.0, Laplacian(field="u")),
        (1.0, GradX(field="p"))
    ])
    
    eq_y = Equation(terms=[
        (-1.0, Laplacian(field="v")),
        (1.0, GradY(field="p"))
    ])
    
    eq_div = Equation(terms=[
        (1.0, MultiFieldDivergence(field="u")) # Field arg is dummy here
    ])
    
    eq_system = EquationSystem(equations=[eq_x, eq_y, eq_div])

    # 2. Domain
    # (x, y) in [0, 1]x[0, 1]. No time here (steady).
    # But Domain expects time in the last dim currently.
    # We'll treat 'y' as 't' for the sake of the base class if needed, 
    # or just use 2D spatial.
    domain = Domain(
        spatial_dims=1, # 1 spatial dim 'x', 'y' as 't'
        bounds=((0, 1), (0, 1)),
        resolution=(20, 20)
    )

    # 3. System
    sys = System(
        equation=eq_system,
        domain=domain,
        fields=["u", "v", "p"],
        constraints=[] # Add BCs for specific problem
    )
    sys.validate()

    # 4. Model & Opt
    model = MultiFieldMLP(fields=["u", "v", "p"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    exp = Experiment(sys, model, opt)
    print("Stokes 2D System built and validated.")

if __name__ == "__main__":
    main()
