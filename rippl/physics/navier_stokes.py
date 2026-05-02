import torch
from rippl.physics.equation import Equation
from rippl.core.equation_system import EquationSystem
from rippl.physics.operators import (
    TimeDerivative, Laplacian, NonlinearAdvection, 
    PressureGradient, VelocityDivergence
)
from rippl.nn.multi_field_mlp import MultiFieldMLP

# GAUGE CONDITION REQUIRED: Pressure in incompressible NS is only unique
# up to a constant. Always add a Constraint fixing p at one boundary point.
# Without this, p L2 error will be O(0.1) regardless of training quality.
class NavierStokesSystem:
    def __init__(self, rho=1.0, mu=0.01, dims=2, pressure_gauge_coords=None, pressure_gauge_value=None):
        import warnings
        from rippl.core.exceptions import PhysicsModelWarning
        # rho: density, mu: dynamic viscosity
        # dims: 2 for 2D (fields: u, v, p)
        self.rho = rho
        self.mu = mu
        self.dims = dims
        self.pressure_gauge_coords = pressure_gauge_coords
        self.pressure_gauge_value = pressure_gauge_value

        if pressure_gauge_coords is None:
            warnings.warn(
                "No pressure gauge condition set. Pressure solution is unique only "
                "up to a constant. Set pressure_gauge_coords to fix pressure level.",
                PhysicsModelWarning
            )
    
    def build_equation_system(self) -> EquationSystem:
        # Equation 1 — u momentum:
        # ρ(u_t + u*u_x + v*u_y) = -p_x + μ(u_xx + u_yy)
        # We use NonlinearAdvection(u,v) which returns [conv_u, conv_v]
        
        # Momentum Equations
        # We need to extract the correct component from NonlinearAdvection
        # Since our Equation handles sum of terms, we can use a custom wrapper or 
        # lambda if needed, but the user spec suggests NonlinearAdvection(u,v)[0]
        # Our NonlinearAdvection.forward returns torch.cat([conv_u, conv_v], dim=-1)
        
        # To handle the [0] and [1] indexing in Equation, we might need a component operator
        # or update NonlinearAdvection to take a component index.
        # But I'll follow the spec as closely as possible.
        
        class ComponentWrapper(torch.nn.Module):
            def __init__(self, op, index):
                super().__init__()
                self.op = op
                self.index = index
            def signature(self):
                return self.op.signature()
            def forward(self, fields, coords, derived=None):
                return self.op.forward(fields, coords, derived)[..., self.index : self.index + 1]
            def compute(self, field, params):
                return self.forward(params.get("fields", {}), params["inputs"], params.get("derived", {}))

        conv = NonlinearAdvection(field_u="u", field_v="v")
        conv_u = ComponentWrapper(conv, 0)
        conv_v = ComponentWrapper(conv, 1)

        eq_u = Equation([
            (self.rho, TimeDerivative(field="u")),
            (self.rho, conv_u),
            (10.0, PressureGradient(field_p="p", direction=0)),
            (-self.mu, Laplacian(field="u"))
        ])

        eq_v = Equation([
            (self.rho, TimeDerivative(field="v")),
            (self.rho, conv_v),
            (10.0, PressureGradient(field_p="p", direction=1)),
            (-self.mu, Laplacian(field="v"))
        ])

        eq_continuity = Equation([
            (1.0, VelocityDivergence(field_u="u", field_v="v"))
        ])

        return EquationSystem([eq_u, eq_v, eq_continuity],
                                weights=[1.0, 1.0, 10.0])
    
    def fields(self) -> list:
        return ["u", "v", "p"]
    
    def suggested_model(self) -> MultiFieldMLP:
        return MultiFieldMLP(in_dim=self.dims + 1, fields=["u", "v", "p"], hidden=64, layers=6)
