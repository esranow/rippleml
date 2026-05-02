import torch
from rippl.physics.equation import Equation
from rippl.core.equation_system import EquationSystem
from rippl.physics.operators import SchrodingerKinetic, PotentialTerm, SchrodingerTimeEvolution
from rippl.nn.multi_field_mlp import MultiFieldMLP

class SchrodingerSystem:
    def __init__(self, potential_fn, hbar=1.0, mass=1.0, dims=1):
        # potential_fn: V(coords) → (N,1)
        self.potential_fn = potential_fn
        self.hbar = hbar
        self.mass = mass
        self.dims = dims
    
    def build_equation_system(self) -> EquationSystem:
        # iℏ∂ψ/∂t = (-ℏ²/2m)∇²ψ + Vψ
        # split into real and imag parts:
        # SchrodingerTimeEvolution returns cat([-hbar*psi_imag_t, hbar*psi_real_t])
        # SchrodingerKinetic returns cat([-(hbar^2/2m)*psi_real_xx, -(hbar^2/2m)*psi_imag_xx])
        # PotentialTerm returns cat([V*psi_real, V*psi_imag])
        
        # We want: LHS - RHS = 0
        # Real: -hbar*psi_imag_t - [-(hbar^2/2m)*psi_real_xx + V*psi_real] = 0
        # Imag: hbar*psi_real_t - [-(hbar^2/2m)*psi_imag_xx + V*psi_imag] = 0
        
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

        lhs = SchrodingerTimeEvolution(hbar=self.hbar)
        kin = SchrodingerKinetic(hbar=self.hbar, mass=self.mass)
        pot = PotentialTerm(potential_fn=self.potential_fn)

        # Real equation: LHS_real - kin_real - pot_real = 0
        eq_real = Equation([
            (1.0, ComponentWrapper(lhs, 0)),
            (-1.0, ComponentWrapper(kin, 0)),
            (-1.0, ComponentWrapper(pot, 0))
        ])

        # Imag equation: LHS_imag - kin_imag - pot_imag = 0
        eq_imag = Equation([
            (1.0, ComponentWrapper(lhs, 1)),
            (-1.0, ComponentWrapper(kin, 1)),
            (-1.0, ComponentWrapper(pot, 1))
        ])

        return EquationSystem([eq_real, eq_imag])
    
    def norm_conservation_loss(self, model, coords_x) -> torch.Tensor:
        # ∫|ψ|² dx = 1 at each time slice
        # approximate integral via quadrature over coords_x
        # coords_x: (N, D)
        u_out = model(coords_x)
        fields = u_out if isinstance(u_out, dict) else {"psi_real": u_out[..., 0:1], "psi_imag": u_out[..., 1:2]}
        psi_real = fields["psi_real"]
        psi_imag = fields["psi_imag"]
        prob_density = psi_real**2 + psi_imag**2
        
        # Simple mean across spatial points (assuming uniform grid or representative sample)
        # For Square Well on [0,1], integral is approx mean() * (1-0)
        norm = prob_density.mean()
        return (norm - 1.0)**2
    
    def fields(self) -> list:
        return ["psi_real", "psi_imag"]
    
    def suggested_model(self) -> MultiFieldMLP:
        return MultiFieldMLP(
            in_dim=self.dims + 1, fields=["psi_real", "psi_imag"], hidden=64, layers=5
        )
