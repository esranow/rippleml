import torch
from rippl.core.equation_system import EquationSystem
from rippl.physics.equation import Equation
from rippl.physics.operators import ElasticEquilibrium
from rippl.nn.multi_field_mlp import MultiFieldMLP

class LinearElasticitySystem:
    def __init__(self, E=1.0, nu=0.3, dims=2,
                 body_force_x=0.0, body_force_y=0.0):
        # E: Young's modulus, nu: Poisson's ratio
        # convert to Lamé parameters:
        # λ = E*nu / ((1+nu)*(1-2*nu))
        # μ = E / (2*(1+nu))
        self.lam = E * nu / ((1+nu) * (1-2*nu))
        self.mu = E / (2*(1+nu))
        self.fx = body_force_x
        self.fy = body_force_y
        self.dims = dims
    
    def build_equation_system(self) -> EquationSystem:
        # ElasticEquilibrium handles both x and y equilibrium 
        # but EquationSystem expects equations that return residuals.
        # Since ElasticEquilibrium returns a concatenated (N, 2) tensor, 
        # we can split it into two equations or use it as is if Equation handles it.
        # However, the contract usually expects one Equation per residual dimension.
        
        class ElasticComponent(torch.nn.Module):
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

        op = ElasticEquilibrium(
            lame_lambda=self.lam, lame_mu=self.mu,
            body_force_x=self.fx, body_force_y=self.fy
        )
        
        eq_x = Equation([(1.0, ElasticComponent(op, 0))])
        eq_y = Equation([(1.0, ElasticComponent(op, 1))])
        
        return EquationSystem([eq_x, eq_y])
    
    def fields(self) -> list:
        return ["ux", "uy"]
    
    def suggested_model(self) -> MultiFieldMLP:
        return MultiFieldMLP(in_dim=self.dims, fields=["ux", "uy"], hidden=64, layers=5)
