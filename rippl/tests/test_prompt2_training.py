import torch
import pytest
import numpy as np
import json
import warnings
from rippl.training.causal import CausalTrainingMixin
from rippl.physics.conservative import StreamFunctionModel, VectorPotentialModel, verify_divergence_free
from rippl.training.ntk_weighting import GradientNormWeighting, NTKDiagonalWeighting, AdaptiveLossBalancer
from rippl.physics.conservation import MassConservation, EnergyConservation, ConservationLaw
from rippl.diagnostics.physics_validator import PhysicsValidator
from rippl.core.system import System, Domain, Constraint
from rippl.core.experiment import Experiment
from rippl.core.equation import Equation
from rippl.core.equation import Equation

# Mocks
class MockOp:
    def signature(self):
        return {"inputs": ["u"], "output": "u", "type": "mock", "requires_derived": []}
    def compute(self, field, params):
        return field * 0.0
    def forward(self, fields, coords, derived=None):
        return fields["u"] * 0.0

class MockCausal(CausalTrainingMixin):
    pass

def test_causal_binned_shape():
    mixin = MockCausal()
    coords = torch.randn(100, 2) # last dim is time
    residuals = torch.randn(100, 1)
    weights = mixin.compute_causal_weights_binned(coords, residuals, n_bins=5)
    assert weights.shape == residuals.shape

def test_causal_continuous_shape():
    mixin = MockCausal()
    coords = torch.randn(100, 2)
    residuals = torch.randn(100, 1)
    weights = mixin.compute_causal_weights_continuous(coords, residuals)
    assert weights.shape == residuals.shape

def test_causal_early_weights_higher():
    mixin = MockCausal()
    coords = torch.zeros(10, 2)
    coords[:, 1] = torch.linspace(0, 1, 10) # time
    residuals = torch.ones(10, 1) # constant error
    weights = mixin.compute_causal_weights_continuous(coords, residuals, epsilon=1.0)
    # w_i = exp(-sum_{j<i} r_j^2)
    # w0 = exp(0) = 1
    # w1 = exp(-1)
    assert weights[0] > weights[-1]

def test_causal_epsilon_in_range():
    mixin = MockCausal()
    res = torch.ones(10, 1) * 100.0
    eps = mixin.optimal_epsilon(res)
    assert 0.1 <= eps <= 100.0

def test_causal_weights_detached():
    mixin = MockCausal()
    coords = torch.randn(10, 2, requires_grad=True)
    residuals = torch.randn(10, 1)
    weights = mixin.compute_causal_weights_continuous(coords, residuals)
    assert not weights.requires_grad

def test_causal_experiment_runs():
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)], resolution=(10,))
    eq = Equation([(1.0, MockOp())])
    system = System(eq, domain)
    model = lambda x: {"u": x[:, 0:1]}
    opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
    exp = Experiment(system, model, opt, causal_training=True, causal_mode="binned")
    res = exp.train(torch.randn(10, 2), epochs=1)
    assert "loss" in res

# Conservative
def test_stream_divergence_free():
    base = lambda x: x[:, 0:1] * x[:, 1:2] # psi = x*y
    model = StreamFunctionModel(base)
    coords = torch.randn(10, 2, requires_grad=True)
    res = verify_divergence_free(model, coords)
    assert res["passed"]

def test_stream_output_keys():
    base = lambda x: x[:, 0:1]
    model = StreamFunctionModel(base)
    out = model(torch.randn(10, 2))
    assert set(out.keys()) == {"u", "v", "psi"}

def test_vector_potential_divergence_free():
    # A = (y, z, x) -> curl(A) = (∂Az/∂y - ∂Ay/∂z, ∂Ax/∂z - ∂Az/∂x, ∂Ay/∂x - ∂Ax/∂y)
    # = (0 - 1, 0 - 1, 0 - 1) = (-1, -1, -1)
    # div(curl(A)) = 0
    base = lambda x: torch.stack([x[:, 1], x[:, 2], x[:, 0]], dim=-1)
    model = VectorPotentialModel(base)
    coords = torch.randn(10, 3, requires_grad=True)
    res = verify_divergence_free(model, coords)
    assert res["passed"]

def test_vector_potential_output_keys():
    base = lambda x: torch.randn(x.shape[0], 3)
    model = VectorPotentialModel(base)
    out = model(torch.randn(10, 3))
    assert set(out.keys()) == {"u", "v", "w", "A"}

def test_verify_divergence_free_passes():
    model = lambda x: {"u": torch.zeros(x.shape[0], 1), "v": torch.zeros(x.shape[0], 1)}
    res = verify_divergence_free(model, torch.randn(10, 2))
    assert res["passed"]

# NTK
def test_gradient_norm_initial_weights_one():
    gnw = GradientNormWeighting(["pde", "const"])
    assert gnw.weights["pde"] == 1.0
    assert gnw.weights["const"] == 1.0

def test_gradient_norm_apply_scalar():
    gnw = GradientNormWeighting(["pde"])
    loss_dict = {"pde": torch.tensor(2.0)}
    total = gnw.apply(loss_dict)
    assert total.item() == 2.0

def test_gradient_norm_updates_weights():
    gnw = GradientNormWeighting(["pde"])
    model = torch.nn.Linear(2, 1)
    loss_dict = {"pde": torch.tensor(2.0, requires_grad=True)}
    total_loss = torch.tensor(5.0, requires_grad=True)
    gnw.update(model, loss_dict, total_loss)
    assert gnw.weights["pde"] != 1.0

def test_ntk_diagonal_apply_scalar():
    ntk = NTKDiagonalWeighting(["pde"])
    loss_dict = {"pde": torch.tensor(3.0)}
    assert ntk.apply(loss_dict).item() == 3.0

def test_adaptive_balancer_gradient_norm_mode():
    balancer = AdaptiveLossBalancer(mode="gradient_norm", loss_names=["pde"])
    assert isinstance(balancer.balancer, GradientNormWeighting)

def test_adaptive_balancer_ntk_mode():
    balancer = AdaptiveLossBalancer(mode="ntk", loss_names=["pde"])
    assert isinstance(balancer.balancer, NTKDiagonalWeighting)

def test_adaptive_loss_experiment_runs():
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)], resolution=(10,))
    eq = Equation([(1.0, MockOp())])
    system = System(eq, domain)
    # Use a real model for NTK/Gradient computation
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 1)
        def forward(self, x):
            return {"u": self.model(x)}
    
    model = Wrapper()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp = Experiment(system, model, opt, adaptive_loss=True)
    res = exp.train(torch.randn(10, 2), epochs=1)
    assert "loss" in res

# Conservation
def test_mass_conservation_reference():
    mc = MassConservation(field="u")
    model = lambda x: torch.ones(x.shape[0], 1)
    mc.set_reference(model, torch.randn(10, 1))
    assert mc.reference == 1.0

def test_mass_conservation_penalty_zero():
    mc = MassConservation(field="u")
    model = lambda x: torch.ones(x.shape[0], 1)
    mc.set_reference(model, torch.randn(10, 1))
    p = mc.penalty(model, torch.randn(10, 1))
    assert p.item() == 0.0

def test_conservation_is_satisfied():
    mc = MassConservation(field="u", tolerance=0.1)
    model = lambda x: torch.ones(x.shape[0], 1)
    mc.set_reference(model, torch.randn(10, 1))
    # current = 1.05
    model2 = lambda x: torch.ones(x.shape[0], 1) * 1.05
    assert mc.is_satisfied(model2, torch.randn(10, 1))

def test_energy_conservation_drift_detected():
    ec = EnergyConservation(energy_fn=lambda m, c: torch.mean(m(c)**2))
    model = lambda x: torch.ones(x.shape[0], 1)
    ec.set_reference(model, torch.randn(10, 1))
    model2 = lambda x: torch.ones(x.shape[0], 1) * 2.0
    assert not ec.is_satisfied(model2, torch.randn(10, 1))

# Validator
def test_validator_residual_stats_keys():
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)], resolution=(10,))
    eq = Equation([(1.0, MockOp())])
    system = System(eq, domain)
    model = lambda x: torch.zeros(x.shape[0], 1)
    validator = PhysicsValidator(system, model, torch.randn(10, 1))
    stats = validator.residual_stats()
    assert set(stats.keys()) == {"mean", "max", "std", "l2", "passed"}

def test_validator_full_report_runs():
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)], resolution=(10,))
    eq = Equation([(1.0, MockOp())])
    system = System(eq, domain)
    model = lambda x: torch.zeros(x.shape[0], 1)
    validator = PhysicsValidator(system, model, torch.randn(10, 1))
    report = validator.full_report()
    assert "residuals" in report

def test_validator_export_json(tmp_path):
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)], resolution=(10,))
    eq = Equation([(1.0, MockOp())])
    system = System(eq, domain)
    model = lambda x: torch.zeros(x.shape[0], 1)
    validator = PhysicsValidator(system, model, torch.randn(10, 1))
    path = tmp_path / "report.json"
    validator.export_report(str(path))
    assert path.exists()


