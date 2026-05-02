import torch
import numpy as np
import pytest
import os
from rippl.training.uq import MCDropoutWrapper, DeepEnsemble, UncertaintyQuantifier, ProbabilisticExperiment
from rippl.training.fno_flywheel import FNOFlywheel
from rippl.nn.fno import FNO
from rippl.core.system import System, Domain
from rippl.core.equation import Equation
from rippl.core.operator_experiment import OperatorExperiment
from rippl.geometry.csg import Circle, Rectangle, Sphere, Box, Annulus, Union, Intersection, Difference, CSGSampler, CSGDomain

# UQ (10)
def test_mc_dropout_wrapper_output_shape():
    base = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    wrapper = MCDropoutWrapper(base, dropout_rate=0.1)
    x = torch.randn(10, 2)
    assert wrapper(x).shape == (10, 1)

def test_mc_dropout_uncertainty_nonzero():
    base = torch.nn.Sequential(torch.nn.Linear(2, 100), torch.nn.ReLU(), torch.nn.Linear(100, 1))
    wrapper = MCDropoutWrapper(base, dropout_rate=0.5)
    x = torch.randn(10, 2)
    res = wrapper.predict_with_uncertainty(x, n_samples=10)
    assert torch.any(res["std"] > 0)

def test_mc_dropout_samples_count():
    base = torch.nn.Linear(2, 1)
    wrapper = MCDropoutWrapper(base)
    res = wrapper.predict_with_uncertainty(torch.randn(5, 2), n_samples=25)
    assert res["samples"].shape == (25, 5, 1)

def test_deep_ensemble_output_shape():
    m1 = torch.nn.Linear(2, 1)
    m2 = torch.nn.Linear(2, 1)
    ensemble = DeepEnsemble([m1, m2])
    res = ensemble.predict_with_uncertainty(torch.randn(5, 2))
    assert res["mean"].shape == (5, 1)

def test_deep_ensemble_uncertainty_nonzero():
    m1 = torch.nn.Linear(1, 1)
    m2 = torch.nn.Linear(1, 1)
    m1.weight.data.fill_(1.0); m2.weight.data.fill_(2.0)
    ensemble = DeepEnsemble([m1, m2])
    res = ensemble.predict_with_uncertainty(torch.tensor([[1.0]]))
    assert res["std"].item() > 0

def test_uq_confidence_interval_bounds():
    uq = UncertaintyQuantifier(None)
    # Mock result
    res = {"mean": torch.tensor([1.0]), "std": torch.tensor([0.1])}
    uq.predict = lambda x, device: res
    ci = uq.confidence_interval(torch.tensor([0.0]))
    assert ci["lower"] < ci["mean"] < ci["upper"]

def test_uq_high_uncertainty_mask_bool():
    uq = UncertaintyQuantifier(None)
    res = {"mean": torch.tensor([1.0]), "std": torch.tensor([0.5])}
    uq.predict = lambda x, device: res
    mask = uq.high_uncertainty_regions(torch.tensor([0.0]), threshold=0.1)
    assert mask.dtype == torch.bool
    assert mask.item() == True

def test_probabilistic_experiment_mc_instantiates():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=Equation([]), domain=domain)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp = ProbabilisticExperiment(sys, model, method="mc_dropout", opt=opt)
    assert isinstance(exp.model, MCDropoutWrapper)

def test_probabilistic_experiment_ensemble_instantiates():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=Equation([]), domain=domain)
    model = torch.nn.Linear(2, 1)
    exp = ProbabilisticExperiment(sys, model, method="ensemble")
    assert exp.method == "ensemble"

def test_uncertainty_report_keys():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=Equation([]), domain=domain)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp = ProbabilisticExperiment(sys, model, method="mc_dropout", opt=opt)
    # Mock predict
    exp.predict_with_uncertainty = lambda x: {"mean": torch.ones(5, 1), "std": torch.ones(5, 1) * 0.1}
    rep = exp.uncertainty_report(torch.randn(5, 2))
    assert set(rep.keys()) == {"method", "mean_std", "max_std", "high_uncertainty_fraction"}

# FNO Flywheel (8)
def test_flywheel_instantiates():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(64,))
    sys = System(equation=Equation([]), domain=domain)
    fno = FNO(n_modes=4, width=8, input_dim=1)
    fw = FNOFlywheel(sys, fno)
    assert fw.n_train == 1000

def test_flywheel_generate_dataset_shapes():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(16,))
    sys = System(equation=Equation([]), domain=domain)
    fno = FNO(n_modes=4, width=8, input_dim=1)
    fw = FNOFlywheel(sys, fno, n_train=2, n_test=1)
    inputs, outputs = fw.generate_dataset()
    assert inputs.shape == (3, 16, 1)
    assert outputs.shape == (3, 16, 1)

def test_flywheel_dataset_cached():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(8,))
    sys = System(equation=Equation([]), domain=domain)
    fw = FNOFlywheel(sys, FNO(4, 4, 1), n_train=1, n_test=1)
    i1, o1 = fw.generate_dataset()
    i2, o2 = fw.generate_dataset()
    assert torch.allclose(i1, i2)

def test_flywheel_train_returns_history():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(8,))
    sys = System(equation=Equation([]), domain=domain)
    fw = FNOFlywheel(sys, FNO(4, 4, 1), n_train=2, n_test=1)
    hist = fw.train(epochs=1)
    assert "loss_history" in hist

def test_flywheel_evaluate_keys():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(8,))
    sys = System(equation=Equation([]), domain=domain)
    fw = FNOFlywheel(sys, FNO(4, 4, 1), n_train=2, n_test=1)
    fw.generate_dataset()
    ev = fw.evaluate()
    assert set(ev.keys()) == {"mean_l2", "max_l2", "min_l2"}

def test_flywheel_pipeline_runs():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(8,))
    sys = System(equation=Equation([]), domain=domain)
    fw = FNOFlywheel(sys, FNO(4, 4, 1), n_train=2, n_test=1)
    res = fw.pipeline(train_epochs=1)
    assert "mean_l2" in res

def test_operator_experiment_from_flywheel():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(8,))
    sys = System(equation=Equation([]), domain=domain)
    fw = FNOFlywheel(sys, FNO(4, 4, 1), n_train=2, n_test=1)
    fw.generate_dataset()
    exp = OperatorExperiment.from_flywheel(fw)
    assert isinstance(exp, OperatorExperiment)

def test_fno_forward_shape_post_flywheel():
    fno = FNO(n_modes=4, width=8, input_dim=1)
    x = torch.randn(5, 16, 1)
    out = fno(x)
    assert out.shape == (5, 16, 1)

# CSG (12)
def test_circle_contains_center():
    c = Circle(center=(0, 0), radius=1.0)
    assert c.contains(torch.tensor([[0.0, 0.0]])).item() == True

def test_circle_excludes_exterior():
    c = Circle(center=(0, 0), radius=1.0)
    assert c.contains(torch.tensor([[2.0, 0.0]])).item() == False

def test_rectangle_contains_interior():
    r = Rectangle(0, 1, 0, 1)
    assert r.contains(torch.tensor([[0.5, 0.5]])).item() == True

def test_sphere_contains_center():
    s = Sphere(center=(0, 0, 0), radius=1.0)
    assert s.contains(torch.tensor([[0.0, 0.0, 0.0]])).item() == True

def test_annulus_excludes_interior_hole():
    a = Annulus(r_inner=0.5, r_outer=1.0)
    assert a.contains(torch.tensor([[0.0, 0.0]])).item() == False

def test_union_contains_both():
    c1 = Circle(center=(-1, 0), radius=1.0)
    c2 = Circle(center=(1, 0), radius=1.0)
    u = c1 | c2
    assert u.contains(torch.tensor([[-1.0, 0.0]])).item() == True
    assert u.contains(torch.tensor([[1.0, 0.0]])).item() == True

def test_intersection_contains_overlap_only():
    c1 = Circle(center=(-0.5, 0), radius=1.0)
    c2 = Circle(center=(0.5, 0), radius=1.0)
    i = c1 & c2
    assert i.contains(torch.tensor([[0.0, 0.0]])).item() == True
    assert i.contains(torch.tensor([[1.2, 0.0]])).item() == False

def test_difference_excludes_subtracted():
    c1 = Circle(center=(0, 0), radius=1.0)
    c2 = Circle(center=(0.5, 0), radius=0.5)
    d = c1 - c2
    assert d.contains(torch.tensor([[0.5, 0.0]])).item() == False
    assert d.contains(torch.tensor([[-0.5, 0.0]])).item() == True

def test_csg_sampler_interior_shape():
    s = CSGSampler(Circle())
    pts = s.sample_interior(10)
    assert pts.shape == (10, 2)

def test_csg_sampler_volume_estimate_circle():
    s = CSGSampler(Circle(radius=1.0))
    vol = s.estimate_volume(n_total=10000)
    assert abs(vol - np.pi) < 0.1 * np.pi # Within 10%

def test_csg_domain_to_collocation_shape():
    d = CSGDomain(Circle(), spatial_dims=2)
    pts = d.to_collocation_points(50, has_time=False)
    assert pts.shape == (50, 2)

def test_csg_domain_with_time_shape():
    d = CSGDomain(Circle(), spatial_dims=2)
    pts = d.to_collocation_points(50, has_time=True)
    assert pts.shape == (50, 3)
