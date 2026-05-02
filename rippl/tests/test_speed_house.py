import pytest
import torch
import torch.nn as nn
from typing import Any

import rippl as rp
from rippl.core.system import Domain
from rippl.core.nondim import AutoScaler
from rippl.core.api import _run_native

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)

class DummyEquation:
    characteristic_velocity = 2.0
    def compute_loss(self, preds, coords):
        return torch.tensor(0.5, requires_grad=True)

def test_compile_returns_module():
    model = MLP(2, 1)
    compiled = rp.compile(model)
    assert isinstance(compiled, nn.Module)

def test_compile_fallback_unsupported_backend():
    model = MLP(2, 1)
    # Give a dummy backend that doesn't exist
    compiled = rp.compile(model, backend="non_existent_backend_xyz123")
    assert isinstance(compiled, nn.Module)

def test_autoscaler_from_domain_equation():
    domain = Domain(spatial_dims=2, bounds=((-5.0, 5.0), (0.0, 2.0)), resolution=(10, 10))
    equation = DummyEquation()
    scaler = AutoScaler.from_domain_equation(domain, equation)
    
    assert scaler.L0 == 10.0  # max(5 - (-5), 2 - 0) = 10.0
    assert scaler.U0 == 2.0
    assert scaler.T0 == 10.0 / 2.0  # L0 / U0

def test_autoscaler_scale_inputs_outputs():
    domain = Domain(spatial_dims=1, bounds=((-5.0, 5.0), (0.0, 2.0)), resolution=(10, 10))
    equation = DummyEquation()
    scaler = AutoScaler.from_domain_equation(domain, equation)
    
    # coords shape (10, 2)
    coords = torch.ones(10, 2) * 10.0
    scaled_inputs = scaler.scale_inputs(coords)
    
    assert scaled_inputs.shape == (10, 2)
    assert torch.allclose(scaled_inputs[:, 0], torch.tensor(1.0)) # 10.0 / 10.0
    assert torch.allclose(scaled_inputs[:, 1], torch.tensor(2.0)) # 10.0 / 5.0

    outputs = torch.ones(10, 1)
    scaled_outputs = scaler.scale_outputs(outputs)
    assert scaled_outputs.shape == (10, 1)
    assert torch.allclose(scaled_outputs, torch.tensor(2.0))

def test_autoscaler_get_state():
    domain = Domain(spatial_dims=1, bounds=((-5.0, 5.0),), resolution=(10,))
    equation = DummyEquation()
    scaler = AutoScaler.from_domain_equation(domain, equation)
    state = scaler.get_state()
    assert state == {"L0": 10.0, "U0": 2.0, "T0": 5.0}

def test_domain_generate_loader():
    domain = Domain(spatial_dims=2, bounds=((-5.0, 5.0), (0.0, 2.0)), resolution=(10, 10))
    loader = domain.generate_loader(batch_size=128)
    
    batch = next(iter(loader))
    coords = batch[0]
    
    # Check shape
    assert coords.shape[0] == 128
    assert coords.shape[1] == 2
    
    # Check bounds
    assert coords[:, 0].min() >= -5.0
    assert coords[:, 0].max() <= 5.0
    assert coords[:, 1].min() >= 0.0
    assert coords[:, 1].max() <= 2.0

try:
    import pytorch_lightning as pl
    from rippl.training.lightning_engine import LightningEngine
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False

@pytest.mark.skipif(not HAS_LIGHTNING, reason="pytorch-lightning not installed")
def test_lightning_engine_instantiates():
    model = MLP(2, 1)
    equation = DummyEquation()
    domain = Domain(spatial_dims=1, bounds=((-1.0, 1.0), (0.0, 1.0)), resolution=(10, 10))
    scaler = AutoScaler.from_domain_equation(domain, equation)
    
    engine = LightningEngine(model=model, equation=equation, scaler=scaler)
    assert isinstance(engine, pl.LightningModule)

@pytest.mark.skipif(not HAS_LIGHTNING, reason="pytorch-lightning not installed")
def test_lightning_engine_training_step():
    model = MLP(2, 1)
    equation = DummyEquation()
    domain = Domain(spatial_dims=1, bounds=((-1.0, 1.0), (0.0, 1.0)), resolution=(10, 10))
    scaler = AutoScaler.from_domain_equation(domain, equation)
    
    engine = LightningEngine(model=model, equation=equation, scaler=scaler)
    engine.trainer = pl.Trainer(accelerator="cpu", strategy="auto", devices=1)
    
    # dummy batch
    batch = [torch.rand(10, 2)]
    
    # Initialize optimizers to set them up
    engine.configure_optimizers()
    
    # Just verify it runs without crashing
    # Note: training_step expects self.optimizers() to return (adam, lbfgs)
    # However, trainer setup is complex in tests, so we manually inject them
    adam, lbfgs = engine.configure_optimizers()
    engine.optimizers = lambda: (adam, lbfgs)
    
    engine.training_step(batch, batch_idx=0)
    assert engine.final_loss is not None

def test_run_native_fallback():
    model = MLP(2, 1)
    equation = DummyEquation()
    domain = Domain(spatial_dims=1, bounds=((-1.0, 1.0), (0.0, 1.0)), resolution=(10, 10))
    
    res = _run_native(domain, equation, model, epochs=2, kwargs={"batch_size": 16})
    assert "model_state" in res
    assert "scaler_state" in res
    assert "final_loss" in res

def test_top_level_exports():
    assert hasattr(rp, "compile")
    assert hasattr(rp, "run")
