import torch
import pytest
import warnings
from rippl.training.pinn_recipe import PINNTrainingRecipe
from rippl.training.lbfgs_config import LBFGSConfig
from rippl.physics.navier_stokes import NavierStokesSystem
from rippl.core.exceptions import PhysicsModelWarning

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.param * x

def test_pinn_recipe_instantiates():
    model = MockModel()
    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=lambda: torch.tensor(1.0, requires_grad=True),
        constraint_loss_fn=lambda: (torch.tensor(0.5, requires_grad=True), {}),
        device=torch.device("cpu")
    )
    assert recipe.phase_a_epochs == 3000

def test_pinn_recipe_phase_a_runs():
    model = MockModel()
    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=lambda: torch.tensor(1.0, requires_grad=True),
        constraint_loss_fn=lambda: (torch.tensor(0.5, requires_grad=True), {}),
        device=torch.device("cpu"),
        phase_a_epochs=10,
        verbose=False
    )
    loss = recipe._phase_a()
    assert isinstance(loss, float)

def test_pinn_recipe_phase_b_runs():
    model = MockModel()
    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=lambda: torch.tensor(1.0, requires_grad=True),
        constraint_loss_fn=lambda: (torch.tensor(0.5, requires_grad=True), {}),
        device=torch.device("cpu"),
        phase_b_epochs=10,
        verbose=False
    )
    loss, epochs = recipe._phase_b()
    assert epochs == 10
    assert isinstance(loss, float)

def test_pinn_recipe_phase_c_runs():
    model = MockModel()
    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=lambda: torch.tensor(1.0, requires_grad=True),
        constraint_loss_fn=lambda: (torch.tensor(0.5, requires_grad=True), {}),
        device=torch.device("cpu"),
        lbfgs_steps=5,
        verbose=False
    )
    loss = recipe._phase_c()
    assert isinstance(loss, float)

def test_dynamic_handoff_triggers():
    model = MockModel()
    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=lambda: torch.tensor(1.0, requires_grad=True),
        constraint_loss_fn=lambda: (torch.tensor(0.5, requires_grad=True), {}),
        device=torch.device("cpu")
    )
    # Flat loss history
    history = [1.0] * 300
    assert bool(recipe._dynamic_handoff_check(history, window=200)) is True

def test_dynamic_handoff_no_trigger():
    model = MockModel()
    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=lambda: torch.tensor(1.0, requires_grad=True),
        constraint_loss_fn=lambda: (torch.tensor(0.5, requires_grad=True), {}),
        device=torch.device("cpu")
    )
    # Decreasing loss
    history = [float(i) for i in range(300, 0, -1)]
    assert bool(recipe._dynamic_handoff_check(history, window=200)) is False

def test_lbfgs_config_heat():
    config = LBFGSConfig.for_pde("heat")
    assert config == LBFGSConfig.TIGHT
    assert config["max_iter"] == 50

def test_lbfgs_config_default():
    config = LBFGSConfig.for_pde("unknown")
    assert config == LBFGSConfig.STANDARD
    assert config["max_iter"] == 20

def test_pressure_gauge_warning():
    with pytest.warns(PhysicsModelWarning, match="No pressure gauge condition set"):
        NavierStokesSystem(dims=2)

def test_pressure_gauge_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        NavierStokesSystem(dims=2, pressure_gauge_coords=torch.zeros(1, 2))
        # Filter for PhysicsModelWarning only
        physics_warnings = [warning for warning in w if issubclass(warning.category, PhysicsModelWarning)]
        assert len(physics_warnings) == 0
