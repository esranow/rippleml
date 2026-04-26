import torch
import pytest
from ripple.physics.operators import Laplacian, Gradient, Divergence
from ripple.core.system import System, Domain, Constraint
from ripple.physics.equation import Equation
from ripple.core.equation_system import EquationSystem
from ripple.core.exceptions import RippleValidationError
from ripple.models.multi_field_mlp import MultiFieldMLP

def test_operator_signatures():
    """1. Test that all operators have correct signatures."""
    lap = Laplacian(field="v")
    sig = lap.signature()
    assert sig["inputs"] == ["v"]
    assert sig["output"] == "laplacian(v)"
    assert sig["order"] == 2
    assert sig["type"] == "spatial"
    
    grad = Gradient(field="p")
    sig = grad.signature()
    assert sig["inputs"] == ["p"]
    assert sig["output"] == "grad(p)"
    assert sig["order"] == 1

def test_system_validate_pass():
    """2. Test System.validate() with a consistent multi-field setup."""
    eq = Equation(terms=[(1.0, Laplacian(field="u"))])
    # spatial_dims=1 means bounds must be length 1
    dom = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=eq, domain=dom, fields=["u"])
    assert sys.validate() is True

def test_system_validate_fail():
    """3. Test System.validate() raises error for missing fields."""
    eq = Equation(terms=[(1.0, Laplacian(field="p"))]) # p not in fields
    dom = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=eq, domain=dom, fields=["u"])
    with pytest.raises(RippleValidationError) as excinfo:
        sys.validate()
    assert "field 'p'" in str(excinfo.value)

def test_equation_system_residuals():
    """4. Test EquationSystem.compute_residuals returns a list of tensors."""
    eq1 = Equation(terms=[(1.0, Laplacian(field="u"))])
    eq2 = Equation(terms=[(1.0, Laplacian(field="v"))])
    eq_sys = EquationSystem(equations=[eq1, eq2])
    
    coords = torch.randn(10, 1, requires_grad=True)
    fields = {
        "u": coords**2, # Connected to coords
        "v": coords**3
    }
    
    res = eq_sys.compute_residuals(fields, coords)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].shape == (10, 1)

def test_equation_system_loss_scalar():
    """5. Test EquationSystem.compute_loss returns a scalar tensor."""
    eq1 = Equation(terms=[(1.0, Laplacian(field="u"))])
    eq_sys = EquationSystem(equations=[eq1])
    
    coords = torch.randn(10, 1, requires_grad=True)
    fields = {"u": coords**2}
    
    loss = eq_sys.compute_loss(fields, coords)
    assert loss.dim() == 0 # scalar
    assert not torch.isnan(loss)

def test_field_shape_contract():
    """6. Test System.validate_fields enforcing shape contracts."""
    sys = System(equation=None, domain=None, fields=["u", "p"])
    
    # Valid shapes
    fields_ok = {"u": torch.randn(10, 1), "p": torch.randn(10, 1)}
    sys.validate_fields(fields_ok) # should not raise
    
    # Invalid field name
    fields_bad_name = {"w": torch.randn(10, 1)}
    with pytest.raises(RippleValidationError):
        sys.validate_fields(fields_bad_name)
        
    # Invalid shape (Phase 3 strict contract)
    fields_bad_shape = {"u": torch.randn(10, 2)}
    with pytest.raises(RippleValidationError) as excinfo:
        sys.validate_fields(fields_bad_shape)
    assert "trailing dimension 1" in str(excinfo.value)
