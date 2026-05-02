import torch
import numpy as np
import pytest
import warnings
import os
from rippl.core import System, Domain, Constraint, ReferenceScales, NondimSystem, NondimModelWrapper, DigitalTwin, InverseParameter
from rippl.physics.operators import ArtificialViscosity, Laplacian
from rippl.physics.shock import TVDScheme, add_artificial_viscosity
from rippl.physics.fractional import FractionalLaplacian, CaputoDerivative, FractionalSystem
from rippl.physics.hamilton_jacobi import EikonalOperator, HamiltonianOperator, HJSystem
from rippl.physics.phase_field import AllenCahnOperator, CahnHilliardOperator, PhaseFieldSystem
from rippl.physics.reaction_diffusion import ReactionDiffusionOperator, TuringSystem, FitzHughNagumoSystem, BrusselatorSystem
from rippl.physics.distance import BoxDistance, HardConstraintWrapper, NeumannAnsatzWrapper, MixedBCAnsatz, AnsatzFactory
from rippl.data.sensor import SensorDataset, MultiFidelityFusion
from rippl.core.equation import Equation
from rippl.core.equation_system import EquationSystem
from rippl.nn.multi_field_mlp import MultiFieldMLP
from rippl.core.exceptions import PhysicsModelWarning

# --- Feature 1: Non-dimensionalization ---

def test_reference_scales_defaults():
    rs = ReferenceScales()
    assert rs.L == 1.0
    assert rs.U == 1.0
    assert rs.T == 1.0
    assert rs.reynolds_number() >= 0.99 # close to 1.0

def test_normalize_coords_spatial():
    rs = ReferenceScales(L_ref=2.0)
    coords = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    norm = rs.normalize_coords(coords, has_time=False)
    assert torch.allclose(norm, torch.tensor([[0.5, 0.5]]))

def test_normalize_coords_time():
    rs = ReferenceScales(L_ref=2.0, T_ref=4.0)
    coords = torch.tensor([[1.0, 1.0]], dtype=torch.float32) # (x, t)
    norm = rs.normalize_coords(coords, has_time=True)
    assert torch.allclose(norm, torch.tensor([[0.5, 0.25]]))

def test_denormalize_roundtrip():
    rs = ReferenceScales(L_ref=2.5, U_ref=0.5, phi_ref=10.0)
    coords = torch.randn(10, 2)
    norm = rs.normalize_coords(coords)
    denorm = rs.denormalize_coords(norm)
    assert torch.allclose(coords, denorm)
    
    field = torch.randn(10, 1)
    norm_f = rs.normalize_field(field, "generic")
    denorm_f = rs.denormalize_field(norm_f, "generic")
    assert torch.allclose(field, denorm_f)

def test_nondim_wrapper_forward_shape():
    rs = ReferenceScales()
    model = lambda x: x[:, 0:1] * 2.0
    wrapper = NondimModelWrapper(model, rs, has_time=True)
    coords = torch.randn(5, 2)
    out = wrapper(coords)
    assert out.shape == (5, 1)

def test_large_coefficient_warning():
    eq = Equation([(2000.0, Laplacian(field="u"))]) # Coeff > 1e3
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=eq, domain=domain)
    with pytest.warns(PhysicsModelWarning, match="Large coefficient"):
        sys.validate()

# --- Feature 2: Artificial Viscosity ---

def test_av_zero_below_threshold():
    av = ArtificialViscosity(gradient_threshold=100.0, epsilon_max=0.1)
    coords = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    fields = {"u": 1.0 * coords} # Depends on coords but grad=1 < 100
    # grad is 1, below 100
    out = av.forward(fields, coords)
    assert torch.all(out == 0)

def test_av_positive_above_threshold():
    av = ArtificialViscosity(gradient_threshold=0.1, epsilon_max=0.1, spatial_dims=1)
    coords = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    fields = {"u": 10.0 * coords**2} # grad = 20x, Lap = 20
    out = av.forward(fields, coords)
    assert torch.any(out > 0)

def test_av_output_shape():
    av = ArtificialViscosity()
    coords = torch.randn(5, 2).requires_grad_(True)
    fields = {"u": torch.sum(coords**2, dim=-1, keepdim=True)}
    out = av.forward(fields, coords)
    assert out.shape == (5, 1)

def test_tvd_minmod_shape():
    tvd = TVDScheme(limiter="minmod")
    u = torch.randn(10, 1)
    out = tvd.apply(u, dx=0.1)
    assert out.shape == u.shape

def test_tvd_van_leer_shape():
    tvd = TVDScheme(limiter="van_leer")
    u = torch.randn(10, 1)
    out = tvd.apply(u, dx=0.1)
    assert out.shape == u.shape

# --- Feature 3: Fractional ---

def test_fractional_laplacian_shape():
    fl = FractionalLaplacian(alpha=0.5)
    fields = {"u": torch.randn(16, 1)}
    coords = torch.randn(16, 1)
    out = fl.forward(fields, coords)
    assert out.shape == (16, 1)

def test_fractional_laplacian_alpha_half():
    fl = FractionalLaplacian(alpha=0.5)
    fields = {"u": torch.ones(8, 1)}
    out = fl.forward(fields, None)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

def test_caputo_derivative_shape():
    cd = CaputoDerivative(alpha=0.5)
    derived = {"u_t": torch.randn(10, 1)}
    out = cd.forward(None, None, derived)
    assert out.shape == (10, 1)

def test_fractional_system_subdiffusion_builds():
    eq = FractionalSystem.subdiffusion(alpha=0.5)
    assert isinstance(eq, Equation)
    assert len(eq.terms) == 2

# --- Feature 4: Hamilton-Jacobi ---

def test_eikonal_operator_shape():
    op = EikonalOperator(spatial_dims=2)
    derived = {"V_x": torch.randn(5, 1), "V_y": torch.randn(5, 1)}
    coords = torch.randn(5, 3) # x, y, t
    out = op.forward(None, coords, derived)
    assert out.shape == (5, 1)

def test_eikonal_residual_analytic():
    op = EikonalOperator(spatial_dims=2)
    coords = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)
    derived = {"V_x": torch.tensor([[1.0/np.sqrt(2)]]), "V_y": torch.tensor([[1.0/np.sqrt(2)]])}
    res = op.forward(None, coords, derived)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-5)

def test_hamiltonian_operator_shape():
    H = lambda x, p: torch.sum(p**2, dim=-1, keepdim=True)
    op = HamiltonianOperator(H, spatial_dims=2)
    derived = {"V_t": torch.randn(5, 1), "V_x": torch.randn(5, 1), "V_y": torch.randn(5, 1)}
    coords = torch.randn(5, 3)
    out = op.forward(None, coords, derived)
    assert out.shape == (5, 1)

def test_hj_system_eikonal_builds():
    eq = HJSystem.eikonal()
    assert len(eq.terms) == 1
    assert isinstance(eq.terms[0][1], EikonalOperator)

# --- Feature 5: Phase Field ---

def test_allen_cahn_shape():
    op = AllenCahnOperator()
    fields = {"phi": torch.randn(10, 1)}
    derived = {"phi_t": torch.randn(10, 1), "phi_xx": torch.randn(10, 1)}
    out = op.forward(fields, None, derived)
    assert out.shape == (10, 1)

def test_allen_cahn_residual_analytic():
    op = AllenCahnOperator(M=1.0, epsilon=0.1)
    fields = {"phi": torch.zeros(5, 1)}
    derived = {"phi_t": torch.zeros(5, 1), "phi_xx": torch.zeros(5, 1)}
    res = op.forward(fields, None, derived)
    assert torch.allclose(res, torch.zeros_like(res))

def test_cahn_hilliard_shape():
    op = CahnHilliardOperator()
    fields = {"phi": torch.randn(10, 1), "mu": torch.randn(10, 1)}
    derived = {"phi_t": torch.randn(10, 1), "phi_xx": torch.randn(10, 1), "mu_xx": torch.randn(10, 1)}
    out = op.forward(fields, None, derived)
    assert out.shape == (10, 1)

def test_phase_field_system_builds():
    sys = PhaseFieldSystem.cahn_hilliard()
    assert isinstance(sys, EquationSystem)
    assert len(sys.equations) == 2

# --- Feature 6: Reaction Diffusion ---

def test_rd_operator_shape():
    R = lambda f, c: f["u"]**2
    op = ReactionDiffusionOperator(diffusivity=0.1, reaction_fn=R)
    fields = {"u": torch.randn(10, 1)}
    derived = {"u_t": torch.randn(10, 1), "u_xx": torch.randn(10, 1)}
    out = op.forward(fields, None, derived)
    assert out.shape == (10, 1)

def test_turing_system_builds():
    ts = TuringSystem()
    eqs = ts.build_equation_system()
    assert len(eqs.equations) == 2

def test_fitzhugh_nagumo_builds():
    fn = FitzHughNagumoSystem()
    eqs = fn.build_equation_system()
    assert len(eqs.equations) == 2

def test_brusselator_builds():
    br = BrusselatorSystem()
    eqs = br.build_equation_system()
    assert len(eqs.equations) == 2

# --- Feature 7: Ansatz ---

def test_dirichlet_1d_ansatz_bc_satisfied():
    model = lambda x: torch.ones(x.shape[0], 1)
    ansatz = AnsatzFactory.dirichlet_1d(model, a=1.0, b=2.0)
    c0 = torch.tensor([[0.0]])
    c1 = torch.tensor([[1.0]])
    assert torch.allclose(ansatz(c0), torch.tensor([[1.0]]))
    assert torch.allclose(ansatz(c1), torch.tensor([[2.0]]))

def test_neumann_ansatz_zero_flux():
    model = lambda x: torch.ones(x.shape[0], 1)
    D = lambda x: x * (1 - x)
    ansatz = NeumannAnsatzWrapper(model, D)
    out = ansatz(torch.tensor([[0.5]]))
    assert out.shape == (1, 1)

def test_mixed_bc_ansatz_shape():
    model = lambda x: torch.ones(x.shape[0], 1)
    g = lambda x: x
    D = lambda x: x
    ansatz = MixedBCAnsatz(model, g, D)
    out = ansatz(torch.tensor([[0.5]]))
    assert out.shape == (1, 1)

def test_ansatz_factory_dirichlet_1d():
    model = lambda x: x
    ansatz = AnsatzFactory.dirichlet_1d(model)
    assert isinstance(ansatz, MixedBCAnsatz)

# --- Feature 8: Digital Twin ---

def test_inverse_parameter_softplus_positive():
    p = InverseParameter("test", initial_value=0.5, transform="softplus")
    p._raw.data = torch.tensor([-10.0])
    assert p.get() > 0

def test_inverse_parameter_sigmoid_bounded():
    lo, hi = 0.1, 0.9
    p = InverseParameter("test", initial_value=0.5, bounds=(lo, hi), transform="sigmoid")
    p._raw.data = torch.tensor([100.0])
    assert p.get() <= hi + 1e-6
    p._raw.data = torch.tensor([-100.0])
    assert p.get() >= lo - 1e-6

def test_inverse_parameter_bounds_penalty_zero():
    p = InverseParameter("test", initial_value=0.5, bounds=(0, 1))
    assert p.bounds_penalty() == 0

def test_inverse_parameter_bounds_penalty_positive():
    p = InverseParameter("test", initial_value=1.5, bounds=(0, 1))
    p._raw.data = torch.tensor([1.5])
    assert p.bounds_penalty() > 0

def test_digital_twin_instantiates():
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=Equation([]), domain=domain)
    dt = DigitalTwin(sys, None, [], sensor_coords=torch.randn(5, 2))
    assert dt.w_data == 1.0

def test_digital_twin_from_csv():
    import pandas as pd
    df = pd.DataFrame({"x": [0.0, 1.0], "t": [0.0, 0.0], "u": [0.0, 0.0]})
    df.to_csv("test_sensor.csv", index=False)
    
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(10,))
    sys = System(equation=Equation([]), domain=domain)
    dt = DigitalTwin.from_csv(sys, None, [], "test_sensor.csv", ["x", "t"], {"u": "u"})
    assert dt.sensor_coords.shape == (2, 2)
    os.remove("test_sensor.csv")

def test_digital_twin_report_keys():
    p = InverseParameter("alpha", 0.1, units="m2/s")
    dt = DigitalTwin(None, None, [p])
    rep = dt.report()
    assert "alpha" in rep
    assert rep["alpha"]["units"] == "m2/s"

# --- Feature 9: Sensor Data ---

def test_sensor_dataset_from_numpy():
    coords = np.zeros((10, 2))
    fields = {"u": np.zeros((10, 1))}
    ds = SensorDataset.from_numpy(coords, fields)
    assert len(ds) == 10

def test_sensor_dataset_data_loss_shape():
    ds = SensorDataset(torch.randn(5, 2), {"u": torch.randn(5, 1)})
    model = lambda x: torch.randn(x.shape[0], 1)
    loss = ds.data_loss(model)
    assert loss.dim() == 0

def test_sensor_dataset_split_sizes():
    ds = SensorDataset(torch.randn(10, 2), {"u": torch.randn(10, 1)})
    tr, val = ds.split(0.8)
    assert len(tr) >= 7 and len(tr) <= 9 # randomness
    assert len(val) >= 1 and len(val) <= 3

def test_multifidelity_fusion_loss_scalar():
    ds1 = SensorDataset(torch.randn(5, 2), {"u": torch.randn(5, 1)}, fidelity=1.0)
    ds2 = SensorDataset(torch.randn(10, 2), {"u": torch.randn(10, 1)}, fidelity=0.1)
    fusion = MultiFidelityFusion([ds1, ds2])
    model = lambda x: torch.randn(x.shape[0], 1)
    loss = fusion.total_data_loss(model)
    assert loss.dim() == 0

def test_multifidelity_auto_balance():
    ds1 = SensorDataset(torch.randn(5, 2), {"u": torch.randn(5, 1)}, fidelity=0.5)
    ds2 = SensorDataset(torch.randn(5, 2), {"u": torch.randn(5, 1)}, fidelity=0.5)
    fusion = MultiFidelityFusion([ds1, ds2], auto_balance=True)
    model = lambda x: torch.zeros(x.shape[0], 1)
    loss = fusion.total_data_loss(model)
    assert loss >= 0

def test_fractional_system_superdiffusion_builds():
    eq = FractionalSystem.superdiffusion(alpha=0.5)
    assert isinstance(eq, Equation)

def test_hj_system_optimal_control_builds():
    L = lambda x: torch.zeros(x.shape[0], 1)
    f = lambda x: torch.zeros(x.shape[0], 2)
    eq = HJSystem.optimal_control(L, f)
    assert isinstance(eq, Equation)

def test_sensor_dataset_from_csv():
    import pandas as pd
    df = pd.DataFrame({"x": [0.0, 1.0], "t": [0.0, 0.0], "u": [0.0, 0.0]})
    df.to_csv("test_sensor_data.csv", index=False)
    ds = SensorDataset.from_csv("test_sensor_data.csv", ["x", "t"], {"u": "u"})
    assert len(ds) == 2
    os.remove("test_sensor_data.csv")

def test_sensor_dataset_iqr_filter():
    import pandas as pd
    df = pd.DataFrame({"u": [1.0, 1.1, 1.2, 100.0]}) # 100 is outlier
    df_filtered = SensorDataset._iqr_filter(df, ["u"])
    assert len(df_filtered) == 3
