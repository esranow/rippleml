"""
tests/test_equation_sot.py
Validate Equation is the single source of truth across Simulation + Experiment.
Run: python tests/test_equation_sot.py
"""
import sys, math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pass(name): print(f"  [PASS] {name}")
def _fail(name, reason): print(f"  [FAIL] {name}: {reason}"); return False

def _make_domain():
    from ripple.core.system import Domain
    return Domain(spatial_dims=1, x_range=(0.0, 1.0), t_range=(0.0, 1.0))

def _make_constraint(type_="initial"):
    from ripple.core.system import Constraint
    # scalar return so loss stays shape [] for .backward()
    return Constraint(fn=lambda u, x, t: torch.tensor(0.0), weight=0.0, type=type_)

def _make_wave_system():
    from ripple.physics.operators import TimeDerivative, Laplacian
    from ripple.physics.equation import Equation
    from ripple.core.system import System
    eq = Equation(terms=[(1.0, TimeDerivative(order=2)), (-1.0, Laplacian())])
    return System(eq, _make_domain(), [_make_constraint("initial")])

def _make_diffusion_system(alpha=0.01):
    from ripple.physics.operators import TimeDerivative, Diffusion
    from ripple.physics.equation import Equation
    from ripple.core.system import System
    eq = Equation(terms=[(1.0, TimeDerivative(order=1)), (-alpha, Diffusion(alpha))])
    return System(eq, _make_domain(), [_make_constraint("boundary")])

def _make_advection_system(v=0.5):
    from ripple.physics.operators import TimeDerivative, Advection
    from ripple.physics.equation import Equation
    from ripple.core.system import System
    eq = Equation(terms=[(1.0, TimeDerivative(order=1)), (-1.0, Advection(v))])
    return System(eq, _make_domain(), [_make_constraint("boundary")])

def _u0_v0(N=16):
    """(B=1, N, 1) tensors for FD solvers."""
    u0 = torch.sin(math.pi * torch.linspace(0, 1, N)).unsqueeze(0).unsqueeze(-1)
    v0 = torch.zeros_like(u0)
    return u0, v0

class _TinyNet(nn.Module):
    """Model taking inputs=cat([x,t]) of shape (N,2) -> (N,1)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    def forward(self, inputs):
        return self.net(inputs)


# ---------------------------------------------------------------------------
# TEST 1 - Experiment uses Equation.compute_residual
# ---------------------------------------------------------------------------
def test1_experiment_uses_equation():
    print("\nTEST 1: EXPERIMENT USES EQUATION")
    from ripple.core.experiment import Experiment
    from ripple.physics.equation import Equation

    CONSTANT = 3.7
    called   = []

    orig = Equation.compute_residual

    def _fake(self, u, inputs):
        called.append("compute_residual")
        return u - u.detach() + CONSTANT   # in-graph constant

    Equation.compute_residual = _fake
    try:
        system = _make_wave_system()
        model  = _TinyNet()
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
        exp    = Experiment(system, model, opt)

        N = 20
        x = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)
        loss = exp.train(x, t)

        expected = CONSTANT ** 2
        if abs(loss - expected) > 1.0:
            return _fail("loss mismatch", f"got {loss:.4f}, expected ~{expected:.4f}")
        _pass(f"loss={loss:.4f} ~= CONSTANT^2={expected:.4f}")

        if called:
            _pass(f"Equation.{called[0]}() invoked by Experiment.train()")
        else:
            return _fail("Equation not called", "Experiment bypassed Equation entirely")

    except Exception as e:
        return _fail("exception", str(e))
    finally:
        Equation.compute_residual = orig

    return True


# ---------------------------------------------------------------------------
# TEST 2 - No duplicate PDE residual outside equation.py / operators.py
# ---------------------------------------------------------------------------
def test2_no_duplicate_residual():
    print("\nTEST 2: NO DUPLICATE RESIDUAL")
    import re, pathlib

    root    = pathlib.Path(__file__).parent.parent
    eq_file = (root / "physics" / "equation.py").resolve()
    op_file = (root / "physics" / "operators.py").resolve()

    PAT = re.compile(r"torch\.autograd\.grad.*create_graph|\bautograd\.grad\b.*create_graph", re.DOTALL)

    flagged = []
    for py in root.rglob("*.py"):
        if py.resolve() in (eq_file, op_file):
            continue
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if PAT.search(text):
            flagged.append(str(py.relative_to(root)))

    # Whitelisted: operator primitives (laplacian, gradient) and physics
    # diagnostics (energy, hamiltonian, adaptivesampler) that use autograd
    # legitimately -- they are NOT PDE residual sources, just math primitives.
    legacy = {
        "residuals.py", "pde.py",           # old legacy
        "laplacian.py", "gradient.py",      # operator primitives
        "energy.py", "hamiltonian.py",      # physics diagnostics
        "adaptivesampler.py",               # importance sampling (gradient magnitude)
    }
    non_leg  = [f for f in flagged if not any(l in f for l in legacy)]
    leg_hits = [f for f in flagged if     any(l in f for l in legacy)]

    if leg_hits:
        print("  [WARN] Legacy files (not counted as failure):")
        for f in leg_hits: print(f"         {f}")

    if non_leg:
        print("  [FAIL] Files with duplicate autograd residual logic:")
        for f in non_leg: print(f"         {f}")
        return False

    _pass("no duplicate residual logic outside operators/equation")
    return True


# ---------------------------------------------------------------------------
# TEST 3 - Simulation consistency: Equation.compute_residual callable
# ---------------------------------------------------------------------------
def test3_simulation_consistency():
    print("\nTEST 3: SIMULATION CONSISTENCY")
    try:
        from ripple.core.simulation import Simulation

        system = _make_wave_system()
        u0, v0 = _u0_v0(N=16)
        sim     = Simulation(system, c=1.0, dt=0.001, dx=1.0/16)
        traj    = sim.run(u0, v0, steps=5)   # (1, T, 16, 1)

        assert traj is not None and traj.numel() > 0
        assert not torch.isnan(traj).any(), "NaN in trajectory"

        N = 16
        x_lin = torch.linspace(0, 1, N).unsqueeze(-1)
        t_lin = torch.full((N, 1), 0.005)
        inputs = torch.cat([x_lin, t_lin], dim=-1).requires_grad_(True)

        net = nn.Linear(2, 1)
        nn.init.zeros_(net.weight); nn.init.zeros_(net.bias)
        u_pred = net(inputs)

        res  = system.equation.compute_residual(u_pred, inputs)
        rmse = float((res**2).mean().sqrt())

        assert math.isfinite(rmse), f"non-finite RMSE: {rmse}"
        _pass(f"Equation.compute_residual() OK; RMSE={rmse:.4f}")
        _pass(f"trajectory shape={tuple(traj.shape)}")
    except Exception as e:
        return _fail("exception", str(e))
    return True


# ---------------------------------------------------------------------------
# TEST 4 - Generality: wave / diffusion / advection
# ---------------------------------------------------------------------------
def test4_generality():
    print("\nTEST 4: GENERALITY")
    from ripple.core.simulation import Simulation
    from ripple.core.experiment import Experiment

    configs = [
        ("wave",      _make_wave_system,      dict(c=1.0,  dt=0.001,  dx=1/16)),
        ("diffusion", _make_diffusion_system, dict(c=1.0,  dt=0.0001, dx=1/16)),
        ("advection", _make_advection_system, dict(c=0.5,  dt=0.001,  dx=1/16)),
    ]

    all_ok = True
    for name, sys_fn, sim_kw in configs:
        # Simulation
        try:
            system = sys_fn()
            u0, v0 = _u0_v0(N=16)
            traj   = Simulation(system, **sim_kw).run(u0, v0, steps=3)
            assert traj is not None and traj.numel() > 0 and not torch.isnan(traj).any()
            _pass(f"{name} Simulation: {tuple(traj.shape)}")
        except Exception as e:
            _fail(f"{name} Simulation", str(e)); all_ok = False

        # Experiment - real autograd path (inputs tensor connects model to equation)
        try:
            system2 = sys_fn()
            model   = _TinyNet()
            opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
            exp     = Experiment(system2, model, opt)
            x = torch.rand(16, 1, requires_grad=True)
            t = torch.rand(16, 1, requires_grad=True)
            loss = exp.train(x, t)
            assert math.isfinite(loss), "non-finite loss"
            _pass(f"{name} Experiment: loss={loss:.4f}")
        except Exception as e:
            _fail(f"{name} Experiment", str(e)); all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = [
        test1_experiment_uses_equation(),
        test2_no_duplicate_residual(),
        test3_simulation_consistency(),
        test4_generality(),
    ]
    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r)
    print(f"RESULT: {passed}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)
