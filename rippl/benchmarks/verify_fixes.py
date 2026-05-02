import torch, math
import sys
import os

# Ensure rippl is in path
sys.path.append(os.getcwd())

# Test 1: Laplacian does not include time dim
from rippl.physics.operators import Laplacian, TimeDerivative
from rippl.physics.derivatives import compute_all_derivatives
from rippl.core.equation import Equation

coords = torch.rand(100, 2, requires_grad=True)  # (x, t)
u = torch.sin(math.pi * coords[:, 0:1]) * torch.cos(math.pi * coords[:, 1:2])

# Laplacian of sin(πx)cos(πt) w.r.t x only = -π²*sin(πx)cos(πt)
lap = Laplacian(field="u", spatial_dims=1)
derived = compute_all_derivatives({"u": u}, coords, ["u_xx"])
result = lap.forward({"u": u}, coords, derived)
expected = -(math.pi**2) * torch.sin(math.pi * coords[:, 0:1]) * torch.cos(math.pi * coords[:, 1:2])
err = (result - expected).abs().max().item()
print(f"Laplacian spatial-only test: {err:.2e} {'PASS' if err < 1e-3 else 'FAIL'}")

# Test 2: Physics sanity — wave equation residual on analytic solution
from rippl.nn.residual import HybridWaveResidualBlock
from rippl.core.equation import Equation as Eq

coords2 = torch.rand(1000, 2, requires_grad=True)
u2 = torch.sin(math.pi * coords2[:, 0:1]) * torch.cos(math.pi * coords2[:, 1:2])
# Wave equation: u_tt - u_xx = 0
eq = Eq([(1.0, TimeDerivative(2)), (-1.0, Laplacian(spatial_dims=1))])
block = HybridWaveResidualBlock(a=1.0, b=0.0, c=1.0, use_correction=False, spatial_dim=1)
res = block.residual(u2, eq, coords2)
val = res.abs().max().item()
print(f"Physics sanity check: {val:.2e} {'PASS' if val < 1e-2 else 'FAIL'}")

# Test 3: Stokes pressure gradient in 1D
from rippl.physics.operators import PressureGradient
p_coords = torch.rand(100, 1, requires_grad=True)  # 1D, no time
p = 1 - p_coords  # p(x) = 1-x, so p_x = -1
derived_p = compute_all_derivatives({"p": p}, p_coords, ["p_x"])
pg = PressureGradient(field_p="p", direction=0)
p_x_result = pg.forward({"p": p}, p_coords, derived_p)
p_x_expected = -torch.ones_like(p)
err_p = (p_x_result - p_x_expected).abs().max().item()
print(f"Pressure gradient 1D test: {err_p:.2e} {'PASS' if err_p < 1e-3 else 'FAIL'}")

print("\nAll fix verifications complete.")
