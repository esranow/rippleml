import torch, math, sys
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from rippl.nn.siren import Siren
from rippl.nn.residual import HybridWaveResidualBlock
from rippl.core.system import Constraint, Domain
from rippl.physics.equation import Equation
from rippl.physics.operators import TimeDerivative, Laplacian

# 1. SETUP
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analytic_solution(x, t):
    return torch.sin(math.pi * x) * torch.cos(math.pi * t)

# PHASE 0 — PHYSICS SANITY CHECK
print("--- PHASE 0: PHYSICS SANITY CHECK ---")
coords_verify = torch.rand(1000, 2, requires_grad=True).to(device)
x_v = coords_verify[..., 0:1]
t_v = coords_verify[..., 1:2]
u_analytic = analytic_solution(x_v, t_v)

# Verify with HybridWaveResidualBlock
# The block expects an Equation object and coords
eq_verify = Equation(terms=[(1.0, TimeDerivative(order=2)), (-1.0, Laplacian())])
res_block = HybridWaveResidualBlock(use_correction=False)
res_v = res_block.residual(u_analytic, eq_verify, coords_verify)
max_res = torch.abs(res_v).max().item()

if max_res > 1e-2:
    print(f"PHYSICS BROKEN (max residual: {max_res:.6e}) — stopping.")
    sys.exit(1)
else:
    print(f"PHYSICS OK: {max_res:.6e}")

# LOSS FUNCTION
def compute_loss(model, coords_interior, constraint_list, w_constraint=100.0):
    coords_interior = coords_interior.to(device).requires_grad_(True)
    u = model(coords_interior)
    
    # manual autograd residual computation as requested
    grads = torch.autograd.grad(u.sum(), coords_interior, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    u_xx = torch.autograd.grad(u_x.sum(), coords_interior, create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t.sum(), coords_interior, create_graph=True)[0][:, 1:2]
    
    res = u_tt - u_xx  # c=1
    loss_res = (res ** 2).mean()
    
    loss_con = torch.tensor(0.0, device=device)
    for c in constraint_list:
        coords_c = c.coords.to(device)
        u_pred = model(coords_c)
        u_target = c.value(coords_c) if callable(c.value) else c.value.to(device)
        loss_con += F.mse_loss(u_pred, u_target)
    
    total_loss = loss_res + w_constraint * loss_con
    return total_loss, loss_res, loss_con

# CONSTRAINTS SETUP
x_ic_pts = torch.rand(2000, 1)
t_ic_pts = torch.zeros(2000, 1)
coords_ic = torch.cat([x_ic_pts, t_ic_pts], dim=1)
val_ic = torch.sin(math.pi * x_ic_pts)

t_bc_pts = torch.rand(2000, 1)
coords_bc_left = torch.cat([torch.zeros(2000, 1), t_bc_pts], dim=1)
coords_bc_right = torch.cat([torch.ones(2000, 1), t_bc_pts], dim=1)
val_bc = torch.zeros(2000, 1)

ic_constraint = Constraint(type="initial", field="u", coords=coords_ic, value=val_ic)
bc_left = Constraint(type="dirichlet", field="u", coords=coords_bc_left, value=val_bc)
bc_right = Constraint(type="dirichlet", field="u", coords=coords_bc_right, value=val_bc)
global_constraints = [ic_constraint, bc_left, bc_right]

def train_time_marching(model_config):
    # model_config: (hidden_features, hidden_layers)
    hf, hl = model_config
    model = Siren(input_dim=2, output_dim=1, hidden_layers=[hf]*hl, omega_0=30.0).to(device)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, patience=150, factor=0.5, min_lr=1e-6
    )
    
    time_windows = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    all_losses = []
    
    for i, (t_start, t_end) in enumerate(time_windows):
        # Sample interior points
        x_col = torch.rand(3000, 1)
        t_col = t_start + (t_end - t_start) * torch.rand(3000, 1)
        coords_interior = torch.cat([x_col, t_col], dim=1).to(device)
        
        # Build window constraints
        window_constraints = [bc_left, bc_right]
        if i == 0:
            window_constraints.append(ic_constraint)
        else:
            # Interface constraint
            x_int = torch.rand(500, 1)
            t_int = torch.full((500, 1), t_start)
            coords_int = torch.cat([x_int, t_int], dim=1).to(device)
            model.eval()
            with torch.no_grad():
                val_int = model(coords_int)
            window_constraints.append(Constraint(type="interface", field="u", coords=coords_int.cpu(), value=val_int.cpu()))
        
        # Adam training
        model.train()
        for epoch in range(3000):
            optimizer_adam.zero_grad()
            loss, l_res, l_con = compute_loss(model, coords_interior, window_constraints)
            loss.backward()
            optimizer_adam.step()
            scheduler.step(loss)
            all_losses.append(loss.item())
        
        adam_loss = loss.item()
        
        # LBFGS training
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=0.5, max_iter=20, 
            line_search_fn="strong_wolfe"
        )
        
        for step in range(200):
            def closure():
                optimizer_lbfgs.zero_grad()
                loss, _, _ = compute_loss(model, coords_interior, window_constraints)
                loss.backward()
                return loss
            loss_lbfgs = optimizer_lbfgs.step(closure)
            all_losses.append(loss_lbfgs.item())
        
        final_lbfgs_loss = loss_lbfgs.item()
        
        # Window L2 error
        x_w = torch.linspace(0, 1, 50)
        t_w = torch.linspace(t_start, t_end, 50)
        Xw, Tw = torch.meshgrid(x_w, t_w, indexing='ij')
        coords_w = torch.stack([Xw, Tw], dim=-1).to(device)
        u_analytic_w = analytic_solution(Xw, Tw).to(device)
        model.eval()
        with torch.no_grad():
            u_pred_w = model(coords_w).squeeze(-1)
        l2_w = torch.norm(u_pred_w - u_analytic_w) / torch.norm(u_analytic_w)
        
        print(f"Window {i}: Adam Loss={adam_loss:.2e}, LBFGS Loss={final_lbfgs_loss:.2e}, L2={l2_w.item():.2e}")

    # Final Full Evaluation
    x_eval = torch.linspace(0, 1, 200)
    t_eval = torch.linspace(0, 1, 200)
    Xe, Te = torch.meshgrid(x_eval, t_eval, indexing='ij')
    coords_eval = torch.stack([Xe, Te], dim=-1).to(device)
    u_analytic_eval = analytic_solution(Xe, Te).to(device)
    
    model.eval()
    with torch.no_grad():
        u_rippl_final = model(coords_eval).squeeze(-1)
    
    l2_final = torch.norm(u_rippl_final - u_analytic_eval) / torch.norm(u_analytic_eval)
    l2_val = l2_final.item()
    
    return l2_val, u_rippl_final.cpu(), u_analytic_eval.cpu(), all_losses

# Phase 1: Small Model
print("\n--- PHASE 1: Training (128x4) ---")
l2_p1, u_p1, u_a, losses_p1 = train_time_marching((128, 4))
print(f"Final L2 error: {l2_p1:.6e}")
status = "PASS" if l2_p1 < 1e-2 else "FAIL"
print(f"Status: {status}")

# Fallback if Phase 1 fails
if status == "FAIL":
    print("\n--- RECOVERY: Training (256x5) ---")
    l2_p2, u_p2, u_a, losses_p2 = train_time_marching((256, 5))
    print(f"RECOVERY L2 error: {l2_p2:.6e}")
    status_p2 = "PASS" if l2_p2 < 1e-2 else "FAIL"
    print(f"Status: {status_p2}")
    # Use P2 results for plotting
    u_final = u_p2
    losses_final = losses_p1 + losses_p2
    final_l2 = l2_p2
else:
    u_final = u_p1
    losses_final = losses_p1
    final_l2 = l2_p1

# PLOT
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
im0 = axes[0].imshow(u_a.T.numpy(), origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
axes[0].set_title("Analytic Solution")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u_final.T.numpy(), origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
axes[1].set_title("Rippl Solution")
plt.colorbar(im1, ax=axes[1])

error = torch.abs(u_final - u_a)
im2 = axes[2].imshow(error.T.numpy(), origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='magma')
axes[2].set_title("Absolute Error")
plt.colorbar(im2, ax=axes[2])

axes[3].plot(losses_final)
axes[3].set_yscale('log')
axes[3].set_title("Concatenated Loss Curve")
axes[3].set_xlabel("Steps")

plt.tight_layout()
plt.savefig("rippl/benchmarks/wave1d_result.png")
print("\nSaved plot to rippl/benchmarks/wave1d_result.png")

print("\n=== PHASE 1 RESULT ===")
print(f"Physics check: {max_res:.6e}")
print(f"Final L2 error: {final_l2:.6e}")
print(f"Status: {'PASS' if final_l2 < 1e-2 else 'FAIL'}")
