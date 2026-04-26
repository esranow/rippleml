import torch, math, sys
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(False)

def analytic_solution(x, t, alpha=0.1):
    return torch.sin(math.pi * x) * torch.exp(-alpha * (math.pi**2) * t)

def main():
    print("Estimated time: Adam ~25min, LBFGS ~20min, Total ~45min")
    print(f"Device: {device}")
    
    alpha = 0.1
    
    # 1. Raw SIREN
    class SineLayer(torch.nn.Module):
        def __init__(self, in_f, out_f, omega=30.0, is_first=False):
            super().__init__()
            self.omega = omega
            self.linear = torch.nn.Linear(in_f, out_f)
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1/in_f, 1/in_f)
                else:
                    self.linear.weight.uniform_(-math.sqrt(6/in_f)/omega, math.sqrt(6/in_f)/omega)
        def forward(self, x):
            return torch.sin(self.omega * self.linear(x))

    class SIREN(torch.nn.Module):
        def __init__(self, hidden_layers=4, hidden_dim=128):
            super().__init__()
            layers = [SineLayer(2, hidden_dim, is_first=True)]
            for _ in range(hidden_layers - 1):
                layers.append(SineLayer(hidden_dim, hidden_dim))
            self.net = torch.nn.Sequential(*layers)
            self.final = torch.nn.Linear(hidden_dim, 1)
        def forward(self, x):
            return self.final(self.net(x))

    model = SIREN().to(device)

    # 2. Fixed points sampled once
    x_col = torch.rand(3000, 1)
    t_col = torch.rand(3000, 1)
    coords_col = torch.cat([x_col, t_col], dim=1).to(device)
    
    x_ic = torch.rand(2000, 1)
    coords_ic = torch.cat([x_ic, torch.zeros(2000, 1)], dim=1).to(device)
    u_ic_target = torch.sin(math.pi * x_ic).to(device)
    
    t_bc = torch.rand(2000, 1)
    coords_bc_l = torch.cat([torch.zeros(2000, 1), t_bc], dim=1).to(device)
    coords_bc_r = torch.cat([torch.ones(2000, 1), t_bc], dim=1).to(device)
    u_bc_target = torch.zeros(2000, 1).to(device)

    # 3. Loss function
    def compute_loss():
        # Residual: u_t - alpha * u_xx = 0
        c = coords_col.clone().requires_grad_(True)
        u = model(c)
        du = torch.autograd.grad(u.sum(), c, create_graph=True)[0]
        u_x, u_t = du[:, 0:1], du[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), c, create_graph=True)[0][:, 0:1]
        loss_res = (u_t - alpha * u_xx).pow(2).mean()
        
        # IC
        loss_ic = F.mse_loss(model(coords_ic), u_ic_target)
        
        # BC
        loss_bc = F.mse_loss(model(coords_bc_l), u_bc_target) + F.mse_loss(model(coords_bc_r), u_bc_target)
        
        total = loss_res + 100.0 * (loss_ic + loss_bc)
        return total, loss_res, loss_ic, loss_bc

    # 4. Adam
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    print("Adam training (5000 epochs)...")
    for epoch in range(5000):
        opt.zero_grad()
        loss, res, ic, bc = compute_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: res={res.item():.2e} ic={ic.item():.2e} bc={bc.item():.2e}")

    # 5. LBFGS
    print("LBFGS training (200 steps)...")
    opt_l = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    for step in range(200):
        def closure():
            opt_l.zero_grad()
            l, _, _, _ = compute_loss()
            l.backward()
            return l
        opt_l.step(closure)
        if step % 50 == 0:
            l, res, ic, bc = compute_loss()
            print(f"LBFGS {step}: res={res.item():.2e} ic={ic.item():.2e} bc={bc.item():.2e}")

    # 6. Eval
    x_e = torch.linspace(0, 1, 200)
    t_e = torch.linspace(0, 1, 200)
    X, T = torch.meshgrid(x_e, t_e, indexing='ij')
    coords_e = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)
    with torch.no_grad():
        u_pred = model(coords_e).cpu().reshape(200, 200)
    u_true = analytic_solution(X, T, alpha)
    l2 = (u_pred - u_true).pow(2).sum().sqrt() / u_true.pow(2).sum().sqrt()
    
    print(f"\n=== RESULT ===")
    print(f"L2 error: {l2.item():.6e}")
    if l2 < 1e-2:
        print("PASS")
    else:
        print("FAIL")

    # 7. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(u_true.numpy().T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
    axes[0].set_title("Analytic")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(u_pred.numpy().T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
    axes[1].set_title("Predicted")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow((u_pred - u_true).abs().numpy().T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='magma')
    axes[2].set_title("Abs Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("ripple/benchmarks/heat1d_result.png")
    print("Plot saved to ripple/benchmarks/heat1d_result.png")

if __name__ == "__main__":
    main()
