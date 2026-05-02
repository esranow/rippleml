"""
rippl.benchmarks.multifidelity_heat — Fusing Sparse High-Fi and Dense Low-Fi Data.
"""
import torch
import numpy as np
from rippl.data.sensor import SensorDataset, MultiFidelityFusion
from rippl.core.system import System, Domain
from rippl.physics.operators import TimeDerivative, Laplacian
from rippl.core.equation import Equation
from rippl.nn.multi_field_mlp import MultiFieldMLP

def main():
    # 1. High-Fidelity Data (Sparse, No Noise)
    coords_hi = torch.rand(50, 2)
    u_hi = torch.sin(np.pi * coords_hi[:, 0:1]) * torch.exp(-0.1 * np.pi**2 * coords_hi[:, 1:2])
    ds_hi = SensorDataset(coords_hi, {"u": u_hi}, fidelity=1.0, noise_std=0.0)
    
    # 2. Low-Fidelity Data (Dense, Noisy)
    coords_lo = torch.rand(200, 2)
    u_lo_clean = torch.sin(np.pi * coords_lo[:, 0:1]) * torch.exp(-0.1 * np.pi**2 * coords_lo[:, 1:2])
    u_lo = u_lo_clean + 0.05 * torch.randn_like(u_lo_clean)
    ds_lo = SensorDataset(coords_lo, {"u": u_lo}, fidelity=0.1, noise_std=0.05)
    
    # 3. Physics Regularization
    eq = Equation([(1.0, TimeDerivative(field="u")), (-0.1, Laplacian(field="u"))])
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(50,))
    sys = System(equation=eq, domain=domain, fields=["u"])
    
    # 4. Fusion
    fusion = MultiFidelityFusion([ds_hi, ds_lo], physics_weight=0.1)
    
    model = MultiFieldMLP(in_dim=2, fields=["u"], hidden=32, layers=3)
    
    print("Multi-Fidelity Fusion System built.")

if __name__ == "__main__":
    main()
