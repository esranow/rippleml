"""
rippl.benchmarks.digital_twin_heat — Identifying Thermal Diffusivity.
"""
import torch
import numpy as np
from rippl.core.system import System, Domain
from rippl.physics.operators import TimeDerivative, Laplacian
from rippl.core.equation import Equation
from rippl.core.inverse import InverseParameter, DigitalTwin
from rippl.nn.multi_field_mlp import MultiFieldMLP

def main():
    # 1. Unknown Parameter
    alpha_true = 0.1
    alpha_learn = InverseParameter("alpha", initial_value=0.5, transform="softplus")
    
    # 2. Equation with learned parameter
    # u_t - alpha * u_xx = 0
    eq = Equation([
        (1.0, TimeDerivative(field="u")),
        (-1.0, lambda f, c, d: alpha_learn.get() * Laplacian(field="u").forward(f, c, d))
    ])
    
    # 3. Domain
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(50,))
    
    # 4. System
    sys = System(equation=eq, domain=domain, fields=["u"])
    
    # 5. Synthetic Sensor Data
    # Exact solution: u(x,t) = exp(-alpha*pi^2*t)*sin(pi*x)
    t_sensor = torch.linspace(0, 1, 10).reshape(-1, 1)
    x_sensor = torch.linspace(0, 1, 10).reshape(-1, 1)
    T, X = torch.meshgrid(t_sensor.squeeze(), x_sensor.squeeze(), indexing='ij')
    coords_sensor = torch.stack([X.flatten(), T.flatten()], dim=-1)
    
    u_true = torch.exp(-alpha_true * np.pi**2 * coords_sensor[:, 1:2]) * \
             torch.sin(np.pi * coords_sensor[:, 0:1])
    
    # 6. Digital Twin
    model = MultiFieldMLP(in_dim=2, fields=["u"], hidden=32, layers=3)
    
    dt = DigitalTwin(sys, model, parameters=[alpha_learn],
                    sensor_coords=coords_sensor, 
                    sensor_fields={"u": u_true})
    
    print("Digital Twin System built for Heat equation.")

if __name__ == "__main__":
    main()
