import torch
from ripple.physics.equation import Equation
from ripple.physics.operators import TimeDerivative, Laplacian
from ripple.core import System, Domain
from ripple.core.experiment import Experiment

# basic example
class MLP(torch.nn.Module):
    def forward(self, x, t):
        return x * t

eq = Equation([(1.0, TimeDerivative(order=2)), (-1.0, Laplacian())])
sys = System(eq, domain=Domain(spatial_dims=1))

model = MLP()
opt = torch.optim.Adam(model.parameters())

exp = Experiment(sys, model, opt)

x = torch.rand(10, 1, requires_grad=True)
t = torch.rand(10, 1, requires_grad=True)

for _ in range(10):
    loss = exp.train(x, t)
    print("Loss decreasing:", loss)
