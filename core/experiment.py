import torch
from ripple.core.system import System

class Experiment:
    def __init__(self, system: System, model, opt):
        self.system = system
        self.model = model
        self.opt = opt

    def train(self, x, t):
        self.opt.zero_grad()
        inputs = torch.cat([x, t], dim=-1)
        inputs.requires_grad_(True)
        u = self.model(inputs)
        res = self.system.equation.compute_residual(u, inputs)
        loss = (res**2).mean()
        for constraint in self.system.constraints:
            loss = loss + constraint.weight * constraint.fn(u, x, t)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss")
        loss.backward()
        self.opt.step()
        return loss.item()
