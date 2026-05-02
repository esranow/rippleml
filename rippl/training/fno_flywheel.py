"""
rippl.training.fno_flywheel — Operator Learning Flywheel (FD Solver -> FNO).
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from rippl.nn.fno import FNO
from rippl.core.operator_experiment import OperatorDataset, OperatorExperiment
from rippl.solvers.fd_solver import solve_diffusion_fd_1d

class FNOFlywheel:
    """
    FNO Flywheel: uses high-fidelity FD solvers to generate datasets
    for training Fourier Neural Operators.
    """
    def __init__(self, system, fno_model: FNO, n_train: int = 1000, n_test: int = 200, device: str = "cpu"):
        self.system = system
        self.fno_model = fno_model
        self.n_train = n_train
        self.n_test = n_test
        self.device = torch.device(device)
        self.fno_model.to(self.device)
        
        self.dataset_train = None
        self.dataset_test = None
        self._cache = {}

    def generate_dataset(self, ic_fn: callable = None) -> tuple:
        """
        Generates random ICs and computes solutions via FD solver.
        """
        if "data" in self._cache:
            return self._cache["data"]
            
        n_samples = self.n_train + self.n_test
        
        # Grid from system domain
        # Assuming 1D for now as per flywheel spec
        bounds = self.system.domain.bounds[0]
        L = bounds[1] - bounds[0]
        N = self.system.domain.resolution[0]
        dx = L / (N - 1)
        x_grid = torch.linspace(bounds[0], bounds[1], N).reshape(N, 1)
        
        # Default IC generator: random Fourier modes
        if ic_fn is None:
            def default_ic(x):
                # x: (N, 1)
                # sum_k a_k * sin(k * pi * x)
                u0 = torch.zeros_like(x)
                for k in range(1, 6):
                    ak = torch.randn(1)
                    u0 += ak * torch.sin(k * np.pi * x / L)
                return u0
            ic_fn = default_ic
            
        inputs = []
        outputs = []
        
        # We need alpha and T for the solver
        # Extracting from system.equation
        # This is a bit heuristic, assuming u_t = alpha * u_xx
        alpha = 0.01
        for c, op in self.system.equation.terms:
            from rippl.physics.operators import Laplacian
            if isinstance(op, Laplacian):
                alpha = -c # u_t - alpha*u_xx = 0 -> u_t = alpha*u_xx
                break
        
        T = 1.0
        dt = 0.5 * (dx**2) / (alpha + 1e-8) * 0.9 # CFL safe
        steps = int(T / dt)
        
        print(f"Generating {n_samples} samples using FD solver (steps={steps})...")
        
        for i in range(n_samples):
            u0 = ic_fn(x_grid).unsqueeze(0) # (1, N, 1)
            # solve_diffusion_fd_1d(u0, steps, alpha, dt, dx) -> (1, steps+1, N, 1)
            sol = solve_diffusion_fd_1d(u0, steps, alpha, dt, dx)
            u_T = sol[:, -1, :, :] # (1, N, 1)
            
            inputs.append(u0.squeeze(0))
            outputs.append(u_T.squeeze(0))
            
        inputs = torch.stack(inputs) # (n_samples, N, 1)
        outputs = torch.stack(outputs) # (n_samples, N, 1)
        
        self.dataset_train = OperatorDataset(inputs[:self.n_train], outputs[:self.n_train])
        self.dataset_test = OperatorDataset(inputs[self.n_train:], outputs[self.n_train:])
        
        self._cache["data"] = (inputs, outputs)
        return inputs, outputs

    def train(self, epochs: int = 500, lr: float = 1e-3, batch_size: int = 32) -> dict:
        if self.dataset_train is None:
            self.generate_dataset()
            
        print(f"Training FNO for {epochs} epochs...")
        exp = OperatorExperiment(self.fno_model, self.dataset_train, system=self.system)
        # Assuming OperatorExperiment.train supports batch_size or we use defaults
        # The provided OperatorExperiment.train uses batch_size=32 hardcoded.
        # I'll use it as is or modify it if allowed.
        # "must be 270+ passing" suggests I should not break existing code.
        
        history = []
        optimizer = torch.optim.Adam(self.fno_model.parameters(), lr=lr)
        loader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for a_batch, u_batch in loader:
                a_batch, u_batch = a_batch.to(self.device), u_batch.to(self.device)
                optimizer.zero_grad()
                u_pred = self.fno_model(a_batch)
                loss = torch.nn.functional.mse_loss(u_pred, u_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            history.append(avg_loss)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1} | Loss: {avg_loss:.6e}")
                
        return {"loss_history": history}

    def evaluate(self, n_test: int = None) -> dict:
        n_test = n_test or self.n_test
        loader = DataLoader(self.dataset_test, batch_size=n_test)
        
        self.fno_model.eval()
        with torch.no_grad():
            a_batch, u_batch = next(iter(loader))
            a_batch, u_batch = a_batch.to(self.device), u_batch.to(self.device)
            u_pred = self.fno_model(a_batch)
            
            # L2 error: ||u_pred - u_true||_2 / ||u_true||_2
            diff = (u_pred - u_batch).pow(2).sum(dim=(1, 2)).sqrt()
            norm = u_batch.pow(2).sum(dim=(1, 2)).sqrt()
            l2_errors = diff / (norm + 1e-8)
            
        return {
            "mean_l2": l2_errors.mean().item(),
            "max_l2": l2_errors.max().item(),
            "min_l2": l2_errors.min().item()
        }

    def pipeline(self, ic_fn: callable = None, train_epochs: int = 500) -> dict:
        inputs, outputs = self.generate_dataset(ic_fn=ic_fn)
        history = self.train(epochs=train_epochs)
        results = self.evaluate()
        return {**history, **results}
