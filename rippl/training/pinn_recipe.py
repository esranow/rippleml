import torch
import time
import numpy as np
from typing import Dict, Optional, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rippl.training.lbfgs_config import LBFGSConfig

class PINNTrainingRecipe:
    """
    Validated training strategy for PINNs.
    Implements: constraint curriculum → lazy NTK Adam → dynamic LBFGS handoff.
    Proven on wave equation (L2=1.36e-03) and heat equation (L2=4.17e-04).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: callable,
        constraint_loss_fn: callable,
        device: torch.device,
        phase_a_epochs: int = 3000,       # constraints only
        phase_b_epochs: int = 10000,      # full loss, Adam
        lbfgs_steps: int = 500,           # second-order refinement
        adam_lr: float = 5e-4,
        lbfgs_lr: float = 1.0,
        grad_clip: float = 1.0,
        ntk_freq: int = 500,              # lazy NTK update frequency
        plateau_patience: int = 300,      # epochs before lr decay
        plateau_factor: float = 0.5,
        min_lr: float = 1e-6,
        constraint_weight: float = 100.0,
        causal: bool = True,
        causal_mode: str = "binned",
        causal_bins: int = 20,
        verbose: bool = True,
        log_freq: int = 1000,
        lbfgs_config: Optional[dict] = None
    ):
        self.model = model
        self.loss_fn = loss_fn # callable returning (total, pde_res_only) or just total
        self.constraint_loss_fn = constraint_loss_fn # callable returning (total, dict_of_components)
        self.device = device
        self.phase_a_epochs = phase_a_epochs
        self.phase_b_epochs = phase_b_epochs
        self.lbfgs_steps = lbfgs_steps
        self.adam_lr = adam_lr
        self.lbfgs_lr = lbfgs_lr
        self.grad_clip = grad_clip
        self.ntk_freq = ntk_freq
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.min_lr = min_lr
        self.constraint_weight = constraint_weight
        self.causal = causal
        self.causal_mode = causal_mode
        self.causal_bins = causal_bins
        self.verbose = verbose
        self.log_freq = log_freq
        self.lbfgs_config = lbfgs_config or LBFGSConfig.STANDARD

        # State
        self.ntk_weights = {"pde": 1.0, "const": 1.0}
        self.loss_history = []

    def run(self) -> dict:
        """Execute full training pipeline."""
        start_time = time.time()
        
        # Phase A
        if self.verbose: print(f"--- Phase A: Constraint Curriculum ({self.phase_a_epochs} epochs) ---")
        loss_a = self._phase_a()
        
        # Phase B
        if self.verbose: print(f"--- Phase B: Adam + Lazy NTK ({self.phase_b_epochs} epochs) ---")
        loss_b, epochs_run = self._phase_b()
        
        # Phase C
        if self.verbose: print(f"--- Phase C: LBFGS Refinement ({self.lbfgs_steps} steps) ---")
        loss_c = self._phase_c()
        
        total_time = time.time() - start_time
        converged = loss_c < 1e-4

        return {
            "phase_a_final_loss": float(loss_a),
            "phase_b_final_loss": float(loss_b),
            "phase_b_epochs_run": int(epochs_run),
            "phase_c_final_loss": float(loss_c),
            "total_time": total_time,
            "converged": converged
        }

    def _phase_a(self) -> float:
        """Constraint only training."""
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        final_loss = 0.0
        for epoch in range(self.phase_a_epochs):
            opt.zero_grad()
            loss, _ = self.constraint_loss_fn()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()
            final_loss = loss.item()
            if self.verbose and epoch % self.log_freq == 0:
                print(f"Phase A Epoch {epoch}: loss={final_loss:.2e}")
        return final_loss

    def _phase_b(self) -> tuple:
        """Full loss training with Adam and Lazy NTK."""
        opt = torch.optim.Adam(self.model.parameters(), lr=self.adam_lr)
        sched = ReduceLROnPlateau(opt, mode='min', factor=self.plateau_factor, 
                                  patience=self.plateau_patience, min_lr=self.min_lr)
        
        epochs_run = 0
        final_loss = 0.0
        
        for epoch in range(self.phase_b_epochs):
            epochs_run += 1
            opt.zero_grad()
            
            # Compute losses
            # loss_fn should return (pde_loss, pde_res) if causal is used, else just pde_loss
            pde_out = self.loss_fn()
            pde_loss = pde_out[0] if isinstance(pde_out, (list, tuple)) else pde_out
            
            const_loss, _ = self.constraint_loss_fn()
            
            # Lazy NTK Update
            if epoch % self.ntk_freq == 0:
                self._update_ntk_weights(pde_loss, const_loss)
            
            # Combine weighted loss
            total_loss = self.ntk_weights["pde"] * pde_loss + \
                         self.ntk_weights["const"] * self.constraint_weight * const_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()
            
            final_loss = total_loss.item()
            self.loss_history.append(final_loss)
            sched.step(final_loss)
            
            # Check for handoff
            current_lr = opt.param_groups[0]['lr']
            if current_lr < self.min_lr * 2:
                if self._dynamic_handoff_check(self.loss_history):
                    if self.verbose: print(f"Dynamic handoff triggered at epoch {epoch}")
                    break
            
            if self.verbose and epoch % self.log_freq == 0:
                print(f"Phase B Epoch {epoch}: loss={final_loss:.2e} lr={current_lr:.2e}")
                
        return final_loss, epochs_run

    def _phase_c(self) -> float:
        """L-BFGS exploitation phase."""
        opt = torch.optim.LBFGS(self.model.parameters(), **self.lbfgs_config)
        
        final_loss = 0.0
        
        def closure():
            opt.zero_grad()
            pde_out = self.loss_fn()
            pde_loss = pde_out[0] if isinstance(pde_out, (list, tuple)) else pde_out
            const_loss, _ = self.constraint_loss_fn()
            
            total_loss = self.ntk_weights["pde"] * pde_loss + \
                         self.ntk_weights["const"] * self.constraint_weight * const_loss
            total_loss.backward()
            return total_loss

        for step in range(self.lbfgs_steps):
            loss = opt.step(closure)
            final_loss = loss.item()
            if self.verbose and step % 50 == 0:
                print(f"Phase C L-BFGS Step {step}: loss={final_loss:.2e}")
                
        return final_loss

    def _update_ntk_weights(self, pde_loss, const_loss):
        """Lazy NTK update using gradient norm ratio."""
        self.model.zero_grad()
        pde_loss.backward(retain_graph=True)
        g_pde = self._get_grad_norm()
        
        self.model.zero_grad()
        const_loss.backward(retain_graph=True)
        g_const = self._get_grad_norm()
        
        if g_const > 1e-8:
            self.ntk_weights["const"] = float(g_pde / g_const)
        self.ntk_weights["pde"] = 1.0
        
        self.model.zero_grad()
        if self.verbose:
            print(f"[NTK Update] weights: {self.ntk_weights}")

    def _get_grad_norm(self):
        grads = [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
        if not grads: return 0.0
        return torch.cat(grads).norm().item()

    def _dynamic_handoff_check(self, loss_history: list, window: int = 200) -> bool:
        """
        Returns True when Adam should hand off to LBFGS.
        Condition: std(loss[-window:]) / mean(loss[-window:]) < 1e-3
        """
        if len(loss_history) < window:
            return False
        recent = np.array(loss_history[-window:])
        return np.std(recent) / (np.mean(recent) + 1e-12) < 1e-3
