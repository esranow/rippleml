import torch
import torch.nn.functional as F
from typing import Any, Dict

try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False

if HAS_LIGHTNING:
    class LightningEngine(pl.LightningModule):
        def __init__(self, model, equation, scaler, lr=1e-3,
                     constraint_weight=100.0, lbfgs_steps=500,
                     causal=False):
            super().__init__()
            self.model = model
            self.equation = equation
            self.scaler = scaler
            self.lr = lr
            self.constraint_weight = constraint_weight
            self.lbfgs_steps = lbfgs_steps
            self.causal = causal
            self.automatic_optimization = False  # manual for L-BFGS handoff
            self._phase = "adam"  # "adam" or "lbfgs"
            self._loss_history = []
            self.final_loss = None

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            coords = batch[0]
            adam_opt, lbfgs_opt = self.optimizers()
            is_ddp = self.trainer.num_devices > 1

            def compute_loss():
                c = coords.requires_grad_(True)
                scaled = self.scaler.scale_inputs(c)
                u = self.model(scaled)
                u_phys = self.scaler.scale_outputs(u)
                pde_loss = self.equation.compute_loss(
                    {"u": u_phys} if not isinstance(u_phys, dict) else u_phys, c
                )
                total = pde_loss # Add constraint losses here if applicable
                return total, pde_loss

            if self._phase == "adam":
                adam_opt.zero_grad()
                total, pde_loss = compute_loss()
                self.manual_backward(total)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                adam_opt.step()
                
                self._loss_history.append(total.item())
                if len(self._loss_history) > 400:
                    window = self._loss_history[-200:]
                    rel_std = torch.tensor(window).std() / (torch.tensor(window).mean() + 1e-8)
                    if rel_std < 1e-3:
                        self._phase = "lbfgs"
                        self._lbfgs_count = 0
                        
                self.log("pde_loss", pde_loss, prog_bar=True, sync_dist=is_ddp)
                self.final_loss = total.item()

            elif self._phase == "lbfgs":
                def closure():
                    lbfgs_opt.zero_grad()
                    total, _ = compute_loss()
                    self.manual_backward(total)
                    return total
                lbfgs_opt.step(closure)
                self._lbfgs_count += 1
                if self._lbfgs_count >= self.lbfgs_steps:
                    self.trainer.should_stop = True

        def configure_optimizers(self):
            adam = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            lbfgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0,
                                        max_iter=20, line_search_fn="strong_wolfe")
            return [adam, lbfgs]
else:
    class LightningEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("pytorch-lightning required. pip install rippl[distributed]")
