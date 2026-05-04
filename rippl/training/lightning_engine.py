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
                     causal=False, causal_mode="continuous", causal_epsilon=None, causal_bins=20,
                     adaptive_loss=False, adaptive_loss_mode="gradient_norm", adaptive_loss_freq=100,
                     hard_bcs=False):
            super().__init__()
            self.model = model
            self.equation = equation
            self.scaler = scaler
            self.lr = lr
            self.constraint_weight = constraint_weight
            self.lbfgs_steps = lbfgs_steps
            self.causal = causal
            self.causal_mode = causal_mode
            self.causal_epsilon = causal_epsilon
            self.causal_bins = causal_bins
            self.adaptive_loss = adaptive_loss
            self.adaptive_loss_mode = adaptive_loss_mode
            self.adaptive_loss_freq = adaptive_loss_freq
            self.hard_bcs = hard_bcs
            self.automatic_optimization = False  # manual for L-BFGS handoff
            self._phase = "adam"  # "adam" or "lbfgs"
            self._loss_history = []
            self.final_loss = None

            if self.adaptive_loss:
                from rippl.training.ntk_weighting import AdaptiveLossBalancer
                self._balancer = AdaptiveLossBalancer(
                    mode=self.adaptive_loss_mode, 
                    loss_names=["pde", "ic", "bc"], 
                    update_freq=self.adaptive_loss_freq
                )
            else:
                self._balancer = None

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            coords = batch[0]
            adam_opt, lbfgs_opt = self.optimizers()

            def compute_loss():
                c = coords.requires_grad_(True)
                scaled = self.scaler.scale_inputs(c)
                u = self.model(scaled)
                u_phys = self.scaler.scale_outputs(u)
                
                if self.causal and self._phase == "adam":
                    # compute pointwise residuals before reduction
                    pointwise_res = self.equation.compute_pointwise_residual(
                        {"u": u_phys} if not isinstance(u_phys, dict) else u_phys, c
                    )
                    
                    # apply causal weights
                    from rippl.training.causal import CausalTrainingMixin
                    mixin = CausalTrainingMixin()
                    if self.causal_mode == "continuous":
                        weights = mixin.compute_causal_weights_continuous(
                            c, pointwise_res, epsilon=self.causal_epsilon
                        )
                    else:
                        weights = mixin.compute_causal_weights_binned(
                            c, pointwise_res, n_bins=self.causal_bins,
                            epsilon=self.causal_epsilon
                        )
                    pde_loss = (weights * pointwise_res.pow(2)).mean()
                else:
                    res = self.equation.compute_residual(u_phys, c)
                    pde_loss = res.pow(2).mean() if res.shape[-1] == 1 else res.pow(2).sum(dim=-1).mean()
                
                # Mock constraints for now unless we integrate actual system constraint data
                ic_loss = torch.tensor(0.0, device=coords.device, requires_grad=True)
                bc_loss = torch.tensor(0.0, device=coords.device, requires_grad=True)
                
                if self.hard_bcs:
                    bc_loss = torch.tensor(0.0, device=coords.device, requires_grad=True)
                
                loss_dict = {
                    "pde": pde_loss,
                    "ic": ic_loss,
                    "bc": bc_loss
                }

                total_loss = sum(loss_dict.values())
                
                if self.adaptive_loss:
                    self._balancer.step(self.model, loss_dict, total_loss, self.current_epoch)
                    total_loss = self._balancer.apply(loss_dict)

                return total_loss, pde_loss

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
                        
                self.log("pde_loss", pde_loss, prog_bar=True)
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
