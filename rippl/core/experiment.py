import torch
from typing import List, Optional, Any, Dict
from rippl.core.system import System
from rippl.core.equation_system import EquationSystem

from rippl.training.causal import CausalTrainingMixin

class Experiment(CausalTrainingMixin):
    def __init__(self, system: System, model, opt, 
                 use_hard_constraints: bool = False,
                 adaptive_collocation: bool = False,
                 causal_training: bool = False,
                 causal_mode: str = "continuous",
                 causal_epsilon: Optional[float] = None,
                 causal_bins: int = 10,
                 adaptive_loss: bool = False,
                 adaptive_loss_mode: str = "gradient_norm",
                 adaptive_loss_freq: int = 100,
                 adaptive_loss_alpha: float = 0.9,
                 conservation_laws: Optional[List] = None,
                 validate: bool = False):
        self.system = system
        self.model = model
        self.opt = opt
        self.use_hard_constraints = use_hard_constraints
        self.adaptive_collocation = adaptive_collocation
        self.causal_training = causal_training
        self.causal_mode = causal_mode
        self.causal_epsilon = causal_epsilon
        self.causal_bins = causal_bins
        self.adaptive_loss = adaptive_loss
        self.adaptive_loss_mode = adaptive_loss_mode
        self.adaptive_loss_freq = adaptive_loss_freq
        self.adaptive_loss_alpha = adaptive_loss_alpha
        self.conservation_laws = conservation_laws or []
        self.validate_after_train = validate
        
        if self.adaptive_loss:
            from rippl.training.ntk_weighting import AdaptiveLossBalancer
            loss_names = ["pde"] + [f"const_{i}" for i in range(len(self.system.constraints))]
            self.balancer = AdaptiveLossBalancer(
                mode=self.adaptive_loss_mode,
                loss_names=loss_names,
                update_freq=self.adaptive_loss_freq,
                alpha=self.adaptive_loss_alpha
            )
        
        if self.use_hard_constraints:
            from rippl.physics.distance import BoxDistance, HardConstraintWrapper
            dist_fn = BoxDistance(self.system.domain.bounds)
            self.model = HardConstraintWrapper(
                model, dist_fn, 
                particular_solution=self.system.particular_solution
            )
            
        if self.adaptive_collocation:
            from rippl.training.adaptive_sampler import AdaptiveCollocationSampler
            self.sampler = AdaptiveCollocationSampler(self.system.domain)

    def _compute_grad_norm(self, loss_component: torch.Tensor, model: torch.nn.Module) -> float:
        """Isolated helper to compute gradient norm for NTK update."""
        model.zero_grad()
        loss_component.backward(retain_graph=True)
        grad_norm_sq = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
        model.zero_grad() # Clear immediately
        return torch.sqrt(grad_norm_sq).item()

    def train(self, coords: torch.Tensor, epochs: int = 1, ntk_freq: int = 500, patience: int = 200, logger: Any = None, params: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """Run the training loop with Lazy NTK weighting and dynamic L-BFGS handoff."""
        best_loss = float('inf')
        patience_counter = 0
        ntk_weights = {name: 1.0 for name in (["pde"] + [f"const_{i}" for i in range(len(self.system.constraints))])}
        
        # Ensure coords require grad for physics derivatives
        coords = coords.clone().detach().requires_grad_(True)

        def _get_losses(current_coords):
            """Core loss computation closure."""
            # 1. Forward pass
            u_out = self.model(current_coords)
            fields = u_out if isinstance(u_out, dict) else {self.system.fields[0]: u_out}
            
            # 2. Physics Loss
            spatial_dims = self.system.domain.spatial_dims
            if isinstance(self.system.equation, EquationSystem):
                residuals = self.system.equation.compute_residuals(fields, current_coords, spatial_dims=spatial_dims)
                if self.causal_training:
                    weighted_losses = []
                    for res in residuals:
                        w = self.compute_causal_weights_binned(current_coords, res, n_bins=self.causal_bins, epsilon=self.causal_epsilon) if self.causal_mode == "binned" else \
                            self.compute_causal_weights_continuous(current_coords, res, epsilon=self.causal_epsilon)
                        weighted_losses.append(torch.mean(w * res**2))
                    loss_pde = sum(w_eq * l for w_eq, l in zip(self.system.equation.weights, weighted_losses))
                else:
                    loss_pde = self.system.equation.compute_loss(fields, current_coords, spatial_dims=spatial_dims)
            else:
                pde_res = self.system.equation.compute_residual(fields[self.system.fields[0]], current_coords, spatial_dims=spatial_dims)
                if self.causal_training:
                    w = self.compute_causal_weights_binned(current_coords, pde_res, n_bins=self.causal_bins, epsilon=self.causal_epsilon) if self.causal_mode == "binned" else \
                        self.compute_causal_weights_continuous(current_coords, pde_res, epsilon=self.causal_epsilon)
                    loss_pde = (w * pde_res**2).mean()
                else:
                    loss_pde = (pde_res**2).mean()
            
            # 3. Constraint Loss
            loss_dict = {"pde": loss_pde}
            import torch.nn.functional as F
            from rippl.core.system import NeumannConstraint
            for i, c in enumerate(self.system.constraints):
                if isinstance(c, NeumannConstraint):
                    c_coords = c.coords.to(current_coords.device).requires_grad_(True)
                    u_pred_all = self.model(c_coords)
                    fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                    u_pred = fields_c[c.field]
                    grad = torch.autograd.grad(u_pred.sum(), c_coords, create_graph=True, retain_graph=True)[0]
                    val_pred = grad[..., c.normal_direction : c.normal_direction + 1]
                else:
                    if self.use_hard_constraints and c.type == "dirichlet": continue
                    c_coords = c.coords.to(current_coords.device)
                    u_pred_all = self.model(c_coords)
                    fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                    val_pred = fields_c[c.field]
                
                u_target = c.value(c_coords) if callable(c.value) else c.value
                if isinstance(u_target, (float, int)): u_target = torch.full_like(val_pred, float(u_target))
                else: u_target = u_target.to(current_coords.device)
                loss_dict[f"const_{i}"] = F.mse_loss(val_pred, u_target)
            
            # 4. Conservation Loss
            loss_cons = torch.tensor(0.0, device=current_coords.device)
            for law in self.conservation_laws:
                loss_cons = loss_cons + law.penalty(self.model, current_coords)
            
            return loss_dict, loss_cons

        # Feature: Set reference for conservation laws
        if self.conservation_laws:
            for law in self.conservation_laws:
                law.set_reference(self.model, coords)

        # Main Adam Training Loop
        for epoch in range(epochs):
            self.opt.zero_grad()
            
            # Dynamic updates: Collocation + Moving Boundaries
            if self.adaptive_collocation:
                self.sampler.update(self.model, self.system.equation, epoch)
                coords = self.sampler.current_points().requires_grad_(True)
            
            # Feature: Moving Boundary Update (Adam only)
            for c in self.system.constraints:
                if hasattr(c, "update"):
                    c.update(epoch, self.model)
            
            loss_dict, loss_cons = _get_losses(coords)
            
            # Lazy NTK Update
            if self.adaptive_loss and epoch % ntk_freq == 0:
                # Update weights: lambda_i = max(||grad_pde||) / ||grad_const_i||
                g_pde = self._compute_grad_norm(loss_dict["pde"], self.model)
                new_weights = {"pde": 1.0}
                weight_log = ["pde: 1.0"]
                for i in range(len(self.system.constraints)):
                    name = f"const_{i}"
                    g_i = self._compute_grad_norm(loss_dict[name], self.model)
                    w_i = g_pde / (g_i + 1e-8)
                    new_weights[name] = w_i
                    weight_log.append(f"{name}: {w_i:.2f}")
                ntk_weights.update(new_weights)
                print(f"[NTK] Epoch {epoch}: Weight shift -> {', '.join(weight_log)}")

            # Apply Weights
            total_loss = sum(ntk_weights[k] * v for k, v in loss_dict.items()) + 10.0 * loss_cons
            
            if torch.isnan(total_loss): raise RuntimeError("Training encountered NaN loss")
            
            # Adam step
            total_loss.backward()
            self.opt.step()

            if logger:
                l_vals = {k: v.item() for k, v in loss_dict.items()}
                l_vals["total"] = total_loss.item()
                logger.log_epoch(epoch, l_vals, params=params)

            # Plateau Detection (Handoff trigger)
            current_loss_val = total_loss.item()
            if current_loss_val < best_loss - 1e-5:
                best_loss = current_loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"[HANDOFF] Epoch {epoch}: Adam plateaued at loss {current_loss_val:.4e}. Triggering L-BFGS exploitation...")
                break

        # L-BFGS Exploitation Phase
        if hasattr(self.model, "parameters") and list(self.model.parameters()):
            print("[L-BFGS] Starting exploitation phase...")
            lbfgs_opt = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
            
            def closure():
                lbfgs_opt.zero_grad()
                l_dict, l_cons = _get_losses(coords)
                # Apply FROZEN ntk_weights
                total = sum(ntk_weights[k] * v for k, v in l_dict.items()) + 10.0 * l_cons
                if total.requires_grad:
                    total.backward()
                return total

            # Run L-BFGS for a fixed number of steps
            for step in range(100):
                lbfgs_opt.step(closure)
        # else: L-BFGS skipped for functional models

        if self.validate_after_train:
            from rippl.diagnostics.physics_validator import PhysicsValidator
            validator = PhysicsValidator(self.system, self.model, coords)
            validator.full_report()

        return {
            "loss": best_loss,
            "meta": {"epochs_adam": epoch + 1, "final_ntk_weights": ntk_weights}
        }
