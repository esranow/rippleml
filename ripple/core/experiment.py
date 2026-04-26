import torch
from ripple.core.system import System
from ripple.core.equation_system import EquationSystem

class Experiment:
    def __init__(self, system: System, model, opt):
        self.system = system
        self.model = model
        self.opt = opt

    def train(self, coords, epochs=1):
        """Clean training loop with residual + constraint loss."""
        for _ in range(epochs):
            self.opt.zero_grad()
            
            # 1. Forward pass & Wrapping
            coords.requires_grad_(True)
            u_out = self.model(coords)
            
            # Auto-wrap if single tensor
            fields = u_out if isinstance(u_out, dict) else {"u": u_out}
            
            # 2. Physics Loss
            if isinstance(self.system.equation, EquationSystem):
                loss_pde = self.system.equation.compute_loss(fields, coords)
            else:
                # Fallback to single field logic
                # Extract 'u' field for legacy Equation.compute_residual
                pde_res = self.system.equation.compute_residual(fields["u"], coords)
                loss_pde = (pde_res**2).mean()
            
            # 3. Constraint Loss
            loss_const = torch.tensor(0.0, device=coords.device)
            import torch.nn.functional as F
            for c in self.system.constraints:
                c_coords = c.coords.to(coords.device)
                u_pred_all = self.model(c_coords)
                
                # Extract specific field for constraint
                fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                u_pred = fields_c[c.field]
                
                u_target = c.value(c_coords) if callable(c.value) else c.value
                if isinstance(u_target, (float, int)):
                    u_target = torch.full_like(u_pred, u_target)
                else:
                    u_target = u_target.to(coords.device)
                    
                loss_const = loss_const + F.mse_loss(u_pred, u_target)
            
            # Phase 2 introduced weighted loss (100x default)
            total_loss = loss_pde + 100.0 * loss_const
            
            if torch.isnan(total_loss):
                raise RuntimeError("Training encountered NaN loss")
                
            total_loss.backward()
            self.opt.step()
            
        return {
            "loss": total_loss.item(),
            "meta": {"epochs": epochs, "pde_loss": loss_pde.item()}
        }
