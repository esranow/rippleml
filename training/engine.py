import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import yaml # Assuming config input might be dict or we parse inside if needed, but signature says Dict

from ripple.models import build_model  # triggers all @register_model decorators
from ripple.physics.pde import PDESpec
from ripple.physics.residuals import build_residual_fn
from ripple.physics.boundary import DirichletBC, NeumannBC, PeriodicBC
from ripple.training.callbacks import Callback, CheckpointCallback
from ripple.physics_blocks import BLOCK_REGISTRY

# Optimizer Registry (Simple mapping for now)
_OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW
}

def get_optimizer(name: str, params, **kwargs) -> optim.Optimizer:
    name = name.lower()
    if name not in _OPTIMIZERS:
        raise ValueError(f"Optimizer {name} not supported.")
    return _OPTIMIZERS[name](params, **kwargs)

def train_from_config(config: Union[str, Path, Dict[str, Any]]) -> None:
    """
    Train a model based on the provided configuration.
    
    Args:
        config (Union[str, Path, Dict[str, Any]]): Config dictionary or path to YAML.
    """
    # 1. Parse Config
    if isinstance(config, (str, Path)):
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    # 2. Setup Device
    device_str = cfg.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"Using device: {device}")

    # 3. Build Model
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.pop("type") # Remove type to pass rest as kwargs
    # Ensure input/output dims are set if not in config (inferred from data usually, but here explicit)
    model = build_model(model_name, model_cfg)
    model.to(device)
    
    # 4. Optimizer
    train_cfg = cfg.get("training", {})
    opt_name = train_cfg.get("optimizer", "adam")
    lr = float(train_cfg.get("learning_rate", 1e-3))
    optimizer = get_optimizer(opt_name, model.parameters(), lr=lr)
    
    # 5. Callbacks
    callbacks: List[Callback] = []
    save_dir = train_cfg.get("save_dir", "./checkpoints")
    save_freq = train_cfg.get("save_freq", 10)
    callbacks.append(CheckpointCallback(save_dir, save_freq))

    # 5b. Build Physics Blocks from config
    blocks_cfg = cfg.get("blocks", [])
    physics_blocks: List[nn.Module] = []
    for blk_cfg in blocks_cfg:
        blk_cfg = dict(blk_cfg)  # copy
        blk_type = blk_cfg.pop("type")
        if blk_type not in BLOCK_REGISTRY:
            raise ValueError(f"Block '{blk_type}' not found. Available: {list(BLOCK_REGISTRY.keys())}")
        block = BLOCK_REGISTRY[blk_type](**blk_cfg).to(device)
        physics_blocks.append(block)
        # Add block params to optimizer
        optimizer.add_param_group({"params": block.parameters(), "lr": lr})
    if physics_blocks:
        logging.info(f"Loaded {len(physics_blocks)} physics block(s): {[b.__class__.__name__ for b in physics_blocks]}")
    
    # 6. Training Loop Config
    epochs = train_cfg.get("epochs", 10)
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    
    mode = cfg.get("mode", "pinn")  # pinn or operator_learning
    
    # 7. Data (Placeholder generation based on task logic)
    # In a real app, we'd use a DataModule. Here we simulate batch generation or use provided specs.
    # For PINN: we need collocation points.
    # For Operator: we need input/output pairs.
    
    # Simple synthetic loop for demonstration as per 'functional' requirement
    # We will generate one fixed batch or random batch per epoch
    
    logging.info(f"Starting training for {epochs} epochs in {mode} mode.")
    
    for cb in callbacks:
        cb.on_train_begin()
        
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        loss_total = torch.tensor(0.0, device=device)
        
        # --- PINN MODE ---
        if mode == "pinn":
            # 1. Physics Loss (Residual)
            # (B, N, D) -> (B, N, D+1) for time? Or just (x, t) as last dim.
            # Assume 1D+time: (B, N, 2)
            # We need domain from config
            
            # Simple placeholder logic for inputs
            # Flat collocation points (M, D): x in [0,1], t in [0,1], last dim = t
            M = 256  # total collocation points
            inputs = torch.rand(M, 2, device=device, requires_grad=True)
            
            # Forward
            with autocast("cuda", enabled=use_amp):
                # Model output u (M, 1)
                u_pred = model(inputs)
                
                # Compute Residual: u_tt - c^2*Lap(u) = 0
                pde = PDESpec(a=1.0, c=1.0)  # wave eq; residual fn uses -c*Lap
                res_fn = build_residual_fn(pde)
                
                res = res_fn(u_pred, inputs)
                loss_physics = torch.mean(res ** 2)
                
                loss_total = loss_total + loss_physics

                # Apply physics blocks
                for blk in physics_blocks:
                    try:
                        blk_out = blk(u_pred, coords=inputs, params=None)
                        if isinstance(blk_out, tuple):
                            _, blk_loss = blk_out
                            loss_total += blk_loss
                    except Exception:
                        pass

                # Apply boundary conditions
                # Boundary condition loss
                bc_cfg = cfg.get("boundary_conditions", [])
                for bc_spec in bc_cfg:
                    bc_type = bc_spec.get("type", "dirichlet")
                    # Boundary points: x=0 boundary, random t
                    N_bc = 64
                    bc_pts = torch.rand(N_bc, 2, device=device)
                    bc_pts[:, 0] = 0.0  # x=0 boundary
                    bc_in = bc_pts.requires_grad_(True)  # (N_bc, 2)
                    bc_u = model(bc_in)
                    if bc_type == "dirichlet":
                        loss_bc = torch.mean(bc_u ** 2)  # enforce u=0
                        loss_total = loss_total + loss_bc
                
        # --- OPERATOR MODE ---
        elif mode == "operator_learning":
            # Data loss
            # Generate random inputs and targets
            # Inputs: (B, N, C_in)
            # Targets: (B, N, C_out)
            B_size = 4
            res = cfg.get("data", {}).get("resolution", [64, 64])
            if isinstance(res, int): N = res
            else: N = res[0] * res[1]
            
            # Dummy data
            inputs = torch.randn(B_size, N, model_cfg.get("input_dim", 1), device=device)
            targets = torch.randn(B_size, N, model_cfg.get("output_dim", 1), device=device)
            
            with autocast("cuda", enabled=use_amp):
                preds = model(inputs)
                loss_data = nn.MSELoss()(preds, targets)
                loss_total += loss_data
                
        else:
             # Fallback
             pass

        # Backward
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Logs
        logs = {"model": model, "optimizer": optimizer, "loss": loss_total.item()}
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
            
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss_total.item():.6f}")

    for cb in callbacks:
        cb.on_train_end()
