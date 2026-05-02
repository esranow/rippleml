import torch
from typing import Any, Dict, Union

def compile(model: torch.nn.Module, backend: str = "inductor", 
            mode: str = "max-autotune") -> torch.nn.Module:
    try:
        return torch.compile(model, backend=backend, mode=mode)
    except Exception:
        # Graceful fallback for unsupported platforms (Windows, older CUDA)
        import warnings
        warnings.warn(f"torch.compile backend='{backend}' unavailable. Running eager mode.")
        return model

def run(domain: Any, equation: Any, model: torch.nn.Module,
        strategy: str = "auto", devices: Union[int, str] = "auto",
        precision: str = "bf16-mixed", epochs: int = 10000,
        **kwargs) -> Dict[str, Any]:
    try:
        import pytorch_lightning as pl
        from rippl.training.lightning_engine import LightningEngine
        return _run_lightning(domain, equation, model, strategy, devices, 
                       precision, epochs, kwargs)
    except ImportError:
        from rippl.training.pinn_recipe import PINNTrainingRecipe
        return _run_native(domain, equation, model, epochs, kwargs)

def _run_lightning(domain, equation, model, strategy, devices, 
                   precision, epochs, kwargs):
    import pytorch_lightning as pl
    from rippl.training.lightning_engine import LightningEngine
    from rippl.core.nondim import AutoScaler
    
    scaler = AutoScaler.from_domain_equation(domain, equation)
    engine = LightningEngine(model=model, equation=equation, scaler=scaler,
                             lr=kwargs.get("lr", 1e-3),
                             constraint_weight=kwargs.get("constraint_weight", 100.0),
                             lbfgs_steps=kwargs.get("lbfgs_steps", 500),
                             causal=kwargs.get("causal", False))
    trainer = pl.Trainer(max_epochs=epochs, strategy=strategy, devices=devices,
                         precision=precision, enable_checkpointing=False, logger=False,
                         callbacks=kwargs.get("callbacks", []))
    loader = domain.generate_loader(batch_size=kwargs.get("batch_size", 2048))
    trainer.fit(engine, train_dataloaders=loader)
    
    return {"model_state": model.state_dict(), 
            "scaler_state": scaler.get_state(),
            "final_loss": engine.final_loss}

def _run_native(domain, equation, model, epochs, kwargs):
    # Fallback: uses existing PINNTrainingRecipe directly
    from rippl.training.pinn_recipe import PINNTrainingRecipe
    from rippl.core.nondim import AutoScaler
    from rippl.core.system import System
    import torch

    device = next(model.parameters()).device
    scaler = AutoScaler.from_domain_equation(domain, equation)
    loader = domain.generate_loader(batch_size=kwargs.get("batch_size", 2048))
    
    def loss_fn():
        batch = next(iter(loader))[0].to(device)
        c = batch.requires_grad_(True)
        scaled = scaler.scale_inputs(c)
        u = model(scaled)
        u_phys = scaler.scale_outputs(u)
        pde_loss = equation.compute_loss({"u": u_phys} if not isinstance(u_phys, dict) else u_phys, c)
        return pde_loss, pde_loss

    def constraint_loss_fn():
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    recipe = PINNTrainingRecipe(
        model=model,
        loss_fn=loss_fn,
        constraint_loss_fn=constraint_loss_fn,
        device=device,
        phase_b_epochs=epochs,
        causal=kwargs.get("causal", False)
    )
    result = recipe.run()
    
    return {
        "model_state": model.state_dict(),
        "scaler_state": scaler.get_state(),
        "final_loss": result.get("phase_c_final_loss", result.get("phase_b_final_loss", 0.0))
    }
