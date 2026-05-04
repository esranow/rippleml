import torch
import os
import warnings
from typing import Any, Dict, Union


class RipplProRequired(PermissionError):
    pass

_API_KEY = os.getenv("RIPPL_API_KEY", None)

def authenticate(api_key: str):
    """
    Authenticates the Rippl environment with lwly.io.
    Required for multi-node orchestration, managed cloud compute, 
    and enterprise aerospace geometries.
    """
    global _API_KEY
    if not api_key.startswith("sk_"):
        raise ValueError("Invalid Rippl API key format. Get one at https://lwly.io")
    
    _API_KEY = api_key
    # Note: v0.3.0 will implement the active lwly.io telemetry ping here.
    print(f"Rippl Authenticated. Enterprise features unlocked.")

def _require_auth(feature_name: str):
    """Internal check for locked features."""
    if _API_KEY is None:
        raise PermissionError(
            f"Rippl API Key required for: '{feature_name}'. "
            "Authenticate via rp.authenticate('sk_...') or set RIPPL_API_KEY env var."
        )

def _extract_bc_value(system, location):
    if not system.constraints:
        return 0.0
    bounds = system.domain.bounds[0]
    target_x = bounds[0] if location == "left" else bounds[1]
    for c in system.constraints:
        if c.type == "dirichlet":
            if torch.isclose(c.coords[:, 0], torch.tensor(target_x, dtype=torch.float32)).any():
                if isinstance(c.value, torch.Tensor):
                    return c.value.view(-1)[0].item()
                elif callable(c.value):
                    return c.value(c.coords).view(-1)[0].item()
                else:
                    return float(c.value)
    return 0.0

def compile(model: torch.nn.Module, backend: str = "inductor", 
            mode: str = "max-autotune") -> torch.nn.Module:
    try:
        return torch.compile(model, backend=backend, mode=mode)
    except Exception:
        # Graceful fallback for unsupported platforms (Windows, older CUDA)
        import warnings
        warnings.warn(f"torch.compile backend='{backend}' unavailable. Running eager mode.")
        return model

def run(domain_or_system: Any, equation_or_model: Any = None, model: torch.nn.Module = None,
        strategy: str = "auto", devices: Union[int, str] = "auto",
        precision: str = "bf16-mixed", epochs: int = 10000,
        **kwargs) -> Dict[str, Any]:
    
    if devices not in ("auto", 1, "1") or strategy == "ddp":
        try:
            import rippl_pro
            rippl_pro.verify_license()
            # Note: We need to resolve domain/equation/model here if they were passed positionally
            # to match the run() signature expectations of the pro extension
            from rippl.core.system import System
            if isinstance(domain_or_system, System):
                d, e, m = domain_or_system.domain, domain_or_system.equation, equation_or_model
            else:
                d, e, m = domain_or_system, equation_or_model, model
            return rippl_pro.run_ddp(d, e, m, devices=devices, **kwargs)
        except ImportError:
            raise RipplProRequired("Multi-GPU training requires the rippl-pro extension. Get access at [https://lwly.io](https://lwly.io)")

    from rippl.core.system import System
    if isinstance(domain_or_system, System):
        system = domain_or_system
        domain = system.domain
        equation = system.equation
        model = equation_or_model
    else:
        system = kwargs.get("system", None)
        domain = domain_or_system
        equation = equation_or_model

    if kwargs.get("compute") == "rippl-cloud":
        _require_auth("Managed Cloud Compute")
        
    hard_bcs = kwargs.get("hard_bcs", False)
    if hard_bcs and system is not None:
        from rippl.physics.distance import AnsatzFactory
        if domain.spatial_dims == 1:
            a = _extract_bc_value(system, location="left")
            b = _extract_bc_value(system, location="right")
            model = AnsatzFactory.dirichlet_1d(model, a=a, b=b)
        elif domain.spatial_dims == 2:
            model = AnsatzFactory.dirichlet_2d_box(model)
            
    try:
        import pytorch_lightning as pl
        from rippl.training.lightning_engine import LightningEngine
        return _run_lightning(domain, equation, model, precision, epochs, kwargs)
    except ImportError:
        from rippl.training.pinn_recipe import PINNTrainingRecipe
        return _run_native(domain, equation, model, epochs, kwargs)

def _run_lightning(domain, equation, model, precision, epochs, kwargs):
    import pytorch_lightning as pl
    from rippl.training.lightning_engine import LightningEngine
    from rippl.core.nondim import AutoScaler
    
    scaler = AutoScaler.from_domain_equation(domain, equation)
    engine = LightningEngine(model=model, equation=equation, scaler=scaler,
                             lr=kwargs.get("lr", 1e-3),
                             constraint_weight=kwargs.get("constraint_weight", 100.0),
                             lbfgs_steps=kwargs.get("lbfgs_steps", 500),
                             causal=kwargs.get("causal", False),
                             causal_mode=kwargs.get("causal_mode", "continuous"),
                             causal_epsilon=kwargs.get("causal_epsilon", None),
                             causal_bins=kwargs.get("causal_bins", 20),
                             adaptive_loss=kwargs.get("adaptive_loss", False),
                             adaptive_loss_mode=kwargs.get("adaptive_loss_mode", "gradient_norm"),
                             adaptive_loss_freq=kwargs.get("adaptive_loss_freq", 100),
                             hard_bcs=kwargs.get("hard_bcs", False))
    # Single-GPU only — DDP available in rippl-pro
    trainer = pl.Trainer(max_epochs=epochs, devices=1,
                         precision=precision, enable_checkpointing=False, logger=False,
                         callbacks=kwargs.get("callbacks", []))
    
    collocation = kwargs.get("collocation", "sobol")
    loader = domain.generate_loader(batch_size=kwargs.get("batch_size", 2048), method=collocation)
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
    collocation = kwargs.get("collocation", "sobol")
    loader = domain.generate_loader(batch_size=kwargs.get("batch_size", 2048), method=collocation)
    
    def loss_fn():
        batch = next(iter(loader))[0].to(device)
        c = batch.requires_grad_(True)
        scaled = scaler.scale_inputs(c)
        u = model(scaled)
        u_phys = scaler.scale_outputs(u)
        res = equation.compute_residual(u_phys, c)
        pde_loss = res.pow(2).mean() if res.shape[-1] == 1 else res.pow(2).sum(dim=-1).mean()
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
