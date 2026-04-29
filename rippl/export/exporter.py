import torch
import json
import os
import warnings
from typing import Optional, Dict, Any
from rippl.core.exceptions import PhysicsModelWarning

def export_model(model: torch.nn.Module, 
                 path: str, 
                 format: str = "torchscript", 
                 metadata: Optional[Dict[str, Any]] = None):
    """
    Export a trained Rippl model for production inference.
    Handles TorchScript and ONNX formats, and generates a Model Card.

    Args:
        model: The trained neural network to export.
        path: Directory where exported artifacts will be saved.
        format: Export format, either 'torchscript' (default) or 'onnx'.
        metadata: Optional dictionary containing 'pde', 'config', 'scales', and other model info.
    """
    os.makedirs(path, exist_ok=True)
    
    # 1. Primary Model Export
    if format == "torchscript":
        try:
            # Prefer torch.jit.script for generalizability
            exported_module = torch.jit.script(model)
        except Exception:
            # Fallback to torch.jit.trace if script fails (common with complex closures or dynamic control flow)
            # We assume a 2D coordinate input (N, 2) as a default sample for tracing.
            sample_input = torch.randn(1, 2)
            exported_module = torch.jit.trace(model, sample_input)
        
        exported_module.save(os.path.join(path, "model.pt"))
        
    elif format == "onnx":
        warnings.warn(
            "ONNX does not reliably support double-backward autograd, which is critical for "
            "evaluating second-order PDEs in PINNs. Use TorchScript for physics-aware inference.", 
            PhysicsModelWarning
        )
        sample_input = torch.randn(1, 2)
        torch.onnx.export(model, sample_input, os.path.join(path, "model.onnx"))
    
    # 2. Generate Model Card
    model_card = {
        "format": format,
        "metadata": metadata or {},
        "pde_solved": metadata.get("pde", "unknown") if metadata else "unknown"
    }
    with open(os.path.join(path, "model_card.json"), "w") as f:
        json.dump(model_card, f, indent=2)
    
    # 3. Save weights and configuration for load_model compatibility
    # This ensures exported models can still be re-loaded via Rippl's model registry.
    torch.save(model.state_dict(), os.path.join(path, "weights.pt"))
    
    # Extract non-dimensionalization scales if available
    scales_data = {}
    if hasattr(model, "scales"):
        scales_data = {
            "values": model.scales.report(),
            "has_time": getattr(model, "has_time", True),
            "field_types": getattr(model, "field_types", {})
        }
    elif metadata and "scales" in metadata:
        scales_data = metadata["scales"]

    # Package registry configuration
    registry_config = {
        "name": metadata.get("name", model.__class__.__name__.lower()) if metadata else model.__class__.__name__.lower(),
        "model_config": metadata.get("config", {}) if metadata else {},
        "scales": scales_data
    }
    
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(registry_config, f, indent=2)
