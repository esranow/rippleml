import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

def export_torchscript(
    model: nn.Module, 
    sample_input: torch.Tensor, 
    path: Union[str, Path]
) -> None:
    """
    Export model to TorchScript (Tracing).
    """
    model.eval()
    try:
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save(str(path))
        logging.info(f"Model exported to TorchScript at {path}")
    except Exception as e:
        logging.error(f"Failed to export TorchScript: {e}")
        # Fallback safe: we don't crash, just log error as per requirement "fallback safe"
        # Could also try scripting if tracing fails, but tracing is standard for these models.

def export_onnx(
    model: nn.Module, 
    sample_input: torch.Tensor, 
    path: Union[str, Path]
) -> None:
    """
    Export model to ONNX.
    """
    model.eval()
    try:
        torch.onnx.export(
            model, 
            sample_input, 
            str(path),
            opset_version=14, # Good compatibility
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        logging.info(f"Model exported to ONNX at {path}")
    except Exception as e:
        logging.error(f"Failed to export ONNX: {e}")

def write_model_card(
    config: Dict[str, Any], 
    metrics: Dict[str, float], 
    path: Union[str, Path]
) -> None:
    """
    Write a model card JSON file.
    """
    card = {
        "model_config": config.get("model", {}),
        "training_config": config.get("training", {}),
        "data_config": config.get("data", {}),
        "metrics": metrics,
        "framework": "ripple",
        "version": "0.0.1"
    }
    
    with open(path, 'w') as f:
        json.dump(card, f, indent=2)
    logging.info(f"Model card written to {path}")
