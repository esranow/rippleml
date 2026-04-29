import torch
import json
import os
import torch.nn as nn
from typing import Dict, Any, Type, Callable, Optional, List

# Global registry for model classes
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Decorator to register a neural network model class in the global registry.

    Args:
        name: The string identifier for the model (e.g., 'mlp', 'fno').

    Returns:
        The decorator function that adds the class to the registry.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(name: str, config: dict) -> nn.Module:
    """
    Instantiate a model from the registry using a configuration dictionary.

    Args:
        name: The identifier of the model to build.
        config: A dictionary of keyword arguments to pass to the model constructor.

    Returns:
        An instantiated torch.nn.Module.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**config)

def load_model(path: str) -> nn.Module:
    """
    Load a model from a directory containing weights.pt and config.json.
    Automatically restores ReferenceScales state if non-dimensionalization was used.

    Args:
        path: Path to the directory containing model artifacts.

    Returns:
        The loaded and initialized torch.nn.Module, possibly wrapped in NondimModelWrapper.
    """
    config_path = os.path.join(path, "config.json")
    weights_path = os.path.join(path, "weights.pt")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = build_model(config["name"], config["model_config"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    
    if "scales" in config and config["scales"]:
        from rippl.core.nondim import ReferenceScales, NondimModelWrapper
        scales_config = config["scales"]
        scales_values = scales_config.get("values", scales_config)
        
        reference_scales = ReferenceScales(**scales_values)
        model = NondimModelWrapper(
            model, 
            reference_scales, 
            has_time=scales_config.get("has_time", True),
            field_types=scales_config.get("field_types", {})
        )
        
    return model

class ModelAdapter(nn.Module):
    """
    Adapter wrapper to normalize input/output contracts between different architectures.
    Ensures that standard MLPs and FNOs can be used interchangeably by the Experiment orchestration.
    """
    def __init__(self, model: nn.Module, architecture: str = "mlp", resolution: Optional[List[int]] = None):
        """
        Initialize the adapter.

        Args:
            model: The underlying neural network.
            architecture: The architecture type ('mlp' or 'fno').
            resolution: The spatial resolution [H, W, ...] for FNO reshaping.
        """
        super().__init__()
        self.model = model
        self.architecture = architecture
        self.resolution = resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic shape normalization.

        Args:
            x: Input coordinates or grid tensor.

        Returns:
            The model output.
        """
        if self.architecture == "fno" and x.ndim == 2:
            # Flattened coordinates to grid tensor if resolution is provided
            if self.resolution:
                batch_size = 1
                x = x.unsqueeze(0).view(batch_size, *self.resolution, -1)
                output = self.model(x)
                return output.view(-1, output.shape[-1])
        return self.model(x)
