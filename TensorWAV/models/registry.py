import logging
from typing import Dict, Any, Callable, Type
import torch.nn as nn

# Type for model constructors
ModelConstructor = Callable[[Dict[str, Any]], nn.Module]

# Global registry for models
_MODEL_REGISTRY: Dict[str, ModelConstructor] = {}

def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Decorator to register a model class with a specific name.

    Args:
        name (str): The name to register the model under.

    Returns:
        Callable: The decorator function.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            logging.warning(f"Model '{name}' is already registered. Overwriting.")
        
        # We assume the model class can be initialized with **config
        # But to be safe and consistent with build_model, we can just store the class
        # and instantiate it later using the config dictionary.
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Build a model instance from the registry.

    Args:
        name (str): Name of the model to build.
        config (Dict[str, Any]): Configuration dictionary for the model.

    Returns:
        nn.Module: The instantiated model.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(_MODEL_REGISTRY.keys())}")
    
    model_cls = _MODEL_REGISTRY[name]
    
    # Instantiate the model with the configuration
    # We pass the config as kwargs or a single dict depending on implementation.
    # The requirement says "build_model(name: str, config: dict)".
    # The models should accept arguments from config.
    # To be flexible, we pass **config.
    try:
        model = model_cls(**config)
    except TypeError as e:
        logging.error(f"Error initializing model '{name}' with config {config}: {e}")
        raise e

    return model
