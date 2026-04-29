import yaml
import json
import os
import torch
from typing import Dict, Any, Type, Callable, Optional, Union, List

_OPERATORS: Dict[str, Type] = {}
_SOLVERS: Dict[str, Callable] = {}

def register_operator(name: str):
    """
    Decorator to register a physics operator class.
    
    Args:
        name: The string identifier for the operator (e.g., 'laplacian').
    """
    def decorator(cls: Type):
        _OPERATORS[name.lower()] = cls
        return cls
    return decorator

def register_solver(name: str):
    """
    Decorator to register a numerical solver function.
    
    Args:
        name: The string identifier for the solver (e.g., 'wave_fd').
    """
    def decorator(fn: Callable):
        _SOLVERS[name.lower()] = fn
        return fn
    return decorator

def get_operator_class(name: str) -> Type:
    """Retrieve an operator class by name."""
    if name.lower() not in _OPERATORS:
        raise ValueError(f"Operator '{name}' not registered. Available: {list(_OPERATORS.keys())}")
    return _OPERATORS[name.lower()]

def get_solver_fn(name: str) -> Callable:
    """Retrieve a solver function by name."""
    if name.lower() not in _SOLVERS:
        raise ValueError(f"Solver '{name}' not registered. Available: {list(_SOLVERS.keys())}")
    return _SOLVERS[name.lower()]

class ConfigParser:
    """
    Parser for Rippl configuration files (YAML/JSON).
    Handles serialization and deserialization of the entire system state.
    """
    
    @staticmethod
    def load(path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from a file path or a dictionary.
        
        Args:
            path_or_dict: Path to a .yaml/.json file or a pre-loaded dictionary.
            
        Returns:
            The loaded configuration dictionary.
        """
        if isinstance(path_or_dict, dict):
            return path_or_dict
            
        if not os.path.exists(path_or_dict):
            raise FileNotFoundError(f"Config file not found: {path_or_dict}")
            
        with open(path_or_dict, 'r') as f:
            if path_or_dict.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            return json.load(f)

    @staticmethod
    def save(config: Dict[str, Any], path: str):
        """
        Save configuration to a file.
        
        Args:
            config: The configuration dictionary to save.
            path: Target file path (.yaml or .json).
        """
        with open(path, 'w') as f:
            if path.endswith(('.yaml', '.yml')):
                yaml.dump(config, f, sort_keys=False)
            else:
                json.dump(config, f, indent=2)
