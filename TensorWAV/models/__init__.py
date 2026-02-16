"""
NeuralWave Core: Models Package
Exposes core model components and registry.
"""
from TensorWAV.models.registry import build_model, register_model

__all__ = ["build_model", "register_model"]
