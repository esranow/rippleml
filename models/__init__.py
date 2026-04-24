"""
ripple: Models Package
Exposes core model components and registry.
"""
from ripple.models.registry import build_model, register_model
# Import all models to trigger @register_model decorator registration
from ripple.models import mlp, siren, fourier_mlp, fno  # noqa: F401

__all__ = ["build_model", "register_model"]
