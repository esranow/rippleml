"""
ripple: Diagnostics Package
"""
from ripple.diagnostics.metrics import l2_error, relative_l2_error
from ripple.diagnostics.energy import wave_energy
from ripple.diagnostics.spectral import spectral_error

__all__ = ["l2_error", "relative_l2_error", "wave_energy", "spectral_error"]
