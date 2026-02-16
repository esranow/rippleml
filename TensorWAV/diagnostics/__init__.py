"""
NeuralWave Core: Diagnostics Package
"""
from TensorWAV.diagnostics.metrics import l2_error, relative_l2_error
from TensorWAV.diagnostics.energy import wave_energy
from TensorWAV.diagnostics.spectral import spectral_error

__all__ = ["l2_error", "relative_l2_error", "wave_energy", "spectral_error"]
