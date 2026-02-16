"""
NeuralWave Core: Solvers Package
"""
from TensorWAV.solvers.fd_solver import solve_wave_fd_1d, solve_wave_fd_2d
from TensorWAV.solvers.spectral_solver import solve_periodic_spectral_1d

__all__ = ["solve_wave_fd_1d", "solve_wave_fd_2d", "solve_periodic_spectral_1d"]
