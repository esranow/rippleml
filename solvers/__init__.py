"""
ripple: Solvers Package
"""
from ripple.solvers.fd_solver import solve_wave_fd_1d, solve_wave_fd_2d
from ripple.solvers.spectral_solver import solve_periodic_spectral_1d

__all__ = ["solve_wave_fd_1d", "solve_wave_fd_2d", "solve_periodic_spectral_1d"]
