"""
NeuralWave Core: Physics Blocks Package

Hybrid physics-aware neural blocks that combine fixed physics operators
with small learnable correction networks (MLP / conv).
"""

from TensorWAV.physics_blocks.laplacian import HybridLaplacianBlock
from TensorWAV.physics_blocks.residual import HybridWaveResidualBlock
from TensorWAV.physics_blocks.spectral import SpectralHybridBlock
from TensorWAV.physics_blocks.energy import EnergyAwareBlock
from TensorWAV.physics_blocks.oscillator import HybridOscillatorBlock
from TensorWAV.physics_blocks.embedding import PDEParameterEmbeddingBlock
from TensorWAV.physics_blocks.gradient import HybridGradientBlock
from TensorWAV.physics_blocks.boundary_block import BoundaryConditionBlock
from TensorWAV.physics_blocks.hamiltonian import HamiltonianBlock
from TensorWAV.physics_blocks.spectral_reg import SpectralRegularizationBlock
from TensorWAV.physics_blocks.multiscale_ff import MultiScaleFourierFeatureBlock
from TensorWAV.physics_blocks.spectral_conv import SpectralConvBlock
from TensorWAV.physics_blocks.hybrid_stepper import HybridTimeStepperBlock
from TensorWAV.physics_blocks.adaptivesampler import AdaptiveSamplingBlock
from TensorWAV.physics_blocks.conservation_block import ConservationConstraintBlock
from TensorWAV.physics_blocks.nn_operator_wrapper import OperatorWrapperBlock

__all__ = [
    "HybridLaplacianBlock",
    "HybridWaveResidualBlock",
    "SpectralHybridBlock",
    "EnergyAwareBlock",
    "HybridOscillatorBlock",
    "PDEParameterEmbeddingBlock",
    "HybridGradientBlock",
    "BoundaryConditionBlock",
    "HamiltonianBlock",
    "SpectralRegularizationBlock",
    "MultiScaleFourierFeatureBlock",
    "SpectralConvBlock",
    "HybridTimeStepperBlock",
    "AdaptiveSamplingBlock",
    "ConservationConstraintBlock",
    "OperatorWrapperBlock",
    "BLOCK_REGISTRY",
]

BLOCK_REGISTRY = {
    "laplacian": HybridLaplacianBlock,
    "wave_residual": HybridWaveResidualBlock,
    "spectral": SpectralHybridBlock,
    "energy": EnergyAwareBlock,
    "oscillator": HybridOscillatorBlock,
    "pde_embedding": PDEParameterEmbeddingBlock,
    "gradient": HybridGradientBlock,
    "boundary": BoundaryConditionBlock,
    "hamiltonian": HamiltonianBlock,
    "spectral_reg": SpectralRegularizationBlock,
    "multiscale_ff": MultiScaleFourierFeatureBlock,
    "spectral_conv": SpectralConvBlock,
    "hybrid_stepper": HybridTimeStepperBlock,
    "adaptive_sampler": AdaptiveSamplingBlock,
    "conservation": ConservationConstraintBlock,
    "operator_wrapper": OperatorWrapperBlock,
}
