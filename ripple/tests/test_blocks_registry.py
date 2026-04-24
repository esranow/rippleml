import torch
import pytest
import numpy as np
from ripple.physics_blocks import (
    HybridLaplacianBlock,
    HybridWaveResidualBlock,
    SpectralHybridBlock,
    EnergyAwareBlock,
    HybridOscillatorBlock,
    PDEParameterEmbeddingBlock,
    BLOCK_REGISTRY,
)

def test_block_imports():
    """Verify that all blocks are in the registry."""
    assert len(BLOCK_REGISTRY) >= 16
    assert "laplacian" in BLOCK_REGISTRY
    assert "spectral" in BLOCK_REGISTRY
    assert "energy" in BLOCK_REGISTRY

def test_laplacian_block_forward():
    block = HybridLaplacianBlock(mode="point", correction_input_dim=2)
    x = torch.rand(2, 20, 1, requires_grad=True)
    u = torch.sin(x)
    out = block(u, coords=x)
    assert out.shape == (2, 20, 1)

def test_spectral_block_forward():
    block = SpectralHybridBlock(n_modes=64, cutoff=8)
    u = torch.randn(2, 64)
    out = block(u)
    assert out.shape == (2, 64)

def test_energy_block_forward():
    block = EnergyAwareBlock(c=1.0, spatial_dim=1)
    coords = torch.rand(2, 30, 2, requires_grad=True)
    u = torch.sin(coords[..., 0:1])
    corrected, penalty = block(u, coords)
    assert corrected.shape == u.shape
    assert penalty.shape == ()

def test_oscillator_block_step():
    block = HybridOscillatorBlock(omega=1.0, alpha=0.0)
    state = torch.randn(4, 2)
    next_state = block.step(state, dt=0.01)
    assert next_state.shape == (4, 2)

def test_embedding_block_forward():
    block = PDEParameterEmbeddingBlock(param_dim=3, embed_dim=16, feature_dim=32)
    params = torch.randn(4, 3)
    emb = block(params=params)
    assert emb.shape == (4, 16)

def test_block_pipeline():
    lap_block = HybridLaplacianBlock(mode="grid", spatial_dim=1, use_correction=False)
    spec_block = SpectralHybridBlock(n_modes=64, cutoff=16, use_correction=False)
    
    u_grid = torch.randn(1, 1, 64)
    lap_out = lap_block(u_grid)
    spec_out = spec_block(lap_out.squeeze(1))
    assert spec_out.shape == (1, 64)

def test_registry_instantiation():
    block = BLOCK_REGISTRY["spectral"](n_modes=32)
    assert isinstance(block, SpectralHybridBlock)
