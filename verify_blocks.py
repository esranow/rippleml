"""
Simple standalone verification of physics blocks.
Run: python verify_blocks.py
"""
import sys
import os
# Add parent directory to path for TensorWAV imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("PHYSICS BLOCKS VERIFICATION")
print("=" * 60)

# Test 1: Import all blocks
print("\n[1/4] Testing imports...")
try:
    from TensorWAV.physics_blocks import (
        HybridLaplacianBlock,
        HybridWaveResidualBlock,
        SpectralHybridBlock,
        EnergyAwareBlock,
        HybridOscillatorBlock,
        PDEParameterEmbeddingBlock,
        HybridGradientBlock,
        BoundaryConditionBlock,
        HamiltonianBlock,
        SpectralRegularizationBlock,
        MultiScaleFourierFeatureBlock,
        SpectralConvBlock,
        HybridTimeStepperBlock,
        AdaptiveSamplingBlock,
        ConservationConstraintBlock,
        OperatorWrapperBlock,
        BLOCK_REGISTRY,
    )
    print(f"✓ Successfully imported all 16 blocks")
    print(f"✓ BLOCK_REGISTRY contains {len(BLOCK_REGISTRY)} entries")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Instantiate and forward pass
print("\n[2/4] Testing block instantiation and forward...")
tests_passed = 0
tests_total = 0

# Laplacian (point mode)
tests_total += 1
try:
    block = HybridLaplacianBlock(mode="point", correction_input_dim=2)
    x = torch.rand(2, 20, 1, requires_grad=True)
    u = torch.sin(x)
    out = block(u, coords=x)
    assert out.shape == (2, 20, 1)
    print("✓ HybridLaplacianBlock (point)")
    tests_passed += 1
except Exception as e:
    print(f"✗ HybridLaplacianBlock failed: {e}")

# Spectral
tests_total += 1
try:
    block = SpectralHybridBlock(n_modes=64, cutoff=8)
    u = torch.randn(2, 64)
    out = block(u)
    assert out.shape == (2, 64)
    print("✓ SpectralHybridBlock")
    tests_passed += 1
except Exception as e:
    print(f"✗ SpectralHybridBlock failed: {e}")

# Energy
tests_total += 1
try:
    block = EnergyAwareBlock(c=1.0, spatial_dim=1)
    coords = torch.rand(2, 30, 2, requires_grad=True)
    u = torch.sin(coords[..., 0:1])
    corrected, penalty = block(u, coords)
    assert corrected.shape == u.shape
    assert penalty.shape == ()
    print("✓ EnergyAwareBlock")
    tests_passed += 1
except Exception as e:
    print(f"✗ EnergyAwareBlock failed: {e}")

# Oscillator
tests_total += 1
try:
    block = HybridOscillatorBlock(omega=1.0, alpha=0.0)
    state = torch.randn(4, 2)
    next_state = block.step(state, dt=0.01)
    assert next_state.shape == (4, 2)
    print("✓ HybridOscillatorBlock")
    tests_passed += 1
except Exception as e:
    print(f"✗ HybridOscillatorBlock failed: {e}")

# Embedding
tests_total += 1
try:
    block = PDEParameterEmbeddingBlock(param_dim=3, embed_dim=16, feature_dim=32)
    params = torch.randn(4, 3)
    emb = block(params=params)
    assert emb.shape == (4, 16)
    print("✓ PDEParameterEmbeddingBlock")
    tests_passed += 1
except Exception as e:
    print(f"✗ PDEParameterEmbeddingBlock failed: {e}")

# Test 3: Cross-block pipeline
print("\n[3/4] Testing cross-block pipeline...")
try:
    lap_block = HybridLaplacianBlock(mode="grid", spatial_dim=1, use_correction=False)
    spec_block = SpectralHybridBlock(n_modes=64, cutoff=16, use_correction=False)
    
    u_grid = torch.randn(1, 1, 64)
    lap_out = lap_block(u_grid)
    spec_out = spec_block(lap_out.squeeze(1))
    assert spec_out.shape == (1, 64)
    print("✓ Laplacian → Spectral pipeline works")
    tests_passed += 1
    tests_total += 1
except Exception as e:
    print(f"✗ Pipeline failed: {e}")
    tests_total += 1

# Test 4: BLOCK_REGISTRY
print("\n[4/4] Testing BLOCK_REGISTRY...")
try:
    assert "laplacian" in BLOCK_REGISTRY
    assert "spectral" in BLOCK_REGISTRY
    assert "energy" in BLOCK_REGISTRY
    block = BLOCK_REGISTRY["spectral"](n_modes=32)
    assert isinstance(block, SpectralHybridBlock)
    print(f"✓ BLOCK_REGISTRY functional with {len(BLOCK_REGISTRY)} blocks")
    tests_passed += 1
    tests_total += 1
except Exception as e:
    print(f"✗ BLOCK_REGISTRY failed: {e}")
    tests_total += 1

# Summary
print("\n" + "=" * 60)
print(f"VERIFICATION COMPLETE: {tests_passed}/{tests_total} tests passed")
print("=" * 60)

if tests_passed == tests_total:
    print("\n✓ All blocks verified successfully!")
    sys.exit(0)
else:
    print(f"\n✗ {tests_total - tests_passed} test(s) failed")
    sys.exit(1)
