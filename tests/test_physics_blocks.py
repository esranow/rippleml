"""
Comprehensive tests for TensorWAV.physics_blocks.

Each test sets seeds for determinism, runs on CPU, and should complete in <10s.
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Seed fixture
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def seed_all():
    torch.manual_seed(42)
    np.random.seed(42)


# ===================================================================== #
# 1. HybridLaplacianBlock
# ===================================================================== #
class TestHybridLaplacianBlock:
    def test_point_mode_analytic_sine(self):
        """Laplacian of sin(πx) should be −π² sin(πx)."""
        from TensorWAV.physics_blocks.laplacian import HybridLaplacianBlock

        block = HybridLaplacianBlock(mode="point", use_correction=False,
                                      correction_input_dim=2)
        B, N = 1, 50
        x = torch.linspace(0.01, 0.99, N).reshape(1, N, 1).requires_grad_(True)
        u = torch.sin(math.pi * x)
        lap = block(u, coords=x)

        expected = -(math.pi ** 2) * torch.sin(math.pi * x)
        assert torch.allclose(lap, expected, atol=0.2), \
            f"Max err: {(lap - expected).abs().max().item()}"

    def test_point_mode_shape(self):
        from TensorWAV.physics_blocks.laplacian import HybridLaplacianBlock

        block = HybridLaplacianBlock(mode="point", correction_input_dim=2)
        x = torch.rand(2, 20, 1, requires_grad=True)
        u = torch.sin(x)
        out = block(u, coords=x)
        assert out.shape == (2, 20, 1)

    def test_grid_mode_1d_shape(self):
        from TensorWAV.physics_blocks.laplacian import HybridLaplacianBlock

        block = HybridLaplacianBlock(mode="grid", spatial_dim=1)
        u = torch.randn(2, 1, 32)
        out = block(u)
        assert out.shape == (2, 1, 32)

    def test_grid_mode_2d_shape(self):
        from TensorWAV.physics_blocks.laplacian import HybridLaplacianBlock

        block = HybridLaplacianBlock(mode="grid", spatial_dim=2)
        u = torch.randn(2, 1, 16, 16)
        out = block(u)
        assert out.shape == (2, 1, 16, 16)


# ===================================================================== #
# 2. HybridWaveResidualBlock
# ===================================================================== #
class TestHybridWaveResidualBlock:
    def test_residual_near_zero_on_analytic_wave(self):
        """u = sin(πx)cos(πt), c=1: wave eq residual should be ≈ 0."""
        from TensorWAV.physics_blocks.residual import HybridWaveResidualBlock

        block = HybridWaveResidualBlock(a=1.0, b=0.0, c=1.0, use_correction=False)
        B, N = 1, 50
        coords = torch.rand(B, N, 2, requires_grad=True)
        u = torch.sin(math.pi * coords[..., 0:1]) * \
            torch.cos(math.pi * coords[..., 1:2])
        res = block.residual(u, coords)
        assert res.abs().max().item() < 0.5, \
            f"Residual too large: {res.abs().max().item()}"

    def test_loss_returns_scalar(self):
        from TensorWAV.physics_blocks.residual import HybridWaveResidualBlock

        block = HybridWaveResidualBlock(use_correction=True)
        coords = torch.rand(2, 30, 2, requires_grad=True)
        u = torch.sin(coords[..., 0:1])
        loss = block.loss(u, coords)
        assert loss.shape == ()

    def test_forward_shape(self):
        from TensorWAV.physics_blocks.residual import HybridWaveResidualBlock

        block = HybridWaveResidualBlock()
        coords = torch.rand(2, 20, 2, requires_grad=True)
        u = torch.sin(coords[..., 0:1])
        out = block(u, coords)
        assert out.shape == (2, 20, 1)


# ===================================================================== #
# 3. SpectralHybridBlock
# ===================================================================== #
class TestSpectralHybridBlock:
    def test_forward_shape(self):
        from TensorWAV.physics_blocks.spectral import SpectralHybridBlock

        block = SpectralHybridBlock(n_modes=64, cutoff=8)
        u = torch.randn(2, 64)
        out = block(u)
        assert out.shape == (2, 64)

    def test_spectral_energy_preserved_no_correction(self):
        from TensorWAV.physics_blocks.spectral import SpectralHybridBlock

        block = SpectralHybridBlock(n_modes=64, cutoff=None, use_correction=False)
        u = torch.randn(2, 64)
        with torch.no_grad():
            out = block(u)
        # Energy should be preserved when weights are all 1 and no cutoff
        e_in = (u ** 2).sum()
        e_out = (out ** 2).sum()
        assert torch.allclose(e_in, e_out, rtol=0.01), \
            f"Energy mismatch: {e_in.item()} vs {e_out.item()}"

    def test_get_spectrum(self):
        from TensorWAV.physics_blocks.spectral import SpectralHybridBlock

        block = SpectralHybridBlock(n_modes=32)
        u = torch.randn(1, 32)
        spec = block.get_spectrum(u)
        assert spec.shape[-1] == 17  # rfft of 32 → 17


# ===================================================================== #
# 4. EnergyAwareBlock
# ===================================================================== #
class TestEnergyAwareBlock:
    def test_energy_and_gating_shape(self):
        from TensorWAV.physics_blocks.energy import EnergyAwareBlock

        block = EnergyAwareBlock(c=1.0, spatial_dim=1)
        coords = torch.rand(2, 30, 2, requires_grad=True)
        u = torch.sin(coords[..., 0:1])
        corrected, penalty = block(u, coords)
        assert corrected.shape == u.shape
        assert penalty.shape == ()

    def test_energy_computation(self):
        from TensorWAV.physics_blocks.energy import EnergyAwareBlock

        block = EnergyAwareBlock(c=1.0)
        coords = torch.rand(1, 20, 2, requires_grad=True)
        u = torch.sin(coords[..., 0:1]) * torch.cos(coords[..., 1:2])
        energy = block.compute_energy(u, coords)
        assert energy.shape == (1, 1)
        assert energy.item() > 0


# ===================================================================== #
# 5. HybridOscillatorBlock
# ===================================================================== #
class TestHybridOscillatorBlock:
    def test_step_shape(self):
        from TensorWAV.physics_blocks.oscillator import HybridOscillatorBlock

        block = HybridOscillatorBlock(omega=1.0, alpha=0.0)
        state = torch.randn(4, 2)
        next_state = block.step(state, dt=0.01)
        assert next_state.shape == (4, 2)

    def test_forward_dynamics(self):
        from TensorWAV.physics_blocks.oscillator import HybridOscillatorBlock

        block = HybridOscillatorBlock(omega=1.0, alpha=0.0, use_correction=False)
        state = torch.tensor([[1.0, 0.0]])
        dsdt = block(state)
        # For harmonic oscillator at (1,0): du/dt=v=0, dv/dt=-ω²u=-1
        assert torch.allclose(dsdt, torch.tensor([[0.0, -1.0]]), atol=0.01)


# ===================================================================== #
# 6. PDEParameterEmbeddingBlock
# ===================================================================== #
class TestPDEParameterEmbeddingBlock:
    def test_embedding_shape(self):
        from TensorWAV.physics_blocks.embedding import PDEParameterEmbeddingBlock

        block = PDEParameterEmbeddingBlock(param_dim=3, embed_dim=16, feature_dim=32)
        params = torch.randn(4, 3)
        emb = block(params=params)
        assert emb.shape == (4, 16)

    def test_film_modulation_shape(self):
        from TensorWAV.physics_blocks.embedding import PDEParameterEmbeddingBlock

        block = PDEParameterEmbeddingBlock(param_dim=3, embed_dim=16, feature_dim=32)
        params = torch.randn(4, 3)
        features = torch.randn(4, 10, 32)
        out = block.modulate(features, params)
        assert out.shape == (4, 10, 32)


# ===================================================================== #
# 7. HybridGradientBlock
# ===================================================================== #
class TestHybridGradientBlock:
    def test_gradient_shape(self):
        from TensorWAV.physics_blocks.gradient import HybridGradientBlock

        block = HybridGradientBlock(spatial_dim=2, use_correction=False)
        coords = torch.rand(2, 20, 2, requires_grad=True)
        u = torch.sin(coords[..., 0:1])
        out = block(u, coords)
        assert out.shape == (2, 20, 2)


# ===================================================================== #
# 8. BoundaryConditionBlock
# ===================================================================== #
class TestBoundaryConditionBlock:
    def test_dirichlet_enforcement(self):
        from TensorWAV.physics_blocks.boundary_block import BoundaryConditionBlock

        block = BoundaryConditionBlock(bc_type="dirichlet", bc_value=0.0,
                                        use_correction=False)
        x = torch.linspace(-1, 1, 50).reshape(1, 50, 1)
        u = torch.ones(1, 50, 1)
        out = block(u, coords=x)
        # Boundary values should be pushed toward 0
        assert out[0, 0, 0].item() < 0.5  # near left boundary

    def test_periodic_shape(self):
        from TensorWAV.physics_blocks.boundary_block import BoundaryConditionBlock

        block = BoundaryConditionBlock(bc_type="periodic", use_correction=False)
        x = torch.linspace(-1, 1, 30).reshape(1, 30, 1)
        u = torch.randn(1, 30, 1)
        out = block(u, coords=x)
        assert out.shape == u.shape


# ===================================================================== #
# 9. HamiltonianBlock
# ===================================================================== #
class TestHamiltonianBlock:
    def test_step_shape(self):
        from TensorWAV.physics_blocks.hamiltonian import HamiltonianBlock

        block = HamiltonianBlock(state_dim=1)
        state = torch.randn(4, 2)
        out = block.step(state, dt=0.01)
        assert out.shape == (4, 2)

    def test_hamiltonian_scalar(self):
        from TensorWAV.physics_blocks.hamiltonian import HamiltonianBlock

        block = HamiltonianBlock(state_dim=2)
        q = torch.randn(3, 2)
        p = torch.randn(3, 2)
        H = block.hamiltonian(q, p)
        assert H.shape == (3, 1)


# ===================================================================== #
# 10. SpectralRegularizationBlock
# ===================================================================== #
class TestSpectralRegularizationBlock:
    def test_penalty_higher_for_noisy(self):
        from TensorWAV.physics_blocks.spectral_reg import SpectralRegularizationBlock

        block = SpectralRegularizationBlock(cutoff=4)
        clean = torch.sin(torch.linspace(0, 2 * math.pi, 64)).unsqueeze(0)
        noisy = clean + 0.5 * torch.sin(30 * torch.linspace(0, 2 * math.pi, 64)).unsqueeze(0)
        loss_clean = block(clean)
        loss_noisy = block(noisy)
        assert loss_noisy.item() > loss_clean.item()


# ===================================================================== #
# 11. MultiScaleFourierFeatureBlock
# ===================================================================== #
class TestMultiScaleFourierFeatureBlock:
    def test_output_shape(self):
        from TensorWAV.physics_blocks.multiscale_ff import MultiScaleFourierFeatureBlock

        block = MultiScaleFourierFeatureBlock(input_dim=2, n_scales=3,
                                               features_per_scale=8)
        coords = torch.rand(2, 50, 2)
        out = block(coords=coords)
        expected_dim = 3 * 8 * 2  # n_scales * fp_scale * (sin+cos)
        assert out.shape == (2, 50, expected_dim)


# ===================================================================== #
# 12. SpectralConvBlock
# ===================================================================== #
class TestSpectralConvBlock:
    def test_forward_shape(self):
        from TensorWAV.physics_blocks.spectral_conv import SpectralConvBlock

        block = SpectralConvBlock(in_channels=1, out_channels=2, modes=8)
        u = torch.randn(2, 1, 32)
        out = block(u)
        assert out.shape == (2, 2, 32)


# ===================================================================== #
# 13. HybridTimeStepperBlock
# ===================================================================== #
class TestHybridTimeStepperBlock:
    def test_step_shape(self):
        from TensorWAV.physics_blocks.hybrid_stepper import HybridTimeStepperBlock

        block = HybridTimeStepperBlock(state_dim=3, method="euler")
        u = torch.randn(4, 3)
        out = block.step(u, dt=0.1)
        assert out.shape == (4, 3)

    def test_rk2_accuracy(self):
        from TensorWAV.physics_blocks.hybrid_stepper import HybridTimeStepperBlock

        block = HybridTimeStepperBlock(state_dim=1, method="rk2",
                                        use_correction=False)
        u = torch.tensor([[1.0]])
        for _ in range(10):
            u = block.step(u, dt=0.1, rhs_fn=lambda u: -u)
        expected = math.exp(-1.0)
        assert abs(u.item() - expected) < 0.05


# ===================================================================== #
# 14. AdaptiveSamplingBlock
# ===================================================================== #
class TestAdaptiveSamplingBlock:
    def test_weights_shape_and_sum(self):
        from TensorWAV.physics_blocks.adaptivesampler import AdaptiveSamplingBlock

        block = AdaptiveSamplingBlock(input_dim=2, use_correction=False)
        x = torch.rand(2, 30, 1, requires_grad=True)
        u = torch.sin(x)
        w = block(u, coords=x)
        assert w.shape == (2, 30, 1)
        # Weights should sum to ~1 along N
        assert torch.allclose(w.sum(dim=1), torch.ones(2, 1), atol=0.1)


# ===================================================================== #
# 15. ConservationConstraintBlock
# ===================================================================== #
class TestConservationConstraintBlock:
    def test_mass_conservation(self):
        from TensorWAV.physics_blocks.conservation_block import ConservationConstraintBlock

        block = ConservationConstraintBlock(mode="mass", use_correction=False)
        u = torch.randn(2, 30, 1)
        target = u.mean(dim=1).detach()
        u_proj, viol = block(u, target_quantity=target)
        assert u_proj.shape == u.shape
        assert viol.item() < 1e-6  # Should be exactly conserved


# ===================================================================== #
# 16. OperatorWrapperBlock
# ===================================================================== #
class TestOperatorWrapperBlock:
    def test_forward_shape(self):
        from TensorWAV.physics_blocks.nn_operator_wrapper import OperatorWrapperBlock

        block = OperatorWrapperBlock(grid_shape=(32,), in_channels=1, out_channels=1)
        coords = torch.rand(2, 20, 1)
        u = torch.randn(2, 20, 1)
        out = block(u, coords)
        assert out.shape == (2, 20, 1)


# ===================================================================== #
# Integration: Cross-block pipeline
# ===================================================================== #
class TestCrossBlockPipeline:
    def test_laplacian_spectral_energy_pipeline(self):
        """Chain laplacian → spectral → energy blocks."""
        from TensorWAV.physics_blocks.laplacian import HybridLaplacianBlock
        from TensorWAV.physics_blocks.spectral import SpectralHybridBlock
        from TensorWAV.physics_blocks.energy import EnergyAwareBlock

        # 1) Laplacian on grid
        lap_block = HybridLaplacianBlock(mode="grid", spatial_dim=1,
                                          use_correction=False)
        u_grid = torch.randn(1, 1, 64)
        lap_out = lap_block(u_grid)
        assert lap_out.shape == (1, 1, 64)

        # 2) Spectral filtering on the Laplacian output
        spec_block = SpectralHybridBlock(n_modes=64, cutoff=16,
                                          use_correction=False)
        spec_out = spec_block(lap_out.squeeze(1))  # (1, 64)
        assert spec_out.shape == (1, 64)

        # 3) Energy check — need point format
        energy_block = EnergyAwareBlock(c=1.0, spatial_dim=1)
        coords = torch.rand(1, 64, 2, requires_grad=True)
        u_point = torch.sin(coords[..., 0:1])
        corrected, penalty = energy_block(u_point, coords)
        assert corrected.shape == u_point.shape
        assert penalty.item() >= 0

    def test_gradient_residual_pipeline(self):
        """Chain gradient + wave residual blocks."""
        from TensorWAV.physics_blocks.gradient import HybridGradientBlock
        from TensorWAV.physics_blocks.residual import HybridWaveResidualBlock

        coords = torch.rand(1, 30, 2, requires_grad=True)
        u = torch.sin(coords[..., 0:1])

        grad_block = HybridGradientBlock(spatial_dim=2, use_correction=False)
        grad_u = grad_block(u, coords)
        assert grad_u.shape == (1, 30, 2)

        res_block = HybridWaveResidualBlock(use_correction=False)
        res = res_block(u, coords)
        assert res.shape == (1, 30, 1)

    def test_embedding_modulates_spectral_conv(self):
        """Embedding block modulates spectral conv features."""
        from TensorWAV.physics_blocks.embedding import PDEParameterEmbeddingBlock
        from TensorWAV.physics_blocks.spectral_conv import SpectralConvBlock

        emb_block = PDEParameterEmbeddingBlock(param_dim=2, embed_dim=8,
                                                 feature_dim=16)
        spec_conv = SpectralConvBlock(in_channels=1, out_channels=1, modes=4)

        params = torch.randn(2, 2)
        u = torch.randn(2, 1, 32)
        out = spec_conv(u)
        assert out.shape == (2, 1, 32)

        # Reshape for FiLM
        features = out.squeeze(1).unsqueeze(-1).expand(-1, -1, 16)  # (2,32,16)
        modulated = emb_block.modulate(features, params)
        assert modulated.shape == (2, 32, 16)

    def test_stepper_with_oscillator(self):
        """Chain oscillator dynamics through the time stepper."""
        from TensorWAV.physics_blocks.oscillator import HybridOscillatorBlock
        from TensorWAV.physics_blocks.hybrid_stepper import HybridTimeStepperBlock

        osc = HybridOscillatorBlock(omega=1.0, alpha=0.0, use_correction=False)
        stepper = HybridTimeStepperBlock(state_dim=2, method="rk2",
                                          use_correction=False)
        state = torch.tensor([[1.0, 0.0]])
        for _ in range(10):
            state = stepper.step(state, dt=0.01, rhs_fn=osc.forward)
        assert state.shape == (1, 2)
        # Should still be close to initial energy
        energy = 0.5 * (state[0, 0] ** 2 + state[0, 1] ** 2)
        assert abs(energy.item() - 0.5) < 0.1
