# Physics Blocks — Hybrid Physics-Aware Neural Blocks

A modular library of `nn.Module` blocks that combine fixed physics operators
(analytic / discrete) with small learnable correction networks (MLP / conv).

## Quick Start

```python
from ripple.physics_blocks import (
    HybridLaplacianBlock,
    SpectralHybridBlock,
    EnergyAwareBlock,
    BLOCK_REGISTRY,
)

# Instantiate a block
lap = HybridLaplacianBlock(mode="point", use_correction=True)

# All blocks follow the same API
output = lap(u, coords=coords, params=params)
```

## Available Blocks

| # | Module | Class | Description |
|---|--------|-------|-------------|
| 1 | `laplacian.py` | `HybridLaplacianBlock` | Autograd / FD Laplacian + MLP |
| 2 | `residual.py` | `HybridWaveResidualBlock` | Wave PDE residual + NN correction |
| 3 | `spectral.py` | `SpectralHybridBlock` | FFT filtering + learnable frequency weights |
| 4 | `energy.py` | `EnergyAwareBlock` | Energy computation + gating correction |
| 5 | `oscillator.py` | `HybridOscillatorBlock` | Duffing oscillator + learnable ω/damping |
| 6 | `embedding.py` | `PDEParameterEmbeddingBlock` | PDE parameter → FiLM modulation |
| 7 | `gradient.py` | `HybridGradientBlock` | Autograd ∇u + MLP |
| 8 | `boundary_block.py` | `BoundaryConditionBlock` | BC enforcement + local NN |
| 9 | `hamiltonian.py` | `HamiltonianBlock` | Symplectic Hamiltonian dynamics |
| 10 | `spectral_reg.py` | `SpectralRegularizationBlock` | FFT-based high-freq penalty |
| 11 | `multiscale_ff.py` | `MultiScaleFourierFeatureBlock` | Multi-scale Fourier features with gating |
| 12 | `spectral_conv.py` | `SpectralConvBlock` | FNO-style spectral conv + local conv |
| 13 | `hybrid_stepper.py` | `HybridTimeStepperBlock` | Euler/RK2 + NN corrector |
| 14 | `adaptivesampler.py` | `AdaptiveSamplingBlock` | Residual-based sampling weights |
| 15 | `conservation_block.py` | `ConservationConstraintBlock` | Conserved-quantity projection |
| 16 | `nn_operator_wrapper.py` | `OperatorWrapperBlock` | Grid ↔ point adaptor |

## Block API Contract

Every block follows:

```python
class SomeHybridBlock(nn.Module):
    def __init__(self, physics_config, nn_config):
        ...
    def forward(self, u, coords=None, params=None):
        physics_term = self.compute_physics(u, coords, params)
        correction = self.correction_net(self._prep_for_nn(u, coords, params))
        return physics_term + correction
```

## Config-Driven Usage

Blocks can be inserted into the training loop via YAML config:

```yaml
blocks:
  - type: spectral
    n_modes: 64
    cutoff: 16
  - type: energy
    c: 1.0
```

## CLI

```bash
python -m ripple.cli --config ripple/configs/demo_blocks_playground.yaml
```

## Tests

```bash
pytest ripple/tests/test_physics_blocks.py -v
```

## Related Papers

- **Paper A**: [Addressing Spectral Bias via Multi-Grade Deep Learning (NeurIPS 2024)](https://arxiv.org/abs/2307.07321)
- **Paper B**: [Amortized Fourier Neural Operators (NeurIPS 2024)](https://arxiv.org/abs/2404.12356)
- **Paper C**: [Factorized Fourier Neural Operators for Wave Propagation](https://arxiv.org/abs/2111.13802)
