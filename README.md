# rippl
**The physics compiler for neural networks.**

[![Tests](https://github.com/esranow/rippl/actions/workflows/ci.yml/badge.svg)](https://github.com/esranow/rippl/actions)
[![PyPI](https://img.shields.io/pypi/v/rippl)](https://pypi.org/project/rippl)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

→ **[Full Documentation](docs/)** · [lwly.io](https://lwly.io) · [Benchmarks](rippl/benchmarks/README.md)

---
## What it is
rippl is a physics compiler. You write the PDE. rippl solves it.
It sits between you and PyTorch — handling residual computation, 
boundary enforcement, causal training, and adaptive loss balancing 
so you don't have to.

---
## 7 Lines
```python
import rippl as rp
import rippl.nn as rnn

model  = rnn.MLP(in_dim=2, out_dim=1)
engine = rp.compile(model)
result = rp.run(domain, equation, engine,
                causal=True, adaptive_loss=True, hard_bcs=True)
```

## Install
```bash
pip install rippl
```

## Physics
```python
from rippl.core.system import System, Domain, Constraint
from rippl.core.equation import Equation
from rippl.physics.operators import TimeDerivative, Diffusion

# Heat equation: ∂u/∂t = 0.1 ∂²u/∂x²
system = System(
    equation=Equation([TimeDerivative(1), Diffusion(alpha=0.1)]),
    domain=Domain(spatial_dims=1, bounds=[(0,1),(0,1)]),
    constraints=[...],
    fields=["u"]
)
```

**Supported:** Wave · Heat · Burgers · Navier-Stokes · Elasticity ·
Schrödinger · Allen-Cahn · Cahn-Hilliard · Eikonal · Hamilton-Jacobi ·
Turing patterns · FitzHugh-Nagumo · Brusselator · Fractional PDEs

## Features
- **Causal training** — wave equation at L2=8.38e-05. Diverges without it.
- **NTK adaptive weighting** — 3x improvement over fixed weights
- **Hard BCs** — exact Dirichlet enforcement via ansatz, no penalty tuning
- **Spectral collocation** — Chebyshev/Legendre for smooth problems
- **Digital Twin** — identify PDE parameters from sensor CSV data
- **UQ** — MC Dropout confidence intervals
- **CSG geometry** — annulus, sphere, boolean operations
- **Auto-Migrate** — transpile DeepXDE scripts to rippl

## Multi-GPU
Multi-GPU training and managed cloud compute available at
[lwly.io](https://lwly.io)

## License
Apache 2.0 — see [LICENSE](LICENSE)
