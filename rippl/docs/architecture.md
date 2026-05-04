# Rippl
### Differentiable Physics Engine for PyTorch

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![PyPI Version](https://img.shields.io/badge/pypi-v0.0.1-blue)
![License](https://img.shields.io/badge/license-MIT-orange)
![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)

---

**Rippl** is a PyTorch-native, C++-less scientific machine learning framework designed for researchers who are tired of bloated dependencies and slow development cycles. It bridges the gap between **Physics-Informed Neural Networks (PINNs)** and **Discrete Numerical Simulation (FDM)** into a unified, Pydantic-validated API.

## Why Rippl? (The Hook)

Most SciML libraries suffer from "Academic Bloat": opaque C++ bindings, fragile custom CUDA kernels, and gradient pathologies that make training feel like alchemy. Rippl cuts through the noise:

- **Strictly Native PyTorch**: No custom C++ or CUDA. We use `F.conv2d` discrete stencils for performance, meaning it runs anywhere PyTorch does—from your MacBook to an H100 cluster.
- **The $O(1)$ Autograd Cache**: Our `requires_derived` system pre-caches tensors and graph paths. Stop wasting VRAM on redundant graph traversals during complex PDE residual computations.
- **Algebraic Hard Constraints**: Forget "soft" boundary losses that clash with your physics. Rippl enforces Dirichlet and Neumann conditions algebraically using distance-field masking, ensuring zero-error boundaries from step one.
- **Differentiable Simulation**: Backprop through time (BPTT) using standard FDM stencils mapped directly to convolution operations.

---

## Installation

```bash
pip install git+https://github.com/esranow/rippleml.git
```

*For local development:*
```bash
git clone https://github.com/esranow/rippl.git
cd rippl && pip install -e .
```

---

## The 10-Line Quickstart

Solve a 1D Wave Equation ($u_{tt} - c^2 u_{xx} = 0$) with zero-gradient overhead:

```python
import torch
from rippl.core import System, Domain, Experiment
from rippl.physics.operators import Laplacian, TimeDerivative
from rippl.physics.equation import Equation

# 1. Define the Physics (u_tt - 1.0 * u_xx = 0)
wave_eq = Equation([
    (1.0, TimeDerivative(order=2)), 
    (-1.0, Laplacian())
])

# 2. Setup the System and Domain
domain = Domain(spatial_dims=1, bounds=[(0, 1)], resolution=(100,))
system = System(equation=wave_eq, domain=domain, fields=["u"])

# 3. Train the PINN
model = torch.nn.Sequential(torch.nn.Linear(2, 50), torch.nn.Tanh(), torch.nn.Linear(50, 1))
exp = Experiment(system, model, torch.optim.Adam(model.parameters()))
exp.train(epochs=100)
```

---

## Deep Dive: Core Architecture

### 1. The Autograd Cache (`requires_derived`)
In standard PINNs, calculating $u_{xxt}$ requires multiple passes through the autograd graph. Rippl's engine analyzes the `Operator` signatures in your `EquationSystem` and builds a directed acyclic graph (DAG) of required derivatives. These are computed in a single optimized pass and cached in the `derived` dictionary, slashing the memory footprint of high-order PDEs.

### 2. Algebraic Boundaries
Soft boundary losses are the primary cause of PINN instability. Rippl uses a `HardConstraintWrapper` combined with a `DistanceFunction` $D(x)$:
$$u_{pred}(x) = G(x) \cdot u_{model}(x) + B(x)$$
Where $G(x)$ vanishes at the boundary and $B(x)$ enforces the exact value. The result? **The optimizer never sees the boundary; it only sees the residual.**

### 3. Differentiable FDM Convolutions
Standard Finite Difference Methods (FDM) are usually isolated from ML training. Rippl maps these stencils to `torch.nn.functional.conv2d`. This allows you to use a traditional solver as a "Layer" in your network, enabling hybrid architectures that are both numerically stable and learnable.

---

## Advanced Physics Capabilities

- **Spectral Bias Mitigation**: Native support for **SIRENs** (Sinusoidal Representation Networks) and **Fourier Neural Operators (FNOs)** to capture high-frequency shock fronts.
- **Conservation Law Enforcement**: Wrappers for enforcing flux conservation and divergence-free fields (e.g., in Navier-Stokes).
- **Inverse Problems**: Parameters (like viscosity $\nu$ or wave speed $c$) can be marked as `requires_grad`, allowing the system to discover physics from raw data automatically.

---

## "Vibecoding" & Contribution Philosophy

Rippl is built for **speed and transparency**. Our codebase is intentionally monolithic and terse. We prioritize architectural elegance over feature-creep.

**The Contract:**
1. **Strict Perimeter**: All inputs are validated via Pydantic. If your dimensions don't match your physics, Rippl fails fast with a clear error message.
2. **Raw Core**: Inside the loop, it's pure, unadulterated PyTorch. No abstraction leaks. No hidden overhead.

If you want to contribute, keep it dry. No boilerplate. No unnecessary classes. If it can be done with a tensor operation, do it with a tensor operation.

---

## License
MIT. Go build something cool.
