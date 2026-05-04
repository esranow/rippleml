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
import rippl
import rippl.nn as nn

model  = nn.MLP(in_dim=2, out_dim=1)
engine = rippl.compile(model)
result = rippl.run(domain, equation, engine,
                causal=True, adaptive_loss=True, hard_bcs=True)
```

---
## Install
```bash
pip install rippl
```

---
## Multi-GPU & Enterprise
Multi-GPU DDP training, `auto.rippl` tokenized orchestrators, and 3D STL mesh ingestion are handled via the `rippl-pro` proprietary extension. 
API keys and cloud compute available at **[lwly.io](https://lwly.io)**

---
## License
Apache 2.0 — see [LICENSE](LICENSE)
