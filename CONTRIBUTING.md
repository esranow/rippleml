# Contributing to NeuralWave Core

Thank you for your interest in contributing to NeuralWave Core!

## Development Setup

1. Clone the repository
2. Install in development mode:
   ```bash
   cd TensorWAV
   pip install -e .
   pip install pytest
   ```

## Code Standards

All contributions must follow these strict guidelines:

1. **Modular architecture** - Keep components decoupled
2. **Type hints everywhere** - All functions must have type annotations
3. **Docstrings** - All public functions/classes must have docstrings
4. **Limited imports** - Only import from:
   - torch, torch.nn, torch.fft
   - numpy, math
   - typing, dataclasses
   - pathlib, yaml, logging
   - matplotlib.pyplot
   - pytest (tests only)
5. **No placeholder logic** - Core features must be fully implemented
6. **CPU-friendly tests** - All tests must run on CPU
7. **Deterministic tests** - Set seeds for reproducibility
8. **Use pathlib** - For all file path operations
9. **Multi-dimensional support** - Tensor logic must support dim=1, 2, or 3

## Testing

Every module must include unit tests:

```bash
pytest TensorWAV/tests
```

Run specific test files:
```bash
pytest TensorWAV/tests/test_models.py -v
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code standards
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Questions?

Open an issue for questions or discussions about contributions.
