# NeuralWave Core

A research-grade PyTorch sub-framework for physics-informed neural networks (PINNs) and operator learning.

## Features

- **Models**: MLP, Fourier MLP, SIREN, Fourier Neural Operator (FNO)
- **Physics**: PDE specification, auto-differentiation residuals, boundary conditions
- **Solvers**: Finite difference and spectral solvers for validation
- **Training**: Unified engine for PINN and operator learning with mixed precision
- **Diagnostics**: Metrics, energy functionals, spectral analysis
- **IO**: Checkpointing, TorchScript/ONNX export, model cards

## Installation

```bash
pip install git+https://github.com/your-repo/TensorWAV.git#subdirectory=TensorWAV
```

Or for local development:

```bash
cd TensorWAV
pip install -e .
```

## Quick Start

### Training a PINN

```bash
python -m TensorWAV.cli --config TensorWAV/configs/demo_pinn_1d.yaml
```

### Running Tests

```bash
pytest TensorWAV/tests
```

### Example: Prediction and Plotting

```bash
python TensorWAV/examples/predict_and_plot.py
```

## Project Structure

```
TensorWAV/
├── models/          # Neural network architectures
├── physics/         # PDE specifications and residuals
├── datasets/        # Data generators
├── solvers/         # Numerical solvers
├── training/        # Training engine and callbacks
├── operators/       # Operator learning utilities
├── io/              # Checkpointing and export
├── diagnostics/     # Metrics and analysis tools
├── configs/         # Example configurations
├── examples/        # Example scripts
└── tests/           # Unit tests
```

## Configuration

Example YAML config for PINN training:

```yaml
name: demo_pinn
task: physics_informed_neural_network
model:
  type: mlp
  input_dim: 2
  output_dim: 1
  hidden_layers: [50, 50, 50]
  activation: tanh
training:
  epochs: 100
  learning_rate: 0.001
  save_dir: ./checkpoints
  save_freq: 10
```

## Testing

All modules include unit tests. Run the full test suite:

```bash
pytest TensorWAV/tests -v
```

Run specific test modules:

```bash
pytest TensorWAV/tests/test_models.py
pytest TensorWAV/tests/test_physics.py
```

## License

MIT License
