import pytest
import torch
import shutil
from pathlib import Path
from TensorWAV.training.engine import train_from_config

# Clean up checkoints after test
@pytest.fixture
def clean_checkpoints():
    yield
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")

def test_train_pinn_flow(clean_checkpoints):
    """
    Test training loop in PINN mode (approx).
    """
    config = {
        "task": "physics_informed_neural_network",
        "model": {
            "type": "mlp",
            "input_dim": 2, # x, t
            "output_dim": 1,
            "hidden_layers": [16, 16],
            "activation": "tanh"
        },
        "training": {
            "epochs": 2,
            "save_freq": 1,
            "save_dir": "test_checkpoints",
            "learning_rate": 1e-3,
            "device": "cpu"
        },
        "data": {
            "domain": [-1.0, 1.0]
        }
    }
    
    train_from_config(config)
    
    # Assert checkpoint exists
    assert Path("test_checkpoints/model_epoch_1.pt").exists()
    assert Path("test_checkpoints/model_epoch_2.pt").exists()

def test_train_operator_flow(clean_checkpoints):
    """
    Test training loop in Operator Learning mode.
    """
    config = {
        "task": "operator_learning",
        "model": {
            "type": "fno1d", # simple operator model
            "input_dim": 1,
            "output_dim": 1,
            "modes": 4,
            "width": 8,
            "depth": 1
        },
        "training": {
            "epochs": 2,
            "save_freq": 2,
            "save_dir": "test_checkpoints",
            "device": "cpu"
        },
        "data": {
            "resolution": [32]
        }
    }
    
    train_from_config(config)
    
    # Epoch 2 saved
    assert Path("test_checkpoints/model_epoch_2.pt").exists()
