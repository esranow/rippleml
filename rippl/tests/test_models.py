import pytest
import torch
import numpy as np
import random
import rippl.nn as rnn

# Set deterministic flags
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)

@pytest.fixture(scope="module")
def common_config():
    return {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": [32, 32],
        "batch_size": 4,
        "num_points": 100
    }

def test_mlp(common_config):
    model = rnn.MLP(**{
        "input_dim": common_config["input_dim"],
        "output_dim": common_config["output_dim"],
        "hidden_layers": common_config["hidden_layers"],
        "activation": "tanh"
    })
    
    B, N = common_config["batch_size"], common_config["num_points"]
    x = torch.randn(B, N, common_config["input_dim"])
    y = model(x)
    assert y.shape == (B, N, common_config["output_dim"])

def test_fourier_mlp(common_config):
    model = rnn.FourierMLP(**{
        "input_dim": common_config["input_dim"],
        "output_dim": common_config["output_dim"],
        "hidden_layers": common_config["hidden_layers"],
        "sigma": 1.0,
        "mapping_size": 64
    })
    
    B, N = common_config["batch_size"], common_config["num_points"]
    x = torch.randn(B, N, common_config["input_dim"])
    y = model(x)
    assert y.shape == (B, N, common_config["output_dim"])

def test_siren(common_config):
    model = rnn.Siren(**{
        "input_dim": common_config["input_dim"],
        "output_dim": common_config["output_dim"],
        "hidden_layers": common_config["hidden_layers"],
        "omega_0": 30.0
    })
    
    B, N = common_config["batch_size"], common_config["num_points"]
    x = torch.randn(B, N, common_config["input_dim"])
    y = model(x)
    assert y.shape == (B, N, common_config["output_dim"])

def test_fno1d(common_config):
    model = rnn.FNO1d(**{
        "input_dim": common_config["input_dim"],
        "output_dim": common_config["output_dim"],
        "modes": 16,
        "width": 32,
        "depth": 2
    })
    
    B, N = common_config["batch_size"], 100
    x = torch.randn(B, N, common_config["input_dim"])
    y = model(x)
    assert y.shape == (B, N, common_config["output_dim"])

def test_fno2d():
    res = [32, 32]
    N = res[0] * res[1]
    input_dim = 3
    output_dim = 1
    
    model = rnn.FNO2d(**{
        "input_dim": input_dim,
        "output_dim": output_dim,
        "modes1": 8,
        "modes2": 8,
        "width": 32,
        "depth": 2,
        "resolution": res
    })
    
    B = 2
    x = torch.randn(B, N, input_dim)
    y = model(x)
    assert y.shape == (B, N, output_dim)
