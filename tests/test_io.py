import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import json
from pathlib import Path
from TensorWAV.io.checkpoint import save_checkpoint, load_checkpoint
from TensorWAV.io.export import export_torchscript, export_onnx, write_model_card

@pytest.fixture
def clean_io_dir():
    test_dir = Path("test_io_output")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_checkpoint_flow(clean_io_dir):
    """
    Test saving and loading a checkpoint.
    """
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    path = clean_io_dir / "ckpt.pt"
    
    # Save
    save_checkpoint(model, optimizer, epoch=1, path=path)
    assert path.exists()
    
    # Modify model to ensure loading creates change
    with torch.no_grad():
        model.weight.fill_(0.0)
        
    # Load
    load_checkpoint(model, optimizer, path)
    
    # Check weights are not zero (random init restored)
    assert not torch.allclose(model.weight, torch.zeros_like(model.weight))

def test_export_functions(clean_io_dir):
    """
    Test TorchScript and ONNX export.
    """
    model = nn.Linear(10, 2)
    sample_input = torch.randn(1, 10)
    
    ts_path = clean_io_dir / "model.pt"
    export_torchscript(model, sample_input, ts_path)
    assert ts_path.exists()
    
    onnx_path = clean_io_dir / "model.onnx"
    export_onnx(model, sample_input, onnx_path)
    assert onnx_path.exists()

def test_model_card(clean_io_dir):
    """
    Test model card writing.
    """
    config = {
        "model": {"type": "mlp"},
        "training": {"epochs": 10}
    }
    metrics = {"loss": 0.1}
    path = clean_io_dir / "model_card.json"
    
    write_model_card(config, metrics, path)
    
    assert path.exists()
    with open(path, 'r') as f:
        data = json.load(f)
        assert data["metrics"]["loss"] == 0.1
