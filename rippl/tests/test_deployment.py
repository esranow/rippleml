import torch, os, shutil
import torch.nn as nn
from rippl.logging.logger import RipplLogger
from rippl.export.exporter import export_model

class TestMLP(nn.Module):
    def __init__(self, d_in=2, d_out=1):
        super().__init__()
        self.net = nn.Linear(d_in, d_out)
    def forward(self, x): return self.net(x)

def test_registry():
    cfg = {"d_in": 2, "d_out": 1}
    m = TestMLP(**cfg)
    assert isinstance(m, TestMLP)
    
    path = "test_model_dir"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/config.json", "w") as f:
        import json
        json.dump({"name": "test_mlp", "model_config": cfg}, f)
    torch.save(m.state_dict(), f"{path}/weights.pt")
    
    m2 = TestMLP(**cfg)
    m2.load_state_dict(torch.load(f"{path}/weights.pt", weights_only=True))
    assert isinstance(m2, TestMLP)
    shutil.rmtree(path)

def test_logging():
    log_dir = "test_logs"
    logger = RipplLogger(path=log_dir)
    logger.log_epoch(0, {"loss": 0.5})
    assert os.path.exists(f"{log_dir}/metrics.json")
    shutil.rmtree(log_dir)

def test_export():
    m = TestMLP()
    export_dir = "test_export"
    export_model(m, export_dir, format="torchscript", metadata={"pde": "laplace"})
    assert os.path.exists(f"{export_dir}/model.pt")
    assert os.path.exists(f"{export_dir}/model_card.json")
    shutil.rmtree(export_dir)

if __name__ == "__main__":
    test_registry()
    test_logging()
    test_export()
    print("Deployment tests passed.")
