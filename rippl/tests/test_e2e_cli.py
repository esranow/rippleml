import os
import yaml
import pytest
import torch
import shutil
from rippl.cli import main
from unittest.mock import patch
import sys

def test_cli_train_export_e2e(tmp_path):
    """
    End-to-End test for CLI:
    1. Create a heat equation config.
    2. Run 'rippl train'.
    3. Run 'rippl export'.
    4. Verify all artifacts exist.
    """
    # 1. Setup temporary directory and config
    test_dir = tmp_path / "rippl_test"
    test_dir.mkdir()
    config_path = test_dir / "heat_config.yaml"
    output_dir = test_dir / "model_out"
    
    config = {
        "geometry": {
            "spatial_dims": 2,
            "bounds": [[0, 1], [0, 1]],
            "resolution": [20, 20]
        },
        "physics": {
            "fields": ["u"],
            "equation": [
                [1.0, "timederivative", {"order": 1, "field": "u"}],
                [-0.01, "laplacian", {"field": "u", "spatial_dims": 1}]
            ],
            "constraints": [
                {"type": "dirichlet", "field": "u", "coords": [[0.0, 0.0]], "value": 0.0},
                {"type": "dirichlet", "field": "u", "coords": [[1.0, 0.0]], "value": 0.0},
                {"type": "initial", "field": "u", "coords": [[0.5, 0.0]], "value": 1.0}
            ]
        },
        "model": {
            "name": "mlp",
            "config": {
                "input_dim": 2, # (x, t)
                "output_dim": 1,
                "hidden_layers": [16, 16]
            }
        },
        "training": {
            "epochs": 5,
            "lr": 1e-3
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # 2. Execute 'rippl train'
    train_args = ["rippl", "train", str(config_path), "--output", str(output_dir)]
    with patch.object(sys, 'argv', train_args):
        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"CLI train failed with exit code {e.code}")

    # Verify training artifacts
    assert os.path.exists(os.path.join(output_dir, "weights.pt"))
    assert os.path.exists(os.path.join(output_dir, "config.json"))

    # 3. Execute 'rippl export'
    export_args = ["rippl", "export", str(config_path), str(output_dir), "--format", "torchscript"]
    with patch.object(sys, 'argv', export_args):
        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"CLI export failed with exit code {e.code}")

    # Verify export artifacts
    assert os.path.exists(os.path.join(output_dir, "model.pt"))
    assert os.path.exists(os.path.join(output_dir, "model_card.json"))
    
    print("\n[E2E TEST] CLI train and export passed successfully.")

if __name__ == "__main__":
    # If run directly, use a local 'temp_test' dir
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_train_export_e2e(Path(tmp))
