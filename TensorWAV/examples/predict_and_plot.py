import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

from TensorWAV.models.registry import build_model
from TensorWAV.io.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO)

def predict_and_plot(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    output_path: str = "prediction.png"
) -> None:
    """
    Load a trained model and generate a prediction plot.
    
    Args:
        checkpoint_path (str): Path to model checkpoint.
        model_config (Dict[str, Any]): Model configuration.
        output_path (str): Path to save the output plot.
    """
    # Build model
    model_name = model_config.pop("type")
    model = build_model(model_name, model_config)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, None, checkpoint_path)
    logging.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    # Generate test inputs (1D example)
    # Assume model takes (B, N, input_dim) -> (B, N, output_dim)
    N = 100
    x = torch.linspace(-1, 1, N).view(1, N, 1)
    t = torch.zeros(1, N, 1)
    inputs = torch.cat([x, t], dim=-1)  # (1, N, 2)
    
    # Predict
    with torch.no_grad():
        outputs = model(inputs)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x.squeeze().numpy(), outputs.squeeze().numpy(), 'b-', linewidth=2, label='Prediction')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x, t=0)', fontsize=12)
    ax.set_title('Model Prediction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logging.info(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "checkpoints/model_epoch_10.pt"
    model_config = {
        "type": "mlp",
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": [50, 50, 50],
        "activation": "tanh"
    }
    
    if Path(checkpoint_path).exists():
        predict_and_plot(checkpoint_path, model_config)
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}. Skipping prediction.")
        logging.info("Train a model first using: python -m TensorWAV.cli --config configs/demo_pinn_1d.yaml")
