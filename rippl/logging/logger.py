import json
import os
import torch
from typing import Dict, Any, List, Optional

class RipplLogger:
    """
    Standard logging infrastructure for the Rippl framework.
    Supports local JSON logging of loss curves and parameter histories,
    with optional Weights & Biases (wandb) integration.
    """
    def __init__(self, path: str = "logs", use_wandb: bool = False, project: str = "rippl"):
        """
        Initialize the logger.

        Args:
            path: Directory path where local logs will be saved.
            use_wandb: Whether to initialize Weights & Biases tracking.
            project: The wandb project name.
        """
        self.path = path
        os.makedirs(path, exist_ok=True)
        
        self.metrics_history: List[Dict[str, Any]] = []
        self.parameter_history: Dict[str, List[Any]] = {}
        
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project)
            except ImportError:
                # Silently skip wandb if not installed
                pass

    def log_epoch(self, epoch: int, losses: Dict[str, float], parameters: Optional[Dict[str, torch.Tensor]] = None):
        """
        Record metrics for a training epoch.

        Args:
            epoch: The current epoch number.
            losses: A dictionary of loss components (e.g., {'pde': 0.1, 'bc': 0.05}).
            parameters: Optional dictionary of nn.Parameter tensors to track (e.g., for inverse problems).
        """
        log_entry = {"epoch": epoch, **losses}
        self.metrics_history.append(log_entry)
        
        # Track parameter evolution for inverse/DigitalTwin problems
        if parameters:
            for name, tensor in parameters.items():
                if name not in self.parameter_history:
                    self.parameter_history[name] = []
                self.parameter_history[name].append(tensor.detach().cpu().tolist())
        
        # Sync with wandb if active
        if self.wandb_run:
            self.wandb_run.log(log_entry)
        
        # Periodically save to local JSON to prevent data loss
        self._save_to_disk()

    def _save_to_disk(self):
        """
        Internal helper to write recorded data to local JSON files.
        """
        metrics_file = os.path.join(self.path, "metrics.json")
        params_file = os.path.join(self.path, "params.json")
        
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
            
        if self.parameter_history:
            with open(params_file, "w") as f:
                json.dump(self.parameter_history, f, indent=2)
