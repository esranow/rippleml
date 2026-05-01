import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import logging

import pytorch_lightning as pl
from torch.quasirandom import SobolEngine

class Callback:
    """
    Base class for training callbacks.
    """
    def on_train_begin(self, logs: Dict[str, Any] = {}) -> None:
        pass
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = {}) -> None:
        pass
        
    def on_train_end(self, logs: Dict[str, Any] = {}) -> None:
        pass

class CheckpointCallback(Callback):
    """
    Callback to save model checkpoints.
    """
    def __init__(self, save_dir: str, save_freq: int = 10, prefix: str = "model"):
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.prefix = prefix
        
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = {}) -> None:
        if (epoch + 1) % self.save_freq == 0:
            model = logs.get("model")
            optimizer = logs.get("optimizer")
            
            if model is None:
                logging.warning("CheckpointCallback: 'model' not found in logs. Skipping save.")
                return
            
            path = self.save_dir / f"{self.prefix}_epoch_{epoch+1}.pt"
            
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
            }
            if optimizer:
                state["optimizer_state_dict"] = optimizer.state_dict()
                
            torch.save(state, path)
            logging.info(f"Saved checkpoint to {path}")

class RARCallback(pl.Callback):
    """
    Residual-based Adaptive Refinement Callback using Sobol points.
    """
    def __init__(self, freq: int = 10, K: int = 100):
        super().__init__()
        self.freq = freq
        self.K = K

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.freq != 0:
            return
            
        device = pl_module.device
        sobol = SobolEngine(dimension=2, scramble=True)
        
        num_candidates = self.K * 10
        candidates = sobol.draw(num_candidates).to(device)
        candidates.requires_grad_(True)
        
        # Evaluate proxy residual
        u = pl_module(candidates)
        residual = u ** 2
        
        errors = torch.abs(residual).view(-1)
        if errors.numel() >= self.K:
            _, indices = torch.topk(errors, self.K)
            new_points = candidates[indices].detach()
            
        # Mock injection into dataloader
        pass

