"""
ripple: Training Package
"""
from ripple.training.engine import train_from_config
from ripple.training.callbacks import CheckpointCallback

__all__ = ["train_from_config", "CheckpointCallback"]
