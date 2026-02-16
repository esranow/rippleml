import argparse
import sys
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (Path): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main() -> None:
    """
    Main entry point for NeuralWave Core CLI.
    Parses arguments and initiates training.
    """
    parser = argparse.ArgumentParser(description="NeuralWave Core CLI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        logging.error(f"Config file not found at {config_path}")
        sys.exit(1)
        
    config = load_config(config_path)
    logging.info(f"Loaded configuration from {config_path}")
    
    # Import here to avoid circular imports
    from TensorWAV.training.engine import train_from_config
    
    # Delegate to training engine
    train_from_config(config)

if __name__ == "__main__":
    main()
