import torch
import os
from typing import Union, Dict, Any, Optional, Tuple
from rippl.core.config import ConfigParser
from rippl.core.system import System
from rippl.core.experiment import Experiment
from rippl.nn.registry import build_model, load_model
from rippl.export.exporter import export_model

def train(config_path_or_dict: Union[str, Dict[str, Any]]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Train a physics-informed neural network model based on the provided configuration.

    Args:
        config_path_or_dict: Path to a YAML/JSON config file or a configuration dictionary.

    Returns:
        A tuple of (trained_model, training_results).
    """
    config = ConfigParser.load(config_path_or_dict)
    system = System.from_config(config)
    
    # Model configuration
    model_cfg = config.get("model", {})
    if not model_cfg:
        raise ValueError("Configuration missing 'model' section.")
        
    model = build_model(model_cfg["name"], model_cfg["config"])
    
    # Training configuration
    train_cfg = config.get("training", {})
    if not train_cfg:
        raise ValueError("Configuration missing 'training' section.")
        
    optimizer_name = train_cfg.get("optimizer", "Adam")
    learning_rate = train_cfg.get("lr", 1e-3)
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
    # Experiment orchestration
    experiment_kwargs = train_cfg.get("experiment_kwargs", {})
    experiment = Experiment(system, model, optimizer, **experiment_kwargs)
    
    # Coordinates for training
    # Default to a dense grid from the domain if not specified
    if "data" in config and "coords" in config["data"]:
        coords = torch.tensor(config["data"]["coords"], dtype=torch.float32)
    else:
        coords, _ = system.domain.build_grid()
        coords = coords.reshape(-1, coords.shape[-1])
    
    # Optional logger
    logger = None
    if "logging" in config:
        from rippl.logging.logger import RipplLogger
        logger = RipplLogger(**config["logging"])
        
    results = experiment.train(
        coords, 
        epochs=train_cfg.get("epochs", 100),
        ntk_freq=train_cfg.get("ntk_freq", 500),
        patience=train_cfg.get("patience", 200),
        logger=logger
    )
    
    return model, results

def simulate(config_path_or_dict: Union[str, Dict[str, Any]]) -> torch.Tensor:
    """
    Run a trained model or a numerical solver for inference.

    Args:
        config_path_or_dict: Path to a configuration file or a dictionary.

    Returns:
        A torch.Tensor containing the simulated field values.
    """
    config = ConfigParser.load(config_path_or_dict)
    
    # If a model path is provided in config, load it.
    # Otherwise, check if an FD solver is requested.
    if "model_path" in config:
        model = load_model(config["model_path"])
        system = System.from_config(config)
        coords, _ = system.domain.build_grid()
        # Flatten and evaluate
        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, coords.shape[-1])
        with torch.no_grad():
            output = model(coords_flat)
        # Reshape back to grid
        if isinstance(output, dict):
            return {k: v.view(*original_shape, 1) for k, v in output.items()}
        return output.view(*original_shape, 1)
    
    elif "solver" in config:
        from rippl.core.config import get_solver_fn
        solver_name = config["solver"]["name"]
        solver_fn = get_solver_fn(solver_name)
        return solver_fn(**config["solver"].get("kwargs", {}))
        
    raise ValueError("Simulation requires either 'model_path' or 'solver' in config.")

def identify(config_path_or_dict: Union[str, Dict[str, Any]], data_path: str) -> Dict[str, Any]:
    """
    Run parameter discovery (Digital Twin) using sensor data.

    Args:
        config_path_or_dict: Path to a configuration file or a dictionary.
        data_path: Path to the sensor data file (e.g., CSV).

    Returns:
        A dictionary containing identified parameters and their history.
    """
    config = ConfigParser.load(config_path_or_dict)
    system = System.from_config(config)
    
    # Load model
    model_cfg = config["model"]
    model = build_model(model_cfg["name"], model_cfg["config"])
    
    # Digital Twin setup
    from rippl.core.inverse import DigitalTwin, InverseParameter
    params_data = config["physics"].get("inverse_parameters", [])
    parameters = []
    for p_data in params_data:
        parameters.append(InverseParameter(**p_data))
        
    # Sensor data columns
    data_cfg = config["data"]
    dt = DigitalTwin.from_csv(
        system, model, parameters, 
        csv_path=data_path,
        coord_cols=data_cfg["coord_cols"],
        field_cols=data_cfg["field_cols"],
        data_weight=data_cfg.get("data_weight", 1.0),
        physics_weight=data_cfg.get("physics_weight", 1.0)
    )
    
    train_cfg = config.get("training", {})
    results = dt.train(
        epochs=train_cfg.get("epochs", 5000),
        lr=train_cfg.get("lr", 1e-3)
    )
    
    return results
