import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class OperatorConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class EquationConfig(BaseModel):
    operators: List[OperatorConfig] = []
    params: Dict[str, Any] = {}

class DomainConfig(BaseModel):
    spatial_dims: int = 1
    time_dependent: bool = False
    bounds: List[List[float]] = []
    resolution: List[int] = []

class NetworkConfig(BaseModel):
    params: Dict[str, Any] = {}

def generate_json_schemas(out_dir="schemas/"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    models = {
        "OperatorConfig": OperatorConfig,
        "EquationConfig": EquationConfig,
        "DomainConfig": DomainConfig,
        "NetworkConfig": NetworkConfig
    }
    for name, model in models.items():
        schema = model.model_json_schema()
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(schema, f, indent=2)
