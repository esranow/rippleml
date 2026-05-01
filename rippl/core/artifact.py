import os
import zipfile
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file

class ArtifactCompiler:
    def __init__(self, engine):
        self.engine = engine

    def save(self, path: str, format="safetensors"):
        path = Path(path)
        if path.suffix != ".rpx":
            path = path.with_suffix(".rpx")
            
        temp_dir = Path(".rpx_temp")
        temp_dir.mkdir(exist_ok=True)
        
        if format != "safetensors":
            raise ValueError("Strictly use safetensors, no pickle.")
            
        # Save weights
        save_file(self.engine.net.state_dict(), temp_dir / "weights.safetensors")
        
        # Save config JSONs
        with open(temp_dir / "domain.json", "w") as f:
            json.dump({"spatial_dims": 1, "time_dependent": False}, f)
        with open(temp_dir / "equation.json", "w") as f:
            json.dump({"operators": []}, f)
        with open(temp_dir / "scales.json", "w") as f:
            json.dump({"L": 1.0, "T": 1.0, "U": 1.0}, f)
            
        # Copy diagnostics from engine.validate()
        diag_src = Path(".rpx_diagnostics")
        diag_dst = temp_dir / "diagnostics"
        if diag_src.exists():
            shutil.copytree(diag_src, diag_dst, dirs_exist_ok=True)
        else:
            diag_dst.mkdir()
            
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)
                    
        shutil.rmtree(temp_dir)
