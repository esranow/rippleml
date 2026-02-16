# Physics Blocks Package — File Manifest

## 📦 CREATED FILES

### Core Package Structure

#### `TensorWAV/physics_blocks/__init__.py`
- Package initializer exporting all blocks
- Imports updated to `TensorWAV.physics_blocks...`

#### `TensorWAV/physics_blocks/README.md`
- Documentation updated with `TensorWAV` package name

... (Block files moved to `TensorWAV/physics_blocks/`)

### Infrastructure

#### `setup.py` (NEW)
- Replaces `pyproject.toml` for explicit package installation
- Defines package name `TensorWAV`

#### `TensorWAV/` (NEW DIRECTORY)
- Contains all source code (`physics_blocks`, `training`, `models`, etc.)

### Tests & Scripts

#### `tests/test_physics_blocks.py`
- Updated imports to use `TensorWAV`

#### `manual_verify.py` (NEW)
- Script to verify imports by modifying `sys.path` directly (useful if pip install fails)

#### `fix_imports_recursive.py` (Helper)
- Used to update imports across all files

---

## 🧪 RUN TESTS

If `pip install -e .` fails, run tests by setting PYTHONPATH:

```bash
set PYTHONPATH=.;%PYTHONPATH%
pytest tests/test_physics_blocks.py -v
```

## 🚀 CLI

```bash
python -m TensorWAV.cli --config TensorWAV/configs/demo_blocks_playground.yaml
```
