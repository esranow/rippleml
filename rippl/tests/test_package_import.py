import pytest
import rippl
import torch
import numpy as np
import random

def test_package_version_exists():
    """
    Test that the package version is defined and is a string.
    """
    assert hasattr(rippl, "__version__")
    assert isinstance(rippl.__version__, str)
    assert rippl.__version__ == "0.3.0"

def test_deterministic_seed():
    """
    Test setting seeds for deterministic behavior (compliance check).
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Quick check to ensure torch is behaving predictably
    t1 = torch.rand(2, 2)
    torch.manual_seed(seed)
    t2 = torch.rand(2, 2)
    assert torch.allclose(t1, t2), "Torch random generation is not deterministic with manual_seed"

def test_imports_allowed():
    """
    Verify basic imports required are available in environment.
    """
    import yaml
    import pathlib
    import matplotlib.pyplot
    assert True

