import pytest

def test_ripple_import():
    """Verify that the ripple package is importable."""
    import ripple
    assert ripple.__version__ == "0.0.1"

def test_physics_blocks_import():
    """Verify that key physics blocks can be imported from the namespace."""
    from ripple.physics_blocks import HybridLaplacianBlock, HybridWaveResidualBlock
    assert HybridLaplacianBlock is not None
    assert HybridWaveResidualBlock is not None

def test_block_instantiation():
    """Verify basic instantiation of a core block."""
    from ripple.physics_blocks import HybridLaplacianBlock
    block = HybridLaplacianBlock(mode="point", correction_hidden=16)
    assert block.mode == "point"
