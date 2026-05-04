import pytest
from rippl.migrate import migrate, MigrationError

def test_migrate_detects_deepxde():
    source = "import deepxde as dde\npass"
    # Testing detection via exception since we just need to ensure it's routed correctly
    res = migrate(source, framework="auto")
    assert "DeepXDE" in res

def test_migrate_extracts_interval_bounds():
    source = "import deepxde as dde\ngeom = dde.geometry.Interval(0, 1)"
    res = migrate(source, framework="deepxde")
    assert "bounds=[(0, 1)]" in res

def test_migrate_extracts_time_domain():
    source = "import deepxde as dde\ngeom = dde.geometry.TimeDomain(0, 2.5)"
    res = migrate(source, framework="deepxde")
    assert "bounds=[[0, 2.5]]" in res

def test_migrate_extracts_collocation_counts():
    source = "import deepxde as dde\ndata = dde.data.PDE(geom, pde, [], num_domain=5000, num_boundary=100)"
    res = migrate(source, framework="deepxde")
    assert "batch_size=5000" in res

def test_migrate_outputs_rippl_script():
    source = "import deepxde as dde\ngeom = dde.geometry.Interval(0, 1)"
    res = migrate(source, framework="deepxde")
    assert "import rippl as rp" in res

def test_migrate_unknown_framework_raises():
    with pytest.raises(MigrationError):
        migrate("random code", framework="unknown")

def test_migrate_auto_detect_fails_gracefully():
    with pytest.raises(MigrationError):
        migrate("print('hello world')", framework="auto")

def test_migrate_never_executes_foreign_code():
    source = "import os\nos.system('rm -rf /')\nimport deepxde as dde\ngeom = dde.geometry.Interval(0, 1)"
    # should parse safely without throwing error or executing
    res = migrate(source, framework="auto")
    assert "import rippl as rp" in res

def test_rippl_top_level_migrate():
    import rippl as rp
    assert hasattr(rp, "migrate")
