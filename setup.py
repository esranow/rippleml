from setuptools import setup, find_packages

setup(
    name="ripple",
    version="0.0.1",
    description="ripple: modular physics-ML framework for wave PDEs",
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "pyyaml"],
    # Map the package name 'ripple' to the TensorWAV directory (current dir '.')
    package_dir={"ripple": "."},
    packages=["ripple"] + [
        f"ripple.{pkg}" for pkg in [
            "physics", "solvers", "training", "models",
            "physics_blocks", "operators", "io", "datasets", "diagnostics",
            "core",
        ]
    ],
    entry_points={"console_scripts": ["ripple=ripple.cli:main"]},
)
