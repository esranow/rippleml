from setuptools import setup, find_packages

setup(
    name="TensorWAV",
    version="0.0.1",
    description="A modular, research-grade PyTorch sub-framework",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pyyaml",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "tensorwav-cli=TensorWAV.cli:main",
        ],
    },
)
