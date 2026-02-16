
import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

print("VERIFYING TENSORWAV NAMESPACE...")

try:
    import TensorWAV
    print("✓ import TensorWAV: OK")
except ImportError as e:
    print(f"✗ import TensorWAV: FAILED ({e})")
    sys.exit(1)

try:
    from TensorWAV.physics_blocks import (
        HybridLaplacianBlock,
        HybridWaveResidualBlock,
        SpectralHybridBlock,
        EnergyAwareBlock,
        HybridOscillatorBlock,
        PDEParameterEmbeddingBlock,
        HybridGradientBlock,
        BoundaryConditionBlock,
        HamiltonianBlock,
        SpectralRegularizationBlock,
        MultiScaleFourierFeatureBlock,
        SpectralConvBlock,
        HybridTimeStepperBlock,
        AdaptiveSamplingBlock,
        ConservationConstraintBlock,
        OperatorWrapperBlock,
    )
    print("✓ import all physics_blocks: OK")
except ImportError as e:
    print(f"✗ import physics_blocks: FAILED ({e})")
    sys.exit(1)

try:
    block = HybridLaplacianBlock(mode="point", correction_hidden=16)
    print("✓ instantiate block: OK")
except Exception as e:
    print(f"✗ instantiate block: FAILED ({e})")
    sys.exit(1)

print("\nNAMESPACE VERIFICATION SUCCESSFUL")
