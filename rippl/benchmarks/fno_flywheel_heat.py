"""
rippl.benchmarks.fno_flywheel_heat — FNO Operator Learning for Heat Equation.
"""
import torch
import numpy as np
from rippl.core.system import System, Domain
from rippl.physics.operators import TimeDerivative, Laplacian
from rippl.core.equation import Equation
from rippl.nn.fno import FNO
from rippl.training.fno_flywheel import FNOFlywheel

def main():
    # 1. Physics Definition
    # We use a higher diffusivity for faster dissipation in the benchmark
    alpha = 0.05
    eq = Equation([(1.0, TimeDerivative(field="u")), (-alpha, Laplacian(field="u"))])
    # FNO works best on uniform grids, resolution 64 is standard
    domain = Domain(spatial_dims=1, bounds=((0, 1),), resolution=(64,))
    sys = System(equation=eq, domain=domain)
    
    # 2. FNO Model
    # input_dim=1 (spatial coordinates x), n_modes=16, width=64
    fno = FNO(n_modes=16, width=64, input_dim=1, output_dim=1)
    
    # 3. Flywheel Pipeline
    # Generate 1000 training pairs from FD solver, learn the operator
    flywheel = FNOFlywheel(sys, fno, n_train=1000, n_test=200)
    
    print("--- FNO Flywheel Pipeline Start ---")
    results = flywheel.pipeline(train_epochs=300)
    print("--- FNO Flywheel Pipeline Complete ---")
    
    mean_l2 = results["mean_l2"]
    print(f"Final mean L2 test error: {mean_l2:.4e}")
    
    if mean_l2 < 0.05:
        print("SUCCESS: FNO learned the operator with mean L2 < 0.05")
    else:
        print("FAILURE: Error too high")

if __name__ == "__main__":
    main()
