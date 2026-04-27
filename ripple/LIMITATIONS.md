# Known Limitations

## CRITICAL

### 1. No Shock Capturing (Discontinuity Handling)
Severity: Critical for hyperbolic PDEs
Affected: Inviscid Burgers, Euler equations, any PDE with shocks
Symptom: Gibbs oscillations, NaN gradients, non-physical smearing
Missing: TVD schemes, WENO stencils, flux limiters, artificial viscosity
Workaround: Use viscous regularization (add small ν term). 
            Inviscid problems are unsupported.
Timeline: Pre-Phase 4

### 2. No Non-Dimensionalization
Severity: Critical for high Reynolds number or stiff systems
Affected: NS at Re > 1e4, chemical kinetics, multi-scale physics
Symptom: Gradient starvation, ill-conditioned Hessian, trivial solutions
Missing: Automatic characteristic scale extraction and PDE normalization
Workaround: Manually non-dimensionalize before defining System.
Timeline: Phase 4

### 3. Static Boundary Conditions Only
Severity: High for moving boundary problems
Affected: Melting fronts, vibrating membranes, Stefan problems
Symptom: BoxDistance enforces t=0 geometry for all t
Missing: Time-dependent distance function, NeuralDistanceField
Workaround: None currently. Fixed-geometry problems only.
Timeline: Phase 5

### 4. No Uncertainty Quantification
Severity: High for noisy inverse problems
Affected: All inverse problems with sensor noise
Symptom: Point estimates only, no confidence intervals
Missing: MC Dropout, Bayesian PINNs, Deep Ensembles
Workaround: Run multiple seeds, report variance manually.
Timeline: Phase 5

## ARCHITECTURAL

### 5. FD Solver First-Order Only
Affected: fd_solver.py stencils
Missing: Higher-order schemes, entropy condition enforcement

### 6. Derivative Cache Assumes Static Topology
Affected: equation.py compute_residual caching
Risk: Silent failure if mesh changes during training

### 7. ONNX Export Limited
Affected: io/export.py
Reason: torch.autograd.grad calls do not export cleanly
Workaround: TorchScript export instead