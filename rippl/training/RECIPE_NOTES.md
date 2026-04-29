# PINN Training Recipe — Validated Strategy

## Three-Phase Pipeline
Phase A (constraint curriculum): Train on BCs/ICs only before introducing PDE.
Forces model away from trivial u=0 solution before residual loss activates.

Phase B (lazy NTK + Adam): Full loss with NTK weight updates every 500 epochs.
Key insight: updating weights every epoch morphs the loss landscape under Adam's
feet. Freezing for 500 epochs gives Adam a stationary target.

Phase C (dynamic LBFGS handoff): Switch to LBFGS when Adam flatlines.
Key insight: preserve Adam's optimal state — hand LBFGS a pristine starting
point. LBFGS uses Hessian curvature to step to true minimum.

## Validated Results (T4 GPU)
- Heat 1D: L2=4.17e-04, 131s
- Wave 1D: L2=1.36e-03, 403s  (previously unsolvable without this recipe)
- Stokes 1D: u=7.64e-03, p=8.13e-02, 99s

## Key Parameters
- ntk_freq=500: lazy update, do NOT set to 1 (destroys landscape stability)
- phase_a_epochs=3000: enough to satisfy ICs before residual activates
- grad_clip=1.0: prevents NaN in high-frequency PDEs
- causal_bins=20: finer bins than default for better temporal resolution
