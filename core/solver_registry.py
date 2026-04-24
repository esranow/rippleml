"""ripple.core.solver_registry — Maps Equation term structure to solver key."""
from ripple.physics.operators import TimeDerivative, Diffusion, Advection, Nonlinear, Source


def select_solver(equation) -> str:
    """Inspect equation.terms and return a solver key.

    Returns one of: 'wave', 'diffusion', 'advection', 'advdiff'
    Raises NotImplementedError for unrecognised configurations.
    """
    terms = equation.terms
    has_t2   = any(isinstance(op, TimeDerivative) and op.order == 2 for _, op in terms)
    has_t1   = any(isinstance(op, TimeDerivative) and op.order == 1 for _, op in terms)
    has_diff = any(isinstance(op, Diffusion)      for _, op in terms)
    has_adv  = any(isinstance(op, Advection)      for _, op in terms)
    has_non  = any(isinstance(op, (Nonlinear, Source)) for _, op in terms)

    if has_t2 and has_t1:
        return "damped_wave"
    if has_t2:
        return "wave"
    if has_t1 and has_diff and has_adv:
        return "advdiff"
    if has_t1 and has_diff and has_non:
        return "reaction_diffusion"
    if has_t1 and has_non:
        return "first_order_nonlinear"
    if has_t1 and has_diff:
        return "diffusion"
    if has_t1 and has_adv:
        return "advection"

    ops = [type(op).__name__ for _, op in terms]
    raise NotImplementedError(f"No solver for operators: {ops}")
