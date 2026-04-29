class RipplError(Exception):
    """Base exception for Rippl."""
    pass

class RipplValidationError(RipplError):
    """Raised when system validation fails."""
    pass

class PhysicsModelWarning(UserWarning):
    """Warning for physics modeling issues."""
    pass
