class RippleError(Exception):
    """Base exception for RippleML."""
    pass

class RippleValidationError(RippleError):
    """Raised when system validation fails."""
    pass
