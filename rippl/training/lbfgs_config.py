class LBFGSConfig:
    """
    Validated LBFGS configurations for different PDE types.
    """
    TIGHT = dict(lr=1.0, max_iter=50, history_size=100,
                 line_search_fn="strong_wolfe")   # max accuracy
    STANDARD = dict(lr=1.0, max_iter=20, history_size=50,
                    line_search_fn="strong_wolfe") # default
    FAST = dict(lr=0.5, max_iter=10, history_size=20,
                line_search_fn="strong_wolfe")     # quick refinement

    @staticmethod
    def for_pde(pde_type: str) -> dict:
        mapping = {
            "heat": LBFGSConfig.TIGHT,
            "wave": LBFGSConfig.STANDARD,
            "stokes": LBFGSConfig.TIGHT,
            "ns": LBFGSConfig.FAST,
        }
        return mapping.get(pde_type.lower(), LBFGSConfig.STANDARD)
