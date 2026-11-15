"""Public access to the workspace's existing Bayesian allocation engine.

Allocation handles declared feasible strata and bounds, not task semantics or
proof of real-world frequencies. The implementation remains in its current
shared location so existing guarded callers use the same code.
"""
def plan_production(declaration, root=None):
    """Use the production planner, including its Blender interpreter fallback."""
    from lab.training_common.production_guard import plan_production as _plan_production
    return _plan_production(declaration, root)


def __getattr__(name):
    if name in {'allocate', 'plan_generation'}:
        from lab.training_common import bayesian_allocation
        return getattr(bayesian_allocation, name)
    raise AttributeError(name)


__all__ = ["allocate", "plan_generation", "plan_production"]
