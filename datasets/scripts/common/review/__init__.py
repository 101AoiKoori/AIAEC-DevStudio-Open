"""Cross-task candidate review: sampling, geometry, emptiness evidence, judging.

Task-specific thresholds live in datasets/scripts/<task>/config/review_plan.json;
this package stays task agnostic.
"""

from . import schema

__all__ = ["schema"]
