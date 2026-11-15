"""Cross-task background and source ledger.

The ledger is a derived, recomputable view over task source ``pairing``
declarations, batch manifests, the shared background pool and the Blender
registry. It never writes plans, quotas or media, and it does not replace the
task-owned source configuration or the batch records.
"""

from .ledger import background_candidates, background_eligibility, build_ledger
from .report import DEFAULT_OUTPUT, format_summary, issue_counts, write_ledger

__all__ = [
    "DEFAULT_OUTPUT",
    "background_candidates",
    "background_eligibility",
    "build_ledger",
    "format_summary",
    "issue_counts",
    "write_ledger",
]
