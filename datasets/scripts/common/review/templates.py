"""Load review templates declared in a task's review_plan.json.

A template is a crop taken from an already-labelled frame, used by the
emptiness scan to look for the same appearance elsewhere in a frame that has no
box there. Declared shape in review_plan.json:

    "templates": [
      {"label": "standee_community",
       "source": "datasets/annotated/.../frame_001201.png",
       "xyxy": [226, 104, 274, 224]}
    ]

White-background artwork is not a valid physical template: it does not share
pixels with a photographed standee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def _resolve(path: str | Path, base_dir: str | Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return Path(base_dir) / candidate


def load_templates(
    plan: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    min_side: int = 16,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Return (templates_bgr, loaded_info).

    `loaded_info` records what was actually available so the metric header can
    state the template set instead of assuming it.
    """

    templates: list[np.ndarray] = []
    info: list[dict[str, Any]] = []
    for entry in plan.get("templates") or []:
        label = str(entry.get("label") or "")
        source = entry.get("source")
        xyxy = entry.get("xyxy")
        record: dict[str, Any] = {"label": label, "source": source, "xyxy": xyxy}
        if not source or not xyxy or len(xyxy) != 4:
            record.update({"loaded": False, "reason": "incomplete_declaration"})
            info.append(record)
            continue
        path = _resolve(str(source), base_dir)
        if not path.exists():
            record.update({"loaded": False, "reason": "source_missing"})
            info.append(record)
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            record.update({"loaded": False, "reason": "unreadable"})
            info.append(record)
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        height, width = image.shape[:2]
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        if x2 - x1 < min_side or y2 - y1 < min_side:
            record.update({"loaded": False, "reason": "crop_too_small"})
            info.append(record)
            continue
        crop = image[y1:y2, x1:x2].copy()
        templates.append(crop)
        record.update({"loaded": True, "size": [int(crop.shape[1]), int(crop.shape[0])]})
        info.append(record)
    return templates, info


def template_summary(info: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compact, JSON-safe summary of how the template set was resolved."""

    loaded = [item for item in info if item.get("loaded")]
    return {
        "declared": len(info),
        "loaded": len(loaded),
        "labels": [item.get("label") for item in loaded],
        "failures": [{"label": item.get("label"), "reason": item.get("reason")} for item in info if not item.get("loaded")],
    }
