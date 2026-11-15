"""Bind accepted background-review frames to the current person scene plan.

The review verdict, parent relation, and complete-negative scope have separate
evidence fields. This compiler only distributes ``ready`` frames accepted by
the user; it does not turn ``unknown`` into an empty scene.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "datasets/scripts/person/config"
SELECTION = CONFIG / "review_selection.json"
PLAN = CONFIG / "production_plan.json"
SOURCE = CONFIG / "source.yaml"
DISPATCH = CONFIG / "background_dispatch.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative_file(value: str) -> Path:
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"source missing or outside workspace: {value}")
    return path


def compile_dispatch(*, dry_run: bool = False) -> dict:
    selection = _read(SELECTION)
    if selection.get("task") != "person":
        raise ValueError("person background review selection must declare task person")
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    plan = _read(PLAN)
    if plan.get("model_key") != "person" or plan.get("target_split") != "train":
        raise ValueError("expected a person train plan")
    if not plan.get("strata"):
        raise ValueError("person train plan has no states")
    ready = [item for item in selection["records"] if item.get("verdict") == "ready"]
    if not ready:
        raise ValueError("B6 has no ready background")
    if any(not item.get("accepted_by") for item in ready):
        raise ValueError("B6 ready backgrounds are not user accepted")
    rows = []
    for item in ready:
        frame = item["frame"]
        image_path = _relative_file(frame["path"])
        parent = frame.get("parent_video")
        if not parent:
            raise ValueError(f"missing parent relation: {frame['path']}")
        if not item.get("evidence", {}).get("parent_relation_verified_by"):
            raise ValueError(f"parent relation is not user verified: {frame['path']}")
        _relative_file(parent)
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] not in {(360, 640), (1080, 1920)}:
            raise ValueError(f"unsupported background dimensions: {frame['path']}")
        if item.get("existing_targets"):
            raise ValueError(f"ready background has existing person targets: {frame['path']}")
        scale = 3 if image.shape[1] == 1920 else 1
        polygon = [[min(639, round(x / scale)), min(359, round(y / scale))]
                   for x, y in item["ground_polygon"]]
        foot = [min(639, round(item["foot_xy"][0] / scale)),
                min(359, round(item["foot_xy"][1] / scale))]
        rows.append({
            "image": frame["path"], "path": frame["path"],
            "source_video": parent, "source_group": frame.get("source_group"),
            "state": "confirmed_empty", "boxes": [], "split": "train",
            "human_accepted": True, "accepted_by": "user_20261115",
            "review_batch": selection["batch"], "review_verdict": item["verdict"],
            "review_reason_codes": item.get("reason_codes") or [],
            "parent_link_basis": "user_confirmed_20261115",
            "ground_polygon_640x360": polygon, "foot_xy_640x360": foot,
            "source_image_size": [image.shape[1], image.shape[0]],
        })
    dispatch = {
        "schema_version": 1, "task": "person", "batch": selection["batch"],
        "review_selection": "datasets/scripts/person/config/review_selection.json",
        "source_relation_basis": "user_confirmed_20261115",
        "records": rows,
    }
    updated = copy.deepcopy(plan)
    updated["purpose"] = "Current bounded person batch with B6 reviewed background dispatch; foreground and label scope remain as declared per state."
    manifest = "datasets/scripts/person/config/background_dispatch.json"
    previous_dispatch = _read(DISPATCH)["records"] if DISPATCH.exists() else []
    for index, state in enumerate(updated["strata"]):
        row_index = 3 if index in (6, 9) else (index + index // 18) % len(rows)
        row = rows[row_index]
        old_foot = state["target"]["actors"][0]["foot_xy"]
        shift = [row["foot_xy_640x360"][axis] - old_foot[axis] for axis in (0, 1)]
        for actor in state["target"]["actors"]:
            actor["foot_xy"] = [actor["foot_xy"][axis] + shift[axis] for axis in (0, 1)]
        state["background"] = {"role": "reviewed_background", "manifest": manifest,
                               "record_index": row_index, "image": row["image"]}
        state["parameters"]["ground_polygon"] = row["ground_polygon_640x360"]
        # Existing source lists include the prior background and its video.
        old_background = plan["strata"][index]["background"]["image"]
        old_ref = plan["strata"][index]["background"]
        old_video = (
            previous_dispatch[old_ref["record_index"]]["source_video"]
            if old_ref.get("role") == "reviewed_background" and previous_dispatch
            else "datasets/raw/site_capture_20261112/videos/full_scene.mp4"
        )
        retained = set(state["sources"]) - {old_background, old_video}
        state["sources"] = sorted(retained | {row["image"], row["source_video"]})
    # At least eight of the 36 produced images must use the second parent group.
    # This keeps the first parent at or below the declared 80% cap regardless
    # of which optional strata the allocator chooses.
    site_states = [state for state in updated["strata"]
                   if rows[state["background"]["record_index"]]["source_group"] == "site_capture_full_scene"]
    for state in site_states[:8]:
        state["min_count"] = 1
    if len(site_states) < 8 or sum(state["min_count"] for state in updated["strata"]) > updated["budget"]:
        raise ValueError("background parent-group floor exceeds the plan budget")
    if not dry_run:
        _write(DISPATCH, dispatch)
        _write(PLAN, updated)
        source_text = SOURCE.read_text(encoding="utf-8")
        if "  reviewed_background:" not in source_text:
            anchor = "  cross_task_review:"
            if source_text.count(anchor) != 1:
                raise ValueError("source role insertion point is ambiguous")
            source_text = source_text.replace(anchor, f"  reviewed_background: {manifest}\n{anchor}")
            SOURCE.write_text(source_text, encoding="utf-8")
    return {"ready_backgrounds": len(rows), "dispatched_states": len(updated["strata"]),
            "parent_groups": sorted({row["source_group"] for row in rows}), "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compile_dispatch(dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
