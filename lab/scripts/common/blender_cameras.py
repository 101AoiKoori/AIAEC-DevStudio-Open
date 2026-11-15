"""Validate reusable Blender master-scene camera and coverage plans.

The validator is offline: it checks the authoritative Blender asset registry,
scene file presence, deterministic perspective projection and declared coverage.
It does not open or modify Blender. A live scene inspection remains a separate
evidence step.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from lab.training_common.ground_placement import fit_ground_homography, project

from .contracts import SCHEMA_VERSION, now_local, resolve_workspace, workspace_relative


def _registry_lookup(
    master_scene: str,
    *,
    root: Path,
) -> dict[str, Any]:
    """Call the authoritative registry CLI rather than create a second store."""

    tool = root / "lab/scripts/blender/blender_source_registry.py"
    if not tool.is_file():
        return {"status": "unavailable", "reason": "registry_tool_missing"}
    completed = subprocess.run(
        [sys.executable, str(tool), "show", f"blender:{master_scene}"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        return {
            "status": "missing",
            "reason": (completed.stderr or completed.stdout).strip(),
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "reason": f"invalid registry JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"status": "invalid", "reason": "registry response is not an object"}
    return {"status": "found", "asset": payload}


def _projection_camera(camera: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "camera_location_m": camera.get("location_m"),
        "camera_target_m": camera.get("target_m"),
        "camera_lens_mm": camera.get("lens_mm"),
        "floor_top_m": camera.get("floor_top_m", 0.0),
    }


def _camera_metrics(
    camera: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    width, height = (int(value) for value in camera["resolution"])
    anchor = target.get("anchor_m") or [0.0, 0.0]
    if not isinstance(anchor, Sequence) or len(anchor) != 2:
        raise ValueError("target.anchor_m must contain x/y")
    floor = float(camera.get("floor_top_m", target.get("floor_top_m", 0.0)))
    object_height = float(target.get("height_m", 0.0))
    if object_height <= 0:
        raise ValueError("target.height_m must be positive")
    projection = _projection_camera(camera)
    bottom = project(
        projection,
        (float(anchor[0]), float(anchor[1]), floor + float(target.get("bottom_offset_m", 0.0))),
        image_size=(width, height),
        sensor_width_mm=float(target.get("sensor_width_mm", 36.0)),
    )
    top = project(
        projection,
        (float(anchor[0]), float(anchor[1]), floor + float(target.get("bottom_offset_m", 0.0)) + object_height),
        image_size=(width, height),
        sensor_width_mm=float(target.get("sensor_width_mm", 36.0)),
    )
    inside = (
        0.0 <= float(bottom[0]) < width
        and 0.0 <= float(bottom[1]) < height
        and 0.0 <= float(top[0]) < width
        and 0.0 <= float(top[1]) < height
    )
    pixel_height = abs(float(bottom[1]) - float(top[1]))
    location = np.asarray(camera["location_m"], dtype=float)
    aim = np.asarray(camera["target_m"], dtype=float)
    target_xy = np.asarray([float(anchor[0]), float(anchor[1]), floor + object_height / 2.0])
    offset = location[:2] - np.asarray([float(anchor[0]), float(anchor[1])])
    forward_axis = np.asarray(target.get("front_axis_xy") or [0.0, -1.0], dtype=float)
    if not np.isfinite(forward_axis).all() or np.linalg.norm(forward_axis) == 0:
        raise ValueError("target.front_axis_xy must be finite and nonzero")
    forward_axis /= np.linalg.norm(forward_axis)
    right_axis = np.asarray([-forward_axis[1], forward_axis[0]])
    front_component = float(np.dot(offset, forward_axis))
    side_component = float(np.dot(offset, right_axis))
    relative_angle = math.degrees(math.atan2(abs(side_component), front_component)) if front_component >= 0 else 180.0 - math.degrees(
        math.atan2(abs(side_component), abs(front_component))
    )
    if front_component >= 0 and abs(relative_angle) <= 25:
        view_class = "front"
    elif front_component >= 0 and abs(relative_angle) <= 65:
        view_class = "oblique"
    elif front_component >= 0:
        view_class = "side"
    else:
        view_class = "back"
    distance = float(np.linalg.norm(location - target_xy))
    return {
        "projected_bottom_xy": [round(float(bottom[0]), 3), round(float(bottom[1]), 3)],
        "projected_top_xy": [round(float(top[0]), 3), round(float(top[1]), 3)],
        "target_height_fraction": round(pixel_height / height, 6),
        "target_height_pixels": round(pixel_height, 3),
        "inside_frame": inside,
        "relative_view_angle_deg": round(relative_angle, 3),
        "view_class": view_class,
        "distance_m": round(distance, 6),
        "aim_distance_m": round(float(np.linalg.norm(aim - location)), 6),
    }


def _registration(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if not reference:
        return {"status": "not_declared", "reason": "reference_registration_missing"}
    world = reference.get("world_ground_quad_m")
    image = reference.get("image_ground_quad_px")
    if not isinstance(world, list) or len(world) != 4 or not isinstance(image, list) or len(image) != 4:
        raise ValueError("reference registration needs four world and image ground corners")
    try:
        matrix = fit_ground_homography(world, image)
    except ValueError as exc:
        return {"status": "fail", "reason": str(exc)}
    points = np.asarray(world, dtype=float)
    homogeneous = np.column_stack((points, np.ones(4))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) <= 1e-10):
        return {"status": "fail", "reason": "non_finite_homography_projection"}
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    errors = np.linalg.norm(projected - np.asarray(image, dtype=float), axis=1)
    maximum = float(reference.get("max_reprojection_error_px", 12.0))
    status = "pass" if float(percentile(errors, 95)) <= maximum else "fail"
    return {
        "status": status,
        "homography": matrix.tolist(),
        "mean_error_px": round(float(errors.mean()), 4),
        "max_error_px": round(float(errors.max()), 4),
        "error_95pct_px": round(float(percentile(errors, 95)), 4),
        "threshold_px": maximum,
        "source_image": reference.get("source_image"),
    }


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value))


def validate_camera_plan(
    plan: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    registry_lookup: Callable[[str, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one reusable master-scene camera plan."""

    base = Path(root or Path(__file__).resolve().parents[3]).resolve()
    master = plan.get("master_scene")
    if not isinstance(master, str) or not master:
        raise ValueError("camera plan requires master_scene")
    scene_path = resolve_workspace(master, root=base, must_exist=True)
    lookup = (registry_lookup or _registry_lookup)(master, root=base)
    cameras = plan.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("camera plan requires a nonempty cameras list")
    target = plan.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("camera plan requires a target object")
    errors: list[str] = []
    warnings: list[str] = []
    if lookup.get("status") != "found":
        errors.append("master_scene_not_registered")
    elif lookup["asset"].get("state") in {"removed", "retired", "archived"}:
        errors.append("master_scene_not_active")
    resolutions = {
        tuple(int(value) for value in camera.get("resolution") or [])
        for camera in cameras
        if isinstance(camera, Mapping)
    }
    if len(resolutions) != 1:
        errors.append("camera_resolution_inconsistent")

    results: list[dict[str, Any]] = []
    view_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    scale_bins = plan.get("required_scale_bins") or []
    bin_counts = [0 for _ in scale_bins]
    camera_ids: set[str] = set()
    for index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping):
            raise ValueError(f"cameras[{index}] must be an object")
        camera_id = str(camera.get("id") or "")
        if not camera_id or camera_id in camera_ids:
            errors.append(f"bad_camera_id:{index}")
            continue
        camera_ids.add(camera_id)
        for key in ("location_m", "target_m", "resolution"):
            if not isinstance(camera.get(key), list):
                errors.append(f"{camera_id}:missing_{key}")
        if not isinstance(camera.get("lens_mm"), (int, float)):
            errors.append(f"{camera_id}:missing_lens_mm")
            continue
        metrics = _camera_metrics(camera, target)
        declared_view = str(camera.get("view") or "")
        if metrics["view_class"] != declared_view:
            errors.append(f"{camera_id}:view_class_mismatch:{metrics['view_class']}!={declared_view}")
        if not metrics["inside_frame"]:
            errors.append(f"{camera_id}:target_outside_frame")
        view_counts[declared_view] = view_counts.get(declared_view, 0) + 1
        for tag in camera.get("coverage_tags") or []:
            tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
        for bin_index, scale_range in enumerate(scale_bins):
            if not isinstance(scale_range, list) or len(scale_range) != 2:
                errors.append("bad_required_scale_bin")
                continue
            low, high = float(scale_range[0]), float(scale_range[1])
            if low <= metrics["target_height_fraction"] <= high:
                bin_counts[bin_index] += 1
        results.append({"id": camera_id, **metrics, "declared_view": declared_view})

    for view in plan.get("required_views") or []:
        if not view_counts.get(str(view)):
            errors.append(f"view_not_covered:{view}")
    for tag in plan.get("required_coverage_tags") or []:
        if not tag_counts.get(str(tag)):
            errors.append(f"coverage_tag_missing:{tag}")
    for index, count in enumerate(bin_counts):
        if not count:
            errors.append(f"scale_bin_not_covered:{index}")
    registration = _registration(plan.get("reference_registration"))
    if registration["status"] == "fail":
        errors.append("reference_registration_failed")
    elif registration["status"] == "not_declared":
        warnings.append("reference_registration_missing")
    if not plan.get("scene_truth"):
        warnings.append("scene_truth_missing")
    status = "fail" if errors else ("review" if warnings else "pass_candidate")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_local(),
        "kind": "blender_camera_plan_report",
        "task": plan.get("task"),
        "plan_id": plan.get("id"),
        "master_scene": workspace_relative(scene_path, root=base),
        "registry": lookup,
        "scene_truth": plan.get("scene_truth"),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "cameras": results,
        "coverage": {
            "views": dict(sorted(view_counts.items())),
            "coverage_tags": dict(sorted(tag_counts.items())),
            "scale_bins": [
                {"range": scale_bins[index], "cameras": count}
                for index, count in enumerate(bin_counts)
            ],
            "reference_registration": registration,
        },
        "boundary": (
            "Offline projection and coverage checks do not open Blender, prove object "
            "visibility, validate occlusion pixels, or replace field calibration."
        ),
    }
