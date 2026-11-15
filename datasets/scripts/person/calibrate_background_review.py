"""Prepare and fit a bounded background-review calibration for the person task.

``prepare`` joins tracked calibration cases to measured review metrics, writes
review images and a readable table. ``fit`` accepts only a complete, named
human label set, evaluates the current gate and writes frozen placement
thresholds. A suggested class is never treated as a human label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from datasets.scripts.common.review.judge import patch_is_valid
from datasets.scripts.common.review.placement import DEFAULT_THRESHOLDS as DEFAULT_PLACEMENT_THRESHOLDS, placement_decision
from datasets.scripts.person.human_review_state import backup_files, has_human_content, merge_cases

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "datasets/scripts/person/config/background_review_calibration_20261115.json"
DEFAULT_OUTPUT = "datasets/annotated/common/review/person/v4_background_calibration_20261115"
DEFAULT_THRESHOLDS = "datasets/scripts/person/config/background_review_thresholds_20261115.json"
LABELS = ("clean", "clutter", "geometry_failed")
METRIC_KEYS = (
    "patch_area_ratio",
    "object_ratio",
    "object_max_component_ratio",
    "vertical_structure_ratio",
    "largest_free_rect_ratio",
    "valid_footpoint_ratio",
    "best_body_obstacle_ratio",
    "marking_ratio",
    "confidence",
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(value: str | Path) -> str:
    return _path(value).resolve().relative_to(ROOT.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{_relative(path)} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {_relative(path)}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode image: {_relative(path)}")
    encoded.tofile(str(path))


def _metric_records(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for value in config.get("metrics_sources") or []:
        path = _path(str(value))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if "frame" not in payload:
                continue
            frame_path = str(payload["frame"].get("path") or "")
            batch = path.parent.name
            records[f"{batch}:{frame_path}"] = payload
    return records


def _case_metric(case: Mapping[str, Any], records: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    key = f"{case['metrics_batch']}:{case['path']}"
    metric = records.get(key)
    if metric is None:
        raise ValueError(f"{case['id']}: missing metric record {key}")
    return metric


def _ground_metrics(metric: Mapping[str, Any]) -> tuple[bool, bool, dict[str, Any], dict[str, Any]]:
    ground = metric.get("ground") or {}
    clearance = ground.get("clearance") or {}
    patch = ground.get("placement_patch") or []
    shape_raw = (metric.get("extra") or {}).get("image_shape") or []
    shape = (int(shape_raw[0]), int(shape_raw[1])) if len(shape_raw) >= 2 else None
    patch_ok = False
    patch_detail = "missing"
    if ground.get("ok") and patch:
        patch_ok, patch_detail = patch_is_valid(patch, shape)
    return bool(ground.get("ok")), patch_ok, dict(clearance), {"patch_detail": patch_detail, "patch": patch}


def _caption(case: Mapping[str, Any], metric: Mapping[str, Any]) -> str:
    ground_ok, patch_ok, clearance, _ = _ground_metrics(metric)
    if not ground_ok or not patch_ok:
        metrics_text = f"geometry={ground_ok}/{patch_ok}"
    else:
        metrics_text = (
            f"obj={clearance.get('object_ratio')} comp={clearance.get('object_max_component_ratio')} "
            f"free={clearance.get('largest_free_rect_ratio')} foot={clearance.get('valid_footpoint_ratio')}"
        )
    return f"{case['id']} {case['proposed_label']}  {metrics_text}"


def _diagnostic_overlay(image: np.ndarray, case: Mapping[str, Any], metric: Mapping[str, Any]) -> np.ndarray:
    canvas = image.copy()
    ground_ok, patch_ok, clearance, ground_extra = _ground_metrics(metric)
    ground = metric.get("ground") or {}
    polygon = ground.get("polygon") or []
    patch = ground_extra.get("patch") or []
    if polygon:
        cv2.polylines(
            canvas,
            [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)],
            True,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
    if patch:
        cv2.polylines(
            canvas,
            [np.asarray(patch, dtype=np.int32).reshape(-1, 1, 2)],
            True,
            (0, 190, 90) if patch_ok else (20, 20, 220),
            2,
            cv2.LINE_AA,
        )
    caption = _caption(case, metric)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (12, 12, 12), -1)
    cv2.putText(
        canvas,
        caption,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not ground_ok or not patch_ok:
        cv2.rectangle(
            canvas,
            (canvas.shape[1] // 2 - 160, canvas.shape[0] // 2 - 28),
            (canvas.shape[1] // 2 + 160, canvas.shape[0] // 2 + 28),
            (0, 0, 150),
            -1,
        )
        cv2.putText(
            canvas,
            "GEOMETRY FAILED",
            (canvas.shape[1] // 2 - 118, canvas.shape[0] // 2 + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _contact_sheet(items: list[tuple[str, np.ndarray]], output: Path, title: str) -> None:
    width, height, caption_height = 640, 360, 31
    columns = 4
    rows = math.ceil(len(items) / columns)
    sheet = np.full((42 + rows * (height + caption_height), columns * width, 3), 241, np.uint8)
    cv2.putText(sheet, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    for index, (caption, image) in enumerate(items):
        row, column = divmod(index, columns)
        x = column * width
        y = 42 + row * (height + caption_height)
        sheet[y : y + height, x : x + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (x, y + height), (x + width, y + height + caption_height), (28, 28, 28), -1)
        cv2.putText(
            sheet,
            caption,
            (x + 7, y + height + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    _write_image(output, sheet)


def prepare(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    rebuild_review: bool = False,
) -> dict[str, Any]:
    config_file = _path(config_path)
    config = _read_json(config_file)
    metrics = _metric_records(config)
    output = _path(output_dir)
    rows: list[dict[str, Any]] = []
    original_items: list[tuple[str, np.ndarray]] = []
    diagnostic_items: list[tuple[str, np.ndarray]] = []
    cases: list[dict[str, Any]] = []
    overlay_dir = output / "overlays"
    for case in config.get("cases") or []:
        if not isinstance(case, Mapping):
            raise ValueError("cases must be objects")
        metric = _case_metric(case, metrics)
        image = _read_image(_path(case["path"]))
        ground_ok, patch_ok, clearance, ground_extra = _ground_metrics(metric)
        diagnostic = _diagnostic_overlay(image, case, metric)
        _write_image(overlay_dir / f"{case['id']}.png", diagnostic)
        caption = _caption(case, metric)
        original_items.append((caption, image))
        diagnostic_items.append((f"{caption}  patch={ground_extra['patch_detail']}", diagnostic))
        row = {
            "id": case["id"],
            "path": case["path"],
            "parent_video": case.get("parent_video"),
            "source_group": case.get("source_group"),
            "metrics_batch": case["metrics_batch"],
            "proposed_label": case["proposed_label"],
            "human_label": case.get("human_label"),
            "accepted_by": case.get("accepted_by"),
            "ground_ok": ground_ok,
            "patch_ok": patch_ok,
            "patch_detail": ground_extra["patch_detail"],
        }
        for key in METRIC_KEYS:
            row[key] = clearance.get(key)
        rows.append(row)
        cases.append(
            {
                **row,
                "overlay": _relative(overlay_dir / f"{case['id']}.png"),
                "review_question": "请确认该帧属于 clean、clutter 或 geometry_failed。",
            }
        )
    if len(cases) != 24:
        raise ValueError(f"calibration must contain exactly 24 cases, got {len(cases)}")
    if len({case["path"] for case in cases}) != 24:
        raise ValueError("calibration paths must be unique")
    output.mkdir(parents=True, exist_ok=True)
    _contact_sheet(original_items, output / "contact_sheet_originals.png", "background review calibration originals (person task)")
    _contact_sheet(
        diagnostic_items,
        output / "contact_sheet_geometry.png",
        "background review ground / placement patch (person task)",
    )
    columns = list(rows[0])
    with (output / "calibration_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    review = {
        "schema_version": 1,
        "task": config["task"],
        "batch": config["batch"],
        "config": _relative(config_file),
        "config_sha256": _sha256(config_file),
        "status": "pending_human_background_labels",
        "proposed_allocation": dict(Counter(case["proposed_label"] for case in cases)),
        "cases": cases,
    }
    review_path = output / "review_cases.json"
    backups: list[Path] = []
    preserved_ids: list[str] = []
    if review_path.exists():
        existing = _read_json(review_path)
        if has_human_content(existing):
            if rebuild_review:
                backups = backup_files([review_path])
            else:
                review, preserved_ids = merge_cases(existing, review)
                review["merge_policy"] = "preserved_nonempty_human_fields"
    if backups:
        review["explicit_rebuild"] = True
        review["backups"] = [_relative(path) for path in backups]
    _write_json(review_path, review)
    return {
        "status": "written",
        "output": _relative(output),
        "cases": len(cases),
        "proposed_allocation": review["proposed_allocation"],
        "contact_sheet_originals": _relative(output / "contact_sheet_originals.png"),
        "contact_sheet_geometry": _relative(output / "contact_sheet_geometry.png"),
        "metrics": _relative(output / "calibration_metrics.csv"),
        "preserved_human_cases": preserved_ids,
        "backups": [_relative(path) for path in backups],
    }


def _threshold_midpoint(clean_values: list[float], clutter_values: list[float], mode: str) -> float | None:
    if not clean_values or not clutter_values:
        return None
    if mode == "max":
        clean_edge, clutter_edge = max(clean_values), min(clutter_values)
        if clean_edge >= clutter_edge:
            return None
        return math.floor(((clean_edge + clutter_edge) / 2.0) * 1000.0) / 1000.0
    clean_edge, clutter_edge = min(clean_values), max(clutter_values)
    if clean_edge <= clutter_edge:
        return None
    return math.floor(((clean_edge + clutter_edge) / 2.0) * 1000.0) / 1000.0


def _fit_thresholds(
    config: Mapping[str, Any],
    records: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cases = list(config.get("cases") or [])
    if not cases:
        raise ValueError("calibration has no cases")
    missing = [case["id"] for case in cases if case.get("human_label") not in LABELS]
    if missing:
        raise ValueError(
            "pending_human_labels; cannot fit thresholds for: " + ", ".join(missing)
        )
    reviewers = {case.get("accepted_by") for case in cases}
    if None in reviewers or any(not isinstance(value, str) or not value.strip() for value in reviewers):
        raise ValueError("every calibration case requires a named accepted_by")
    counts = Counter(str(case["human_label"]) for case in cases)
    if any(counts[label] != 8 for label in LABELS):
        raise ValueError(f"calibration must contain 8/8/8 human labels, got {dict(counts)}")

    clean_values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    clutter_values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    evaluated = []
    for case in cases:
        metric = _case_metric(case, records)
        ground_ok, patch_ok, clearance, ground_extra = _ground_metrics(metric)
        human_label = str(case["human_label"])
        if human_label in {"clean", "clutter"} and (not ground_ok or not patch_ok):
            raise ValueError(f"{case['id']}: {human_label} requires valid ground and placement patch")
        if human_label == "clean":
            for key in METRIC_KEYS:
                if clearance.get(key) is not None:
                    clean_values[key].append(float(clearance[key]))
        elif human_label == "clutter":
            for key in METRIC_KEYS:
                if clearance.get(key) is not None:
                    clutter_values[key].append(float(clearance[key]))
        evaluated.append(
            {
                "id": case["id"],
                "path": case["path"],
                "human_label": human_label,
                "proposed_label": case["proposed_label"],
                "ground_ok": ground_ok,
                "patch_ok": patch_ok,
                "patch_detail": ground_extra["patch_detail"],
                "clearance": clearance,
            }
        )

    thresholds = dict(DEFAULT_PLACEMENT_THRESHOLDS)
    choices = {
        "max_object_ratio": _threshold_midpoint(clean_values["object_ratio"], clutter_values["object_ratio"], "max"),
        "max_object_component_ratio": _threshold_midpoint(
            clean_values["object_max_component_ratio"], clutter_values["object_max_component_ratio"], "max"
        ),
        "max_vertical_ratio": _threshold_midpoint(
            clean_values["vertical_structure_ratio"], clutter_values["vertical_structure_ratio"], "max"
        ),
        "min_largest_free_rect_ratio": _threshold_midpoint(
            clean_values["largest_free_rect_ratio"], clutter_values["largest_free_rect_ratio"], "min"
        ),
        "min_valid_footpoint_ratio": _threshold_midpoint(
            clean_values["valid_footpoint_ratio"], clutter_values["valid_footpoint_ratio"], "min"
        ),
        "max_body_obstacle_ratio": _threshold_midpoint(
            clean_values["best_body_obstacle_ratio"], clutter_values["best_body_obstacle_ratio"], "max"
        ),
    }
    for key, value in choices.items():
        if value is not None:
            thresholds[key] = value
    clutter_min = min(clutter_values["object_ratio"])
    clean_max = max(clean_values["object_ratio"])
    if clean_max >= clutter_min:
        raise ValueError("human labels do not separate clean and clutter by object_ratio")
    thresholds["clutter_exclude_ratio"] = math.floor(clutter_min * 1000.0) / 1000.0
    thresholds["confidence_min"] = max(
        float(DEFAULT_PLACEMENT_THRESHOLDS["confidence_min"]),
        math.floor(min(clean_values["confidence"]) * 1000.0) / 1000.0,
    )
    thresholds["borderline_margin"] = 0.10

    predictions = []
    for item in evaluated:
        predicted = "geometry_failed"
        evidence: dict[str, Any] = {"patch_detail": item["patch_detail"]}
        if item["ground_ok"] and item["patch_ok"]:
            decision, evidence = placement_decision(item["clearance"], thresholds)
            predicted = {"usable": "clean", "cluttered": "clutter", "unsure": "unsure"}[decision]
        predictions.append(
            {
                "id": item["id"],
                "human_label": item["human_label"],
                "predicted_label": predicted,
                "match": predicted == item["human_label"],
                "evidence": evidence,
            }
        )
    mismatches = [item for item in predictions if not item["match"]]
    result = {
        "schema_version": 1,
        "task": config["task"],
        "calibration_batch": config["batch"],
        "status": "accepted" if not mismatches else "needs_human_decision",
        "accepted_by": sorted(reviewers),
        "counts": dict(sorted(counts.items())),
        "thresholds": thresholds,
        "separation": {
            "clean_object_ratio_max": clean_max,
            "clutter_object_ratio_min": clutter_min,
            "other_metric_separation": {
                key: value is not None for key, value in choices.items()
            },
        },
        "predictions": predictions,
        "mismatches": mismatches,
        "boundary": (
            "Thresholds are fitted to 24 human-labelled background frames. They are not model "
            "accuracy evidence and are not valid for other tasks without separate calibration."
        ),
    }
    return result, evaluated, {"predictions": predictions}


def fit(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config_file = _path(config_path)
    config = _read_json(config_file)
    records = _metric_records(config)
    result, evaluated, details = _fit_thresholds(config, records)
    result["config"] = _relative(config_file)
    result["config_sha256"] = _sha256(config_file)
    _write_json(_path(output_path), result)
    if result["status"] != "accepted":
        for mismatch in result["mismatches"]:
            print(f"mismatch: {mismatch['id']} human={mismatch['human_label']} predicted={mismatch['predicted_label']}")
    return {
        "status": result["status"],
        "output": _relative(output_path),
        "counts": result["counts"],
        "thresholds": result["thresholds"],
        "matches": sum(item["match"] for item in details["predictions"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-output", default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--rebuild-review",
        action="store_true",
        help="explicitly replace an existing human-reviewed package after backing it up",
    )
    parser.add_argument("action", choices=("prepare", "fit"))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.config, args.output, rebuild_review=args.rebuild_review)
    else:
        result = fit(args.config, args.threshold_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"written", "accepted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
