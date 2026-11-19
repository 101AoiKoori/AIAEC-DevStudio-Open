"""Inventory person annotations without promoting candidates to complete truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = "datasets/annotated/common/review/person/auto_annotation_inventory_20261115.json"
ANNOTATION_LAYERS = "datasets/scripts/person/config/direct_annotation_layers_20261115.json"

DIRECT_REVIEWS = (
    {
        "name": "v4_multivideo_review_20261112",
        "manifest": "datasets/annotated/person/v4_multivideo_review_20261112/review_manifest.json",
        "records_key": "samples",
    },
    {
        "name": "v4_full_scene_diverse_review_20261112",
        "manifest": "datasets/annotated/person/v4_full_scene_diverse_review_20261112/review_manifest.json",
        "records_key": "samples",
    },
    {
        "name": "v4_vehicle_sequence_review_20261112",
        "manifest": "datasets/annotated/person/v4_vehicle_sequence_review_20261112/review_manifest.json",
        "records_key": "images",
    },
    {
        "name": "site_person_vehicle_v4_review",
        "manifest": "datasets/annotated/site_capture_20261112/person_vehicle_v4_review/review_manifest.json",
        "records_key": "images",
        "nested_person": True,
    },
    {
        "name": "v4_four_front_whiteback_candidate_20261113",
        "manifest": "datasets/annotated/person/v4_four_front_whiteback_candidate_20261113/manifest.json",
        "single_record": True,
        "default_state": "candidate_complete_positive",
    },
    {
        "name": "v4_whiteback_crop_candidate_20261113",
        "manifest": "datasets/annotated/person/v4_whiteback_crop_candidate_20261113/manifest.json",
        "single_record": True,
        "default_state": "candidate_complete_negative",
    },
)

YOLO_BATCHES = (
    "datasets/annotated/person/mixed_detection_candidates_20261110",
    "datasets/annotated/person/teammate_real_scene_candidates_20261110",
    "datasets/annotated/person/v4_four_front_whiteback_candidate_20261113",
    "datasets/annotated/person/v4_full_scene_diverse_review_20261112",
    "datasets/annotated/person/v4_multivideo_review_20261112",
    "datasets/annotated/person/v4_vehicle_sequence_review_20261112",
    "datasets/annotated/person/v4_whiteback_crop_candidate_20261113",
    "datasets/annotated/site_capture_20261112/person_vehicle_v4_review",
)

MANIFEST_CANDIDATES = (
    "datasets/annotated/car_real_capture_20261112/target_instances_manifest.jsonl",
    "datasets/annotated/car_real_capture_20261112/sequence_instances_part1_manifest.jsonl",
    "datasets/annotated/car_real_capture_20261112/sequence_instances_part2_manifest.jsonl",
)

SITE_CANDIDATE_MANIFEST = "datasets/annotated/site_capture_20261112/target_instances_manifest.jsonl"
TEAMMATE_SYNTH_LABEL_DIR = "datasets/assets/person/dataset/synth/labels"
HUMAN_ACCEPTANCE = "datasets/scripts/person/config/toolchain_review_acceptance.json"


def _workspace(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_workspace(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in _workspace(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_state(value: Any) -> str:
    state = str(value or "unknown")
    return state.removeprefix("candidate_")


def _reference_path(value: Any, manifest_path: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    direct = _workspace(value).resolve()
    if direct.exists():
        return _relative(direct)
    local = (manifest_path.parent / value).resolve()
    if local.exists():
        return _relative(local)
    return value.replace("\\", "/")


def _record_image(record: Mapping[str, Any], manifest_path: Path) -> str:
    value = record.get("image") or record.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_relative(manifest_path)} contains a record without image")
    resolved = _reference_path(value, manifest_path)
    if resolved is None:
        raise ValueError(f"{_relative(manifest_path)} contains an empty image path")
    return resolved


def _record_canonical_image(record: Mapping[str, Any], manifest_path: Path) -> str:
    for key in ("parent_image", "source_parent"):
        value = record.get(key)
        resolved = _reference_path(value, manifest_path)
        if resolved:
            return resolved
    return _record_image(record, manifest_path)


def _person_fields(record: Mapping[str, Any], *, nested: bool) -> Mapping[str, Any]:
    if nested:
        tasks = record.get("tasks")
        if isinstance(tasks, Mapping):
            person = tasks.get("person")
            if isinstance(person, Mapping):
                return person
    return record


def _label_summary(value: Any, manifest_path: Path) -> dict[str, Any]:
    path = _reference_path(value, manifest_path)
    if not path:
        return {"path": None, "exists": False, "lines": 0}
    resolved = _workspace(path)
    if not resolved.exists():
        return {"path": path, "exists": False, "lines": 0}
    text = resolved.read_text(encoding="utf-8-sig")
    return {
        "path": path,
        "exists": True,
        "nonempty": bool(text.strip()),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
    }


def _label_mismatch(
    label: Mapping[str, Any],
    boxes: Sequence[Mapping[str, Any]],
    image_size: Any,
) -> str | None:
    if not label.get("exists"):
        return None
    if (
        not isinstance(image_size, Sequence)
        or isinstance(image_size, (str, bytes))
        or len(image_size) != 2
    ):
        return "image_size_missing"
    width, height = image_size
    lines = [
        line.split()
        for line in _workspace(str(label["path"])).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) != len(boxes):
        return "box_count"
    for index, (line, box) in enumerate(zip(lines, boxes)):
        class_id = box.get("class_id")
        raw_xyxy = box.get("bbox_xyxy") or box.get("xyxy")
        if (
            type(class_id) is not int
            or not isinstance(raw_xyxy, Sequence)
            or isinstance(raw_xyxy, (str, bytes))
            or len(raw_xyxy) != 4
        ):
            return f"box_{index}_input"
        x1, y1, x2, y2 = (float(value) for value in raw_xyxy)
        expected = (
            class_id,
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        try:
            actual = (int(line[0]), *(float(value) for value in line[1:]))
        except (IndexError, ValueError):
            return f"box_{index}_format"
        if actual[0] != expected[0] or any(
            abs(actual[pos] - expected[pos]) > 1e-5 for pos in range(1, 5)
        ):
            return f"box_{index}_values"
    return None


def _direct_review_summary(
    specification: Mapping[str, Any],
    *,
    human_accepted_images: set[str],
) -> dict[str, Any]:
    manifest_path = _workspace(str(specification["manifest"]))
    data = _read_json(manifest_path)
    if specification.get("single_record"):
        records: list[Mapping[str, Any]] = [data]
    else:
        values = data.get(str(specification["records_key"]))
        if not isinstance(values, list):
            raise ValueError(f"{_relative(manifest_path)} lacks {specification['records_key']}")
        records = [value for value in values if isinstance(value, Mapping)]

    states: Counter[str] = Counter()
    rows = []
    canonical_images = set()
    proposed_boxes = 0
    label_boxes = 0
    label_box_mismatches = []
    for record in records:
        person = _person_fields(record, nested=bool(specification.get("nested_person")))
        raw_state = person.get("state") or record.get("state") or specification.get("default_state")
        state = _normalize_state(raw_state)
        states[state] += 1
        boxes = person.get("boxes") or person.get("instances") or record.get("person_boxes") or []
        if not isinstance(boxes, list):
            boxes = []
        proposed_boxes += len(boxes)
        label = _label_summary(person.get("label", record.get("label")), manifest_path)
        label_boxes += label["lines"]
        processing = record.get("image_processing")
        processed_size = processing.get("output_size") if isinstance(processing, Mapping) else None
        mismatch = _label_mismatch(
            label,
            boxes,
            person.get("image_size") or record.get("image_size") or processed_size,
        )
        if mismatch:
            label_box_mismatches.append(
                {
                    "id": record.get("id"),
                    "proposed_boxes": len(boxes),
                    "label_boxes": label["lines"],
                    "reason": mismatch,
                }
            )
        image = _record_image(record, manifest_path)
        canonical = _record_canonical_image(record, manifest_path)
        canonical_images.add(canonical)
        manifest_accepted = bool(person.get("human_accepted", record.get("human_accepted")))
        semantic_confirmed = manifest_accepted or image in human_accepted_images
        rows.append(
            {
                "id": record.get("id"),
                "image": image,
                "canonical_image": canonical,
                "state": state,
                "proposed_boxes": len(boxes),
                "label": label,
                "manifest_human_accepted": manifest_accepted,
                "human_semantic_confirmed": semantic_confirmed,
            }
        )

    return {
        "name": specification["name"],
        "manifest": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "records": len(rows),
        "unique_canonical_images": len(canonical_images),
        "states": dict(sorted(states.items())),
        "proposed_boxes": proposed_boxes,
        "label_files": sum(bool(row["label"]["path"]) for row in rows),
        "existing_label_files": sum(bool(row["label"]["exists"]) for row in rows),
        "label_boxes": label_boxes,
        "label_box_mismatches": label_box_mismatches,
        "human_semantic_confirmed_records": sum(
            row["human_semantic_confirmed"] for row in rows
        ),
        "records_detail": rows,
    }


def _yolo_batch_summary(value: str) -> dict[str, Any]:
    root = _workspace(value)
    labels = sorted(root.rglob("*.txt"))
    image_root = root / "images"
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    task_person_labels = sorted((root / "labels" / "person").rglob("*.txt"))
    person_labels = task_person_labels or (
        labels if (root / "labels").is_dir() and not task_person_labels else []
    )

    def count(label_paths: Sequence[Path]) -> tuple[int, int]:
        nonempty = 0
        boxes = 0
        for label in label_paths:
            text = label.read_text(encoding="utf-8-sig")
            nonempty += int(bool(text.strip()))
            boxes += sum(1 for line in text.splitlines() if line.strip())
        return nonempty, boxes

    nonempty, boxes = count(labels)
    person_nonempty, person_boxes = count(person_labels)
    return {
        "root": _relative(root),
        "local_images": len(images),
        "label_files": len(labels),
        "nonempty_label_files": nonempty,
        "boxes": boxes,
        "person_label_files": len(person_labels),
        "person_nonempty_label_files": person_nonempty,
        "person_boxes": person_boxes,
    }


def _candidate_records(
    manifests: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, int], set[str]]:
    rows = []
    unique_boxes: set[tuple[str, tuple[float, ...]]] = set()
    unique_images = set()
    counts = Counter()
    for manifest in manifests:
        path = _workspace(manifest)
        for record in _read_jsonl(path):
            image = _reference_path(record.get("path") or record.get("image"), path) or "?"
            canonical = _reference_path(record.get("parent_image"), path) or image
            observations = record.get("task_observations")
            person_observation = (
                observations.get("person")
                if isinstance(observations, Mapping) and isinstance(observations.get("person"), Mapping)
                else None
            )
            task_review = record.get("task_review")
            person_review = (
                task_review.get("person")
                if isinstance(task_review, Mapping) and isinstance(task_review.get("person"), Mapping)
                else None
            )
            person_state = (
                person_observation.get("state")
                if person_observation
                else person_review.get("state")
                if person_review
                else None
            )
            instances = []
            if person_observation:
                instances.extend(
                    instance
                    for instance in person_observation.get("instances") or []
                    if isinstance(instance, Mapping)
                )
            instances.extend(
                instance
                for instance in record.get("observed_instances") or []
                if isinstance(instance, Mapping) and instance.get("task") == "person"
            )
            if not instances and not person_state:
                continue
            counts["records_with_person_observation"] += 1
            counts["person_states_" + str(person_state or "observed_instance_only")] += 1
            counts["exhaustive_true"] += int(
                bool(
                    (person_observation or {}).get("exhaustive")
                    or (person_review or {}).get("exhaustive")
                )
            )
            if instances:
                unique_images.add(canonical)
            for instance in instances:
                bbox = tuple(float(value) for value in instance.get("bbox_xyxy") or [])
                key = (canonical, bbox)
                if key in unique_boxes:
                    counts["duplicate_boxes"] += 1
                    continue
                unique_boxes.add(key)
                counts["unique_boxes"] += 1
                counts["known_class_boxes"] += int(instance.get("class_id") in (0, 1))
                counts["unknown_class_boxes"] += int(instance.get("class_id") not in (0, 1))
                counts[str(instance.get("status") or "unknown_instance_status")] += 1
            rows.append(
                {
                    "manifest": _relative(path),
                    "image": image,
                    "canonical_image": canonical,
                    "state": person_state,
                    "instances": len(instances),
                }
            )
    counts["unique_canonical_images"] = len(unique_images)
    return rows, dict(sorted(counts.items())), unique_images


def _review_metrics_summary() -> dict[str, Any]:
    root = _workspace("datasets/annotated/common/review/person")
    batches = []
    all_images = set()
    total_frames = 0
    verdicts: Counter[str] = Counter()
    accepted = 0
    for directory in sorted(root.glob("v4_review_20261115*")):
        metrics = directory / "metrics.jsonl"
        if not metrics.exists():
            continue
        rows = _read_jsonl(metrics)
        frames = [row for row in rows if isinstance(row.get("frame"), Mapping)]
        images = {
            str(frame["frame"].get("path"))
            for frame in frames
            if frame["frame"].get("path")
        }
        total_frames += len(frames)
        all_images.update(images)
        batch_verdicts: Counter[str] = Counter()
        batch_accepted = 0
        selection = directory / "selection.json"
        if selection.exists():
            for row in _read_json(selection).get("records") or []:
                if isinstance(row, Mapping):
                    batch_verdicts[str(row.get("verdict") or "unknown")] += 1
                    batch_accepted += int(bool(row.get("accepted_by")))
        verdicts.update(batch_verdicts)
        accepted += batch_accepted
        batches.append(
            {
                "batch": directory.name,
                "metrics_frames": len(frames),
                "unique_images": len(images),
                "selection_verdicts": dict(sorted(batch_verdicts.items())),
                "selection_records": sum(batch_verdicts.values()),
                "accepted_by_present": batch_accepted,
            }
        )
    return {
        "scope": "automatic parent-frame review measurements; not person boxes or YOLO labels",
        "metrics_frames": total_frames,
        "unique_images": len(all_images),
        "selection_verdicts": dict(sorted(verdicts.items())),
        "accepted_by_present": accepted,
        "batches": batches,
    }


def _synth_jsonl_summary(label_dir: str) -> dict[str, Any]:
    files = []
    total_rows = 0
    rows_with_bbox = 0
    for split in ("test", "train", "val"):
        path = _workspace(label_dir) / f"{split}.jsonl"
        rows = _read_jsonl(path)
        with_bbox = sum(
            bool(row.get("bbox") or row.get("bbox_xyxy") or row.get("boxes"))
            for row in rows
        )
        total_rows += len(rows)
        rows_with_bbox += with_bbox
        files.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
                "rows": len(rows),
                "rows_with_bbox": with_bbox,
            }
        )
    return {"files": files, "rows": total_rows, "rows_with_bbox": rows_with_bbox}


def _human_acceptance_summary() -> dict[str, Any]:
    path = _workspace(HUMAN_ACCEPTANCE)
    data = _read_json(path)
    by_source: Counter[str] = Counter()
    direct_positive_boxes = 0
    direct_negative_frames = 0
    direct_camera_images = []
    for item in data.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        source_kind = str(item.get("source_kind") or "unknown")
        by_source[source_kind] += 1
        matrix = item.get("status_matrix")
        if source_kind != "direct_camera_capture" or not isinstance(matrix, Mapping):
            continue
        image = item.get("image")
        if isinstance(image, str) and image:
            direct_camera_images.append(image)
        objects = matrix.get("objects") or []
        positive_boxes = sum(
            1
            for obj in objects
            if isinstance(obj, Mapping) and obj.get("annotation_action") == "label_visible_box"
        )
        direct_positive_boxes += positive_boxes
        if matrix.get("presence") == "none":
            direct_negative_frames += 1
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "status": data.get("status"),
        "items": len(data.get("items") or []),
        "by_source_kind": dict(sorted(by_source.items())),
        "direct_camera_positive_boxes": direct_positive_boxes,
        "direct_camera_confirmed_negative_frames": direct_negative_frames,
        "direct_camera_images": direct_camera_images,
        "boundary": data.get("boundary"),
    }


def _overlap(left: Path, right: Path) -> dict[str, Any]:
    left_files = {
        path.relative_to(left).as_posix(): path
        for path in (left / "images").rglob("*")
        if path.is_file()
    }
    right_files = {
        path.relative_to(right).as_posix(): path
        for path in (right / "images").rglob("*")
        if path.is_file()
    }
    common = sorted(set(left_files) & set(right_files))
    return {
        "left": _relative(left),
        "right": _relative(right),
        "left_images": len(left_files),
        "right_images": len(right_files),
        "same_relative_image_paths": len(common),
        "union_images": len(set(left_files) | set(right_files)),
        "sample_overlaps": common[:10],
    }


def build_inventory() -> dict[str, Any]:
    human_acceptance = _human_acceptance_summary()
    human_accepted_images = set(human_acceptance.pop("direct_camera_images", []))
    direct_batches = [
        _direct_review_summary(spec, human_accepted_images=human_accepted_images)
        for spec in DIRECT_REVIEWS
    ]
    direct_images = {
        row["canonical_image"]
        for batch in direct_batches
        for row in batch["records_detail"]
    }
    all_states: Counter[str] = Counter()
    for batch in direct_batches:
        all_states.update(batch["states"])
    direct_records = [
        record
        for batch in direct_batches
        for record in batch["records_detail"]
    ]
    auto_validated_records = [
        record for record in direct_records if record["state"] in {"complete_positive", "complete_negative"}
    ]
    pending_partial_records = [
        record for record in direct_records if record["state"] == "partial"
    ]
    human_semantic_records = [
        record for record in direct_records if record["human_semantic_confirmed"]
    ]

    car_rows, car_counts, car_images = _candidate_records(MANIFEST_CANDIDATES)
    site_rows, site_counts, site_images = _candidate_records((SITE_CANDIDATE_MANIFEST,))
    teammate_synth = _synth_jsonl_summary(TEAMMATE_SYNTH_LABEL_DIR)

    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "person",
        "scope": (
            "Existing person review manifests, candidate YOLO batches, CarReal/SiteCapture "
            "instance manifests, automatic frame-review metrics and the recorded human "
            "semantic acceptance. This is not a model-performance result."
        ),
        "summary": {
            "direct_review": {
                "images": sum(batch["records"] for batch in direct_batches),
                "unique_canonical_images": len(direct_images),
                "label_files": sum(batch["label_files"] for batch in direct_batches),
                "existing_label_files": sum(
                    batch["existing_label_files"] for batch in direct_batches
                ),
                "label_boxes": sum(batch["label_boxes"] for batch in direct_batches),
                "proposed_boxes": sum(batch["proposed_boxes"] for batch in direct_batches),
                "states": dict(sorted(all_states.items())),
                "human_semantic_confirmed_records": sum(
                    batch["human_semantic_confirmed_records"] for batch in direct_batches
                ),
                "status_layers": {
                    "source_available_records": len(direct_records),
                    "annotated_records": sum(record["label"]["exists"] for record in direct_records),
                    "candidate_records": len(direct_records),
                    "candidate_boxes": sum(record["proposed_boxes"] for record in direct_records),
                    "auto_validated_records": len(auto_validated_records),
                    "auto_validated_positive_boxes": sum(
                        record["label"]["lines"] for record in auto_validated_records
                    ),
                    "human_semantic_confirmed_records": len(human_semantic_records),
                    "human_semantic_confirmed_boxes": human_acceptance.get(
                        "direct_camera_positive_boxes", 0
                    ),
                    "human_box_or_absence_confirmed_records": 0,
                    "human_box_confirmed_boxes": 0,
                    "training_admitted_records": 0,
                    "pending_partial_records": len(pending_partial_records),
                    "pending_partial_candidate_boxes": sum(
                        record["proposed_boxes"] for record in pending_partial_records
                    ),
                    "pending_human_semantic_confirmation_records": len(direct_records)
                    - len(human_semantic_records),
                    "pending_human_label_confirmation_records": len(direct_records),
                    "pending_human_box_confirmation_boxes": sum(
                        record["label"]["lines"] for record in auto_validated_records
                    ),
                    "pending_training_admission_records": len(direct_records),
                },
            },
            "manifest_only_candidates": {
                "car_real": {
                    "records": len(car_rows),
                    **car_counts,
                },
                "site_capture": {
                    "records": len(site_rows),
                    **site_counts,
                },
                "car_site_union_canonical_images_not_summed": len(car_images | site_images),
            },
            "automatic_parent_frame_review": _review_metrics_summary(),
            "human_semantic_acceptance": human_acceptance,
            "teammate_synthetic_annotations": teammate_synth,
        },
        "direct_review_batches": direct_batches,
        "candidate_yolo_batches": [_yolo_batch_summary(value) for value in YOLO_BATCHES],
        "duplicate_candidate_batches": [
            _overlap(
                _workspace("datasets/annotated/person/mixed_detection_candidates_20261110"),
                _workspace("datasets/annotated/person/teammate_real_scene_candidates_20261110"),
            )
        ],
        "manifest_only_candidate_details": {
            "car_real": {
                "manifests": list(MANIFEST_CANDIDATES),
                "counts": car_counts,
                "records_detail": car_rows,
            },
            "site_capture": {
                "manifests": [SITE_CANDIDATE_MANIFEST],
                "counts": site_counts,
                "records_detail": site_rows,
            },
        },
        "boundary": (
            "source_available, annotated, candidate, auto_validated, human_confirmed and "
            "training_admitted are independent layers. A missing human decision is pending, "
            "not rejected or excluded. Partial/unknown frames keep their proposed boxes and "
            "do not authorize empty YOLO labels. class_id=null manifest boxes cannot become "
            "person YOLO rows until a class is established. exhaustive=false prevents treating "
            "the listed boxes as a complete image-level annotation."
        ),
    }


def _assert_consistency(inventory: Mapping[str, Any]) -> None:
    direct = inventory["summary"]["direct_review"]
    if direct["label_files"] != direct["existing_label_files"]:
        raise ValueError("direct review references a missing label file")
    mismatches = [
        mismatch
        for batch in inventory["direct_review_batches"]
        for mismatch in batch["label_box_mismatches"]
    ]
    if mismatches:
        raise ValueError(f"direct review label/box mismatches: {mismatches}")
    for batch in inventory["candidate_yolo_batches"]:
        if batch["nonempty_label_files"] > batch["label_files"]:
            raise ValueError(f"{batch['root']} has more nonempty labels than label files")


def _status_layer_document(inventory: Mapping[str, Any]) -> dict[str, Any]:
    auto_validated = []
    pending_partial = []
    for batch in inventory["direct_review_batches"]:
        for record in batch["records_detail"]:
            human_semantic = bool(record["human_semantic_confirmed"])
            row = {
                "id": record["id"],
                "task": "person",
                "source_batch": batch["name"],
                "source_manifest": batch["manifest"],
                "source_manifest_sha256": batch["manifest_sha256"],
                "image": record["image"],
                "canonical_image": record["canonical_image"],
                "state": record["state"],
                "label": record["label"]["path"],
                "label_sha256": (
                    _sha256(record["label"]["path"]) if record["label"]["exists"] else None
                ),
                "positive_boxes": record["proposed_boxes"]
                if record["state"] == "complete_positive"
                else 0,
                "candidate_boxes": record["proposed_boxes"],
                "status_layers": {
                    "source_available": True,
                    "annotated": bool(record["label"]["exists"]),
                    "candidate": True,
                    "auto_validated": record["state"]
                    in {"complete_positive", "complete_negative"},
                    "human_confirmed": {
                        "semantic": human_semantic,
                        "box_or_absence": False,
                        "mask": False,
                    },
                    "training_admitted": False,
                },
                "pending": [
                    *([] if human_semantic else ["human_semantic_confirmation"]),
                    "human_box_or_absence_confirmation",
                    "training_admission",
                ],
                "human_semantic_confirmed": human_semantic,
            }
            if record["state"] in {"complete_positive", "complete_negative"}:
                if not record["label"]["exists"]:
                    raise ValueError(f"{batch['name']}/{record['id']} complete record lacks a label")
                auto_validated.append(row)
            elif record["state"] == "partial":
                pending_partial.append(row)
            else:
                raise ValueError(
                    f"{batch['name']}/{record['id']} has unsupported state {record['state']}"
                )
    states = Counter(row["state"] for row in auto_validated)
    records = [*auto_validated, *pending_partial]
    return {
        "schema_version": 1,
        "generated_at": inventory["generated_at"],
        "kind": "person_direct_annotation_status_layers",
        "task": "person",
        "status": "auto_validated_with_pending",
        "source_inventory": inventory.get("source_inventory"),
        "source_inventory_sha256": inventory.get("source_inventory_sha256"),
        "acceptance_basis": (
            "Each auto-validated record has an explicit complete_positive or complete_negative "
            "state in its source review manifest, and every referenced YOLO label exists and "
            "matches the reviewed boxes. Partial records remain pending with their candidate "
            "boxes intact."
        ),
        "summary": {
            "records": len(records),
            "auto_validated_records": len(auto_validated),
            "complete_positive": states["complete_positive"],
            "complete_negative": states["complete_negative"],
            "positive_boxes": sum(row["positive_boxes"] for row in auto_validated),
            "candidate_boxes": sum(row["candidate_boxes"] for row in records),
            "pending_partial_records": len(pending_partial),
            "status_layers": {
                "source_available_records": len(records),
                "annotated_records": sum(
                    row["status_layers"]["annotated"] for row in records
                ),
                "candidate_records": len(records),
                "candidate_boxes": sum(row["candidate_boxes"] for row in records),
                "auto_validated_records": len(auto_validated),
                "auto_validated_positive_boxes": sum(
                    row["positive_boxes"] for row in auto_validated
                ),
                "human_semantic_confirmed_records": sum(
                    row["status_layers"]["human_confirmed"]["semantic"] for row in records
                ),
                "human_box_or_absence_confirmed_records": 0,
                "human_box_confirmed_boxes": 0,
                "training_admitted_records": 0,
                "pending_partial_records": len(pending_partial),
                "pending_partial_candidate_boxes": sum(
                    row["candidate_boxes"] for row in pending_partial
                ),
                "pending_human_semantic_confirmation_records": sum(
                    "human_semantic_confirmation" in row["pending"] for row in records
                ),
                "pending_human_label_confirmation_records": len(records),
                "pending_human_box_confirmation_boxes": sum(
                    row["positive_boxes"] for row in records
                ),
                "pending_training_admission_records": len(records),
            },
        },
        "records": records,
        "pending_partial": pending_partial,
        "boundary": (
            "Programmatic validation records only matching auto-review labels. It does not "
            "populate human semantic, human box/absence or mask confirmation, or training "
            "admission. Pending records retain their previous annotations and candidate boxes."
        ),
    }


def write_inventory(
    output: str | Path = DEFAULT_OUTPUT,
    status_layers: str | Path = ANNOTATION_LAYERS,
) -> tuple[Path, Path]:
    inventory = build_inventory()
    _assert_consistency(inventory)
    target = _workspace(output).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("output must stay inside the workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inventory["source_inventory"] = _relative(target)
    inventory["source_inventory_sha256"] = _sha256(target)
    layer_document = _status_layer_document(inventory)
    layer_document["source_inventory_sha256"] = _sha256(target)
    status_layers_target = _workspace(status_layers).resolve()
    if not status_layers_target.is_relative_to(ROOT.resolve()):
        raise ValueError("status-layer document must stay inside the workspace")
    status_layers_target.parent.mkdir(parents=True, exist_ok=True)
    status_layers_target.write_text(
        json.dumps(layer_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target, status_layers_target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--status-layers", default=ANNOTATION_LAYERS)
    args = parser.parse_args(argv)
    target, complete = write_inventory(args.output, args.status_layers)
    inventory = _read_json(target)
    layer_document = _read_json(complete)
    print(
        json.dumps(
            {
                "status": "written",
                "output": _relative(target),
                "status_layers": _relative(complete),
                "direct_review": inventory["summary"]["direct_review"],
                "status_layer_summary": layer_document["summary"],
                "manifest_only_candidates": inventory["summary"]["manifest_only_candidates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
