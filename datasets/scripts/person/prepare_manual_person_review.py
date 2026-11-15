"""Prepare a bounded full-frame review package for real person labels.

The source manifests contain AI-reviewed candidate boxes. Those boxes are
proposals, not human truth. This tool writes review overlays and a draft label
manifest with empty ``accepted_by`` fields. It never promotes a record to
complete and does not copy source media.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from datasets.scripts.person.human_review_state import backup_files, has_human_content, merge_cases

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = "datasets/scripts/person/config/manual_person_review_20261115.json"
DEFAULT_OUTPUT = "datasets/annotated/common/review/person/manual_complete_20261115"


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: str | Path) -> str:
    return _workspace_path(path).resolve().relative_to(ROOT.resolve()).as_posix()


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


def _source_records(plan: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    selected: dict[str, dict[str, Any]] = {}
    manifests: dict[str, Path] = {}
    for source in plan.get("sources") or []:
        if not isinstance(source, Mapping):
            raise ValueError("source entries must be objects")
        name = str(source.get("name") or "")
        manifest_value = source.get("manifest")
        if not name or not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError("source requires name and manifest")
        manifest = _workspace_path(manifest_value)
        document = _read_json(manifest)
        key = str(source.get("records_key") or "samples")
        records = document.get(key)
        if not isinstance(records, list):
            raise ValueError(f"{_relative(manifest)} lacks list {key!r}")
        manifests[name] = manifest
        for record in records:
            if not isinstance(record, Mapping):
                continue
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                continue
            selected[f"{name}:{record_id}"] = {
                **dict(record),
                "_source_name": name,
                "_manifest": manifest,
            }
    return selected, manifests


def _image_path(record: Mapping[str, Any]) -> Path:
    value = record.get("image")
    if not isinstance(value, str) or not value:
        raise ValueError("record image is required")
    candidate = _workspace_path(value)
    if not candidate.exists():
        candidate = Path(str(record["_manifest"])).parent / value
    if not candidate.exists():
        raise FileNotFoundError(f"selected image does not exist: {value}")
    return candidate.resolve()


def _parent_video(record: Mapping[str, Any]) -> str | None:
    value = record.get("source_video") or record.get("parent_video")
    return str(value) if value else None


def _xyxy(value: Any, width: int, height: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"invalid xyxy: {value!r}")
    x1, y1, x2, y2 = (int(round(float(item))) for item in value)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"empty xyxy after clipping: {value!r}")
    return [x1, y1, x2, y2]


def _touches_edge(box: Iterable[float], width: int, height: int, margin: int = 2) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _infer_visibility(box: list[int], *, note: str, width: int, height: int) -> str:
    if _touches_edge(box, width, height):
        return "cropped"
    if any(token in note for token in ("遮挡", "occlud", "重叠")):
        return "occluded"
    return "unknown"


def _case_source_group(record: Mapping[str, Any], source_name: str) -> str:
    value = record.get("source_group")
    if value:
        return str(value)
    if source_name == "multivideo":
        return "site_capture_person_standees"
    return "site_capture_full_scene"


def _draft_record(
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    class_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    review_id = str(case.get("review_id") or "")
    if not review_id:
        raise ValueError("case review_id is required")
    image_path = _image_path(record)
    image = _read_image(image_path)
    height, width = image.shape[:2]
    note = str(record.get("notes") or "")
    instances: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for index, box_record in enumerate(record.get("boxes") or [], start=1):
        if not isinstance(box_record, Mapping):
            continue
        box = _xyxy(box_record.get("xyxy"), width, height)
        class_id = box_record.get("class_id")
        proposal_class = class_id if type(class_id) is int and 0 <= class_id < len(class_names) else None
        instance_id = f"person_{index}"
        visibility = _infer_visibility(box, note=note, width=width, height=height)
        instances.append(
            {
                "id": instance_id,
                "class_id": proposal_class,
                "visibility": visibility,
                "visible_bbox_xyxy": box,
                "visibility_evidence": f"{_relative(record['_manifest'])}#record_id={record.get('id')}",
                "proposal": {
                    "class_name": class_names[proposal_class] if proposal_class is not None else "unknown",
                    "appearance_asset": box_record.get("appearance_asset"),
                    "reflection_status": box_record.get("reflection_status"),
                },
            }
        )
        table_rows.append(
            {
                "review_id": review_id,
                "source": record["_source_name"],
                "record_id": record.get("id"),
                "row_id": instance_id,
                "nature": "candidate_task_target",
                "proposed_class": class_names[proposal_class] if proposal_class is not None else "unknown",
                "view": "front",
                "visibility": visibility,
                "xyxy": box,
                "action": "confirm_or_correct_visible_box",
                "note": note,
            }
        )
    hard_negatives = []
    for index, raw_box in enumerate(record.get("white_back_reference_boxes") or [], start=1):
        box = _xyxy(raw_box, width, height)
        hard_negatives.append(
            {
                "id": f"hard_negative_{index}",
                "nature": "white_back",
                "view": "back",
                "visible_box_xyxy": box,
                "annotation_action": "hard_negative_no_box",
            }
        )
        table_rows.append(
            {
                "review_id": review_id,
                "source": record["_source_name"],
                "record_id": record.get("id"),
                "row_id": f"hard_negative_{index}",
                "nature": "white_back",
                "proposed_class": "none",
                "view": "back",
                "visibility": "full",
                "xyxy": box,
                "action": "confirm_no_target_box",
                "note": "白背为硬负例；不计人物正例",
            }
        )
    if not table_rows:
        table_rows.append(
            {
                "review_id": review_id,
                "source": record["_source_name"],
                "record_id": record.get("id"),
                "row_id": "full_frame",
                "nature": "candidate_no_target_objects",
                "proposed_class": "none",
                "view": "unknown",
                "visibility": "unknown",
                "xyxy": [],
                "action": "confirm_empty_or_add_targets",
                "note": note,
            }
        )
    parent_video = _parent_video(record)
    parents = []
    if parent_video:
        parents.append({"path": parent_video, "role": "camera_source"})
    parents.append(
        {
            "path": _relative(record["_manifest"]),
            "role": "candidate_review_manifest",
            "record_id": record.get("id"),
        }
    )
    draft = {
        "id": review_id,
        "source_record_id": record.get("id"),
        "image": _relative(image_path),
        "source_kind": "direct_camera_capture",
        "class_names": class_names,
        "label_state": "partial",
        "partial_reason": "pending_human_full_frame_box_review",
        "review": {
            "full_frame_checked": False,
            "accepted_by": None,
            "box_accepted_by": None,
            "method": "pending_human_full_frame_review",
            "source_candidate_state": record.get("state"),
            "source_note": note,
            "evidence": _relative(record["_manifest"]),
        },
        "parent_sources": parents,
        "instances": instances,
        "hard_negatives": hard_negatives,
        "source_group": _case_source_group(record, str(record["_source_name"])),
        "timestamp_seconds": record.get("source_time_seconds") or record.get("timestamp_seconds"),
    }
    return draft, table_rows, image


def _draw_box(
    image: np.ndarray,
    box: Iterable[int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    top = max(0, y1 - size[1] - 7)
    cv2.rectangle(image, (x1, top), (x1 + size[0] + 6, y1), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _overlay(image: np.ndarray, draft: Mapping[str, Any]) -> np.ndarray:
    canvas = image.copy()
    for instance in draft.get("instances") or []:
        proposed = instance.get("proposal") or {}
        class_name = str(proposed.get("class_name") or "unknown")
        label = f"{instance['id']}:{class_name}"
        _draw_box(canvas, instance["visible_bbox_xyxy"], label, (30, 180, 30))
    for index, negative in enumerate(draft.get("hard_negatives") or [], start=1):
        _draw_box(canvas, negative["visible_box_xyxy"], f"H{index}:no-box", (30, 30, 220))
    header = (
        f"{draft['id']}  source={draft['source_record_id']}  "
        f"proposed_targets={len(draft.get('instances') or [])}  "
        f"hard_negatives={len(draft.get('hard_negatives') or [])}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 27), (12, 12, 12), -1)
    cv2.putText(
        canvas,
        header,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(
    items: list[tuple[str, np.ndarray]],
    output: Path,
    *,
    title: str,
) -> None:
    if not items:
        raise ValueError("contact sheet needs at least one item")
    width, height = 640, 360
    columns = 4
    label_height = 30
    rows = math.ceil(len(items) / columns)
    sheet = np.full((rows * (height + label_height) + 42, columns * width, 3), 241, np.uint8)
    cv2.putText(
        sheet,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for index, (label, image) in enumerate(items):
        row, column = divmod(index, columns)
        x = column * width
        y = 42 + row * (height + label_height)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        sheet[y : y + height, x : x + width] = resized
        cv2.rectangle(sheet, (x, y + height), (x + width, y + height + label_height), (30, 30, 30), -1)
        cv2.putText(
            sheet,
            label,
            (x + 8, y + height + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    _write_image(output, sheet)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "review_id",
        "source",
        "record_id",
        "row_id",
        "nature",
        "proposed_class",
        "view",
        "visibility",
        "xyxy",
        "action",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["xyxy"] = json.dumps(item.get("xyxy") or [], ensure_ascii=False)
            writer.writerow(item)


def build_package(
    plan_path: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path | None = None,
    rebuild_review: bool = False,
) -> dict[str, Any]:
    global ROOT
    previous_root = ROOT
    if root is not None:
        ROOT = Path(root).resolve()
    try:
        plan_file = _workspace_path(plan_path)
        plan = _read_json(plan_file)
        sources, manifests = _source_records(plan)
        output = _workspace_path(output_dir)
        drafts: list[dict[str, Any]] = []
        review_cases: list[dict[str, Any]] = []
        table_rows: list[dict[str, Any]] = []
        original_items: list[tuple[str, np.ndarray]] = []
        overlay_items: list[tuple[str, np.ndarray]] = []
        overlay_dir = output / "overlays"
        for case in plan.get("cases") or []:
            if not isinstance(case, Mapping):
                raise ValueError("case entries must be objects")
            key = f"{case.get('source')}:{case.get('record_id')}"
            record = sources.get(key)
            if record is None:
                raise ValueError(f"selected source record not found: {key}")
            draft, rows, image = _draft_record(case, record, class_names=list(plan["class_names"]))
            drafts.append(draft)
            table_rows.extend(rows)
            review_id = draft["id"]
            _write_image(overlay_dir / f"{review_id}.png", _overlay(image, draft))
            original_items.append((f"{review_id}  {draft['source_record_id']}", image))
            overlay_items.append((f"{review_id}  candidate boxes: green=target / red=hard-negative", _overlay(image, draft)))
            review_cases.append(
                {
                    "review_id": review_id,
                    "source_record_id": draft["source_record_id"],
                    "image": draft["image"],
                    "original": draft["image"],
                    "overlay": _relative(overlay_dir / f"{review_id}.png"),
                    "source_group": draft["source_group"],
                    "timestamp_seconds": draft["timestamp_seconds"],
                    "current_state": "partial",
                    "proposed_target_count": len(draft["instances"]),
                    "proposed_hard_negative_count": len(draft["hard_negatives"]),
                    "human_decision": None,
                    "human_box_corrections": [],
                    "accepted_by": None,
                    "full_frame_checked": False,
                }
            )
        if len(drafts) != len(plan.get("cases") or []):
            raise ValueError("not every configured case produced a draft")
        if len({draft["image"] for draft in drafts}) != len(drafts):
            raise ValueError("review package contains duplicate source images")
        output.mkdir(parents=True, exist_ok=True)
        _contact_sheet(original_items, output / "contact_sheet_originals.png", title=f"{plan['task']} {plan['batch']} originals")
        _contact_sheet(overlay_items, output / "contact_sheet_overlays.png", title=f"{plan['task']} {plan['batch']} box proposals")
        _write_table(output / "review_table.csv", table_rows)
        review_document = {
            "schema_version": 1,
            "task": plan["task"],
            "batch": plan["batch"],
            "purpose": plan["purpose"],
            "plan": _relative(plan_file),
            "plan_sha256": _sha256(plan_file),
            "source_manifests": {
                name: {"path": _relative(path), "sha256": _sha256(path)}
                for name, path in sorted(manifests.items())
            },
            "review_status": {
                "state": "pending_human_full_frame_review",
                "accepted_by": None,
                "box_accepted_by": None,
            },
            "cases": review_cases,
        }
        draft_document = {
            "schema_version": 1,
            "task": plan["task"],
            "batch": plan["batch"],
            "review_plan": _relative(plan_file),
            "records": drafts,
            "boundary": (
                "Draft candidate boxes only. All records are partial and have empty "
                "accepted_by until a named human completes full-frame review."
            ),
        }
        review_path = output / "review_cases.json"
        draft_path = output / "label_manifest.draft.json"
        backups: list[Path] = []
        preserved_ids: list[str] = []
        preserved_manifest = False
        existing_review = _read_json(review_path) if review_path.exists() else None
        existing_draft = _read_json(draft_path) if draft_path.exists() else None
        human_review = bool(existing_review and has_human_content(existing_review))
        human_draft = bool(existing_draft and has_human_content(existing_draft))
        if rebuild_review and (human_review or human_draft):
            backups = backup_files([path for path in (review_path, draft_path) if path.exists()])
        elif human_review:
            review_document, preserved_ids = merge_cases(existing_review, review_document)
            review_document["merge_policy"] = "preserved_nonempty_human_fields"
        if backups:
            review_document["explicit_rebuild"] = True
            review_document["backups"] = [_relative(path) for path in backups]
        _write_json(review_path, review_document)
        _write_json(output / "label_manifest.candidate.json", draft_document)
        if not human_draft or rebuild_review:
            _write_json(draft_path, draft_document)
        else:
            preserved_manifest = True
        instructions = (
            "人工审查只需按 P01–P24 给出自然语言结论：整帧是否正确、哪些人物应新增/删除/改类、"
            "哪些属于白背/侧背/远景/侧向硬负例。现有绿框与红框只是建议，不是真值。"
            "确认完成后由脚本归一化 JSON、重建 YOLO 标签并做 label audit。\n"
        )
        (output / "REVIEW_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8", newline="\n")
        return {
            "status": "written",
            "output": _relative(output),
            "cases": len(drafts),
            "proposed_target_instances": sum(len(draft["instances"]) for draft in drafts),
            "proposed_hard_negatives": sum(len(draft["hard_negatives"]) for draft in drafts),
            "originals_contact_sheet": _relative(output / "contact_sheet_originals.png"),
            "overlays_contact_sheet": _relative(output / "contact_sheet_overlays.png"),
            "review_table": _relative(output / "review_table.csv"),
            "draft_manifest": _relative(output / "label_manifest.draft.json"),
            "candidate_manifest": _relative(output / "label_manifest.candidate.json"),
            "preserved_human_cases": preserved_ids,
            "preserved_human_manifest": preserved_manifest,
            "backups": [_relative(path) for path in backups],
        }
    finally:
        ROOT = previous_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rebuild-review",
        action="store_true",
        help="explicitly replace existing human review files after backing them up",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_package(args.plan, args.output, rebuild_review=args.rebuild_review),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
