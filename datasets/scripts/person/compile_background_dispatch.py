"""Bind automatic background observations to a person candidate scene plan."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
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


def _scaled_empty_observation(evidence):
    """Use the recorded pixel-scale search when revising an existing decision."""
    search = evidence.get("emptiness_search") or {}
    if search.get("reference_frame_height") and evidence.get("automatic_observation_code_hash"):
        return evidence.get("automatic_empty_observation")
    return None


def environment_dispatch(database):
    """Freeze the eligible full-frame person backgrounds for a new plan."""
    from collections import Counter

    from datasets.scripts.common.frame_environment import query_frames, verified_image
    from datasets.scripts.common.source_history import read_source_history
    from datasets.scripts.common.source_ledger.config import build_groups, load_source_config
    from datasets.scripts.common.source_ledger.reservations import candidate_reservation
    from datasets.scripts.person.planned_sources import environment_background_row, recorded_ground

    source = load_source_config(ROOT, 'person')
    groups = build_groups('person', source.get('pairing') or {})
    history = read_source_history(ROOT)
    skipped, rows, parent_splits = Counter(), [], {}
    reservation_conflicts = []
    for decision in query_frames(task='person', role='composite_background', database=database):
        image, split, parent = decision['image'], decision['split'], decision['parent_group']
        if split not in ('train', 'val', 'test') or decision['scope']['kind'] != 'frame' or not parent:
            skipped['split_scope_or_parent'] += 1
            continue
        reservation = candidate_reservation(decision, split, groups, history)
        if reservation['status'] == 'conflict':
            skipped['source_reservation_conflict'] += 1
            reservation_conflicts.append({'frame_id': decision['frame_id'], 'record_id': decision['record_id'],
                                          'parent_group': parent, 'split': split, 'reservation': reservation})
            continue
        label = decision['label']
        boxes = label['instances']
        if (label['completeness'] != 'complete' or label['presence'] not in ('present', 'absent') or
                bool(boxes) != (label['presence'] == 'present')):
            raise ValueError(f'Person background decision disagrees with its full-frame labels: {image}')
        width, height = decision['image_size']
        if min(width, height) <= 1 or (width * 9 != height * 16 and width * 3 != height * 4):
            skipped['unsupported_aspect_ratio'] += 1
            continue
        geometry = decision['geometry'].get('placeable_ground')
        if not geometry or not geometry.get('reference_quad_px'):
            skipped['missing_contact_mask_or_reference_quad'] += 1
            continue
        row = environment_background_row(decision)
        _, ground = recorded_ground(row)
        if ground.status != 'candidate':
            raise ValueError(f'Person background decision has no valid contact pixels: {image}')
        verified_image(decision)
        rows.append(row)
        parent_splits.setdefault(parent, {}).setdefault(split, []).append(decision['record_id'])
    contacts = [{'parent_group': parent, 'splits': splits}
                for parent, splits in sorted(parent_splits.items()) if len(splits) > 1]
    return {'schema_version': 1, 'task': 'person', 'split_assignment': 'source_pairing',
            'source_relation_basis': 'frame_environment', 'records': rows,
            'skipped': dict(skipped), 'source_contacts': contacts,
            'reservation_conflicts': reservation_conflicts}


def compile_dispatch(*, dry_run: bool = False, update_plan: bool = True) -> dict:
    selection = _read(SELECTION)
    if selection.get("task") != "person":
        raise ValueError("person background review selection must declare task person")
    plan = _read(PLAN)
    if plan.get("model_key") != "person" or plan.get("target_split") != "train":
        raise ValueError("expected a person train plan")
    if not plan.get("strata"):
        raise ValueError("person train plan has no states")
    backgrounds = [item for item in selection["records"] if item.get("role", "background") == "background"]
    ready = [item for item in backgrounds if item.get("verdict") == "ready"]
    candidates = []
    for item in backgrounds:
        evidence = item.get("evidence") or {}
        empty = evidence.get("automatic_empty_observation") or {}
        if (item.get("verdict") == "unknown" and
                "placement_patch_unsure" in item.get("reason_codes", []) and
                (evidence.get("measured_placement_patch") or evidence.get("measured_ground_polygon")) and
                evidence.get("measured_ground_foot_xy") and
                empty.get("clean") and not empty.get("suspects")):
            candidates.append(item)
    rows = []
    for item in ready + candidates:
        evidence = item.get("evidence") or {}
        observation = evidence.get("automatic_empty_observation")
        if observation is not None and (not observation.get("clean") or observation.get("suspects")):
            continue
        frame = item["frame"]
        image_path = _relative_file(frame["path"])
        parent = frame.get("parent_video")
        reviewer = item.get("accepted_by") or None
        parent_verifier = evidence.get("parent_relation_verified_by")
        if parent:
            _relative_file(parent)
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError(f"unsupported background dimensions: {frame['path']}")
        if item.get("existing_targets"):
            raise ValueError(f"ready background has existing person targets: {frame['path']}")
        height, width = image.shape[:2]
        ground = evidence.get("measured_placement_patch") or item.get("ground_polygon") or evidence.get("measured_ground_polygon")
        ground_foot = item.get("foot_xy") or evidence.get("measured_ground_foot_xy")
        polygon = [[max(0, min(639, round(x * 640 / width))),
                    max(0, min(359, round(y * 360 / height)))] for x, y in ground]
        foot = [max(0, min(639, round(ground_foot[0] * 640 / width))),
                max(0, min(359, round(ground_foot[1] * 360 / height)))]
        row = {
            "image": frame["path"], "path": frame["path"],
            "source_video": parent, "source_group": frame.get("source_group"),
            "state": "confirmed_empty" if reviewer and item["verdict"] == "ready" else "candidate_complete_negative",
            "boxes": [], "split": "train",
            "human_accepted": bool(reviewer), "accepted_by": reviewer,
            "review_batch": evidence.get("source_batch", selection["batch"]), "review_verdict": item["verdict"],
            "review_reason_codes": item.get("reason_codes") or [],
            "review_evidence": evidence,
            "parent_link_basis": "review_selection_record",
            "parent_link_verified_by": parent_verifier,
            "ground_polygon_640x360": polygon, "foot_xy_640x360": foot,
            "source_image_size": [image.shape[1], image.shape[0]],
        }
        if observation is not None:
            row["semantic_status"] = "auto_candidate_complete_negative"
            row["automatic_empty_observation"] = observation
            if evidence.get("automatic_observation_code_hash"):
                row["automatic_observation_code_hash"] = evidence["automatic_observation_code_hash"]
        rows.append(row)
    previous = _read(DISPATCH) if DISPATCH.exists() else {}
    previous_dispatch = previous.get("records", [])
    explicit_splits = previous.get("split_assignment") == "source_pairing"
    combined = copy.deepcopy(previous_dispatch)
    existing_by_image = {row["image"]: index for index, row in enumerate(combined)}
    for item in backgrounds:
        index = existing_by_image.get(item.get("frame", {}).get("path"))
        if index is None:
            continue
        existing = combined[index]
        evidence = item.get("evidence") or {}
        observation = _scaled_empty_observation(evidence)
        if (existing["state"] == "candidate_complete_negative" and
                not existing.get("boxes") and
                observation is not None and (observation.get("suspects") or item.get("existing_targets"))):
            existing.update(
                state="unknown", semantic_status="unknown", review_verdict=item["verdict"],
                review_reason_codes=item.get("reason_codes") or [],
                review_evidence={**(existing.get("review_evidence") or {}), **evidence},
                automatic_empty_observation=observation,
                automatic_observation_code_hash=evidence["automatic_observation_code_hash"],
            )
    for row in rows:
        existing_index = existing_by_image.get(row["image"])
        if existing_index is None:
            if explicit_splits:
                row["split"] = "unassigned"
            existing_by_image[row["image"]] = len(combined)
            combined.append(row)
            continue
        existing = combined[existing_index]
        if existing.get("boxes") or existing.get("state") == "excluded":
            continue
        if existing.get("state") == "unknown" and _scaled_empty_observation(row["review_evidence"]) is None:
            continue
        if explicit_splits:
            for key in ("split", "source_video", "source_group"):
                row[key] = existing.get(key, row[key])
        if existing.get("placement_region_basis"):
            row["ground_polygon_640x360"] = existing["ground_polygon_640x360"]
            row["foot_xy_640x360"] = existing["foot_xy_640x360"]
        if not row["accepted_by"] and existing.get("accepted_by"):
            row["accepted_by"] = existing["accepted_by"]
            row["human_accepted"] = existing.get("human_accepted", True)
            if existing["state"] != "unknown" or "automatic_empty_observation" not in row:
                row["state"] = existing["state"]
                if existing.get("semantic_status"):
                    row["semantic_status"] = existing["semantic_status"]
        if not row["parent_link_verified_by"]:
            row["parent_link_verified_by"] = existing.get("parent_link_verified_by")
        row["review_evidence"] = {**(existing.get("review_evidence") or {}), **row["review_evidence"]}
        existing.update(row)
    rows = combined
    if not rows:
        raise ValueError("selection has no backgrounds with a usable automatic observation")
    dispatch = {
        **previous,
        "schema_version": 1, "task": "person", "batch": selection["batch"],
        "review_selection": "datasets/scripts/person/config/review_selection.json",
        "source_relation_basis": "review_selection_record",
        "records": rows,
    }
    manifest = "datasets/scripts/person/config/background_dispatch.json"
    updated = copy.deepcopy(plan)
    updated["purpose"] = "Person candidate batch using automatically ready backgrounds."
    indices = [index for index, row in enumerate(rows) if not explicit_splits or (
        row["split"] == "train" and not row.get("source_scene")
        and row["state"] in ("confirmed_empty", "candidate_complete_negative"))]
    if update_plan and not indices:
        raise ValueError("selection has no train backgrounds with complete person observations")
    for index, state in enumerate(updated["strata"] if update_plan else []):
        row_index = indices[index % len(indices)]
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
            else "Archive/Assets/captures/site_capture_20261112/videos/full_scene.mp4"
        )
        retained = set(state["sources"]) - {old_background, old_video}
        retained.add(row["image"])
        if row["source_video"]:
            retained.add(row["source_video"])
        state["sources"] = sorted(retained)
    if not dry_run:
        _write(DISPATCH, dispatch)
        if update_plan:
            _write(PLAN, updated)
        source_text = SOURCE.read_text(encoding="utf-8")
        if "  reviewed_background:" not in source_text:
            anchor = "  cross_task_review:"
            if source_text.count(anchor) != 1:
                raise ValueError("source role insertion point is ambiguous")
            source_text = source_text.replace(anchor, f"  reviewed_background: {manifest}\n{anchor}")
            SOURCE.write_text(source_text, encoding="utf-8")
    return {"ready_backgrounds": len(ready), "backgrounds": len(rows),
            "dispatched_states": len(updated["strata"]) if update_plan else 0,
            "parent_groups": sorted({row["source_group"] for row in rows if row["source_group"]}), "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch-only", action="store_true", help="update backgrounds without rewriting the existing plan")
    parser.add_argument('--environment-database', type=Path, help='read person background decisions from this database')
    parser.add_argument('--environment-snapshot', type=Path, help='write a new person dispatch filename in config/')
    args = parser.parse_args()
    if args.environment_snapshot and not args.environment_database:
        parser.error('--environment-snapshot requires --environment-database')
    if args.environment_database:
        snapshot = environment_dispatch(args.environment_database)
        if args.environment_snapshot and not args.dry_run:
            destination = (ROOT / args.environment_snapshot).resolve()
            if destination.parent != CONFIG or destination.suffix != '.json':
                parser.error('environment dispatch snapshot must be a new JSON file in person/config')
            if not snapshot['records']:
                parser.error('no full-frame backgrounds have placement evidence for this planner')
            with destination.open('x', encoding='utf-8') as stream:
                json.dump(snapshot, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        print(json.dumps({'backgrounds': len(snapshot['records']), 'skipped': snapshot['skipped'],
                          'source_contacts': snapshot['source_contacts'],
                          'reservation_conflicts': snapshot['reservation_conflicts']}, ensure_ascii=False))
        return
    print(json.dumps(compile_dispatch(dry_run=args.dry_run, update_plan=not args.dispatch_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
