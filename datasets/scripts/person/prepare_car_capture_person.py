#!/usr/bin/env python3
"""Build the person direct-capture manifest from the 2026-11-12 car route."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GROUP = "car_real_capture_20261112"
RAW = Path("Archive/Assets/captures") / GROUP
ANNOTATED = Path("datasets/annotated") / GROUP
STEM = "V_20261112_152304"
VIDEO = RAW / f"{STEM}.mp4"
CSV = RAW / f"{STEM}_frames.csv"
COPY_MANIFEST = Path("datasets/source_copy_manifest.jsonl")
SEGMENTS = ANNOTATED / "segments_v1/segments.json"
FIELD = ANNOTATED / "field_eval_v1"
LAYERS = Path("datasets/scripts/person/config/direct_annotation_layers_20261115.json")
OUTPUT = ANNOTATED / "person_direct_captures.json"
WAYPOINT_OUTPUT = ANNOTATED / "waypoint_video_alignment.json"
CLASSES = ["community", "noncommunity"]
MIN_HEIGHT = 0.28 * 480  # PERSON-MIN-SCALE, native 640x480 images

# Existing content cuts, plus the tail of S013 where the PS4 group enters.
POINTS = {
    "P02_PS1": (212, 259),
    "P04_PS2": (405, 466),
    "P07_PS3": (638, 684),
    "P08_PS4": (700, 746),
    "P11_PS5": (935, 989),
    "P15_PS6": (1335, 1394),
}
NEGATIVE_SPANS = (
    (212, 216, "Only floor, fallen scooter and then white backs at the left edge."),
    (257, 259, "The final front exited; white backs and scooter remain."),
    (681, 684, "Only white backs, scooter and adults beyond the course."),
    (700, 701, "Thin white backs and the sign; the first front enters at 702."),
    (744, 746, "Panned to the buildings with only white backs/edges remaining."),
    (935, 939, "Approach shows the light and crossing; the first front enters at 940."),
    (988, 989, "The last front exited to the right; only the crossing and light remain."),
)


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def jsonl(path):
    with (ROOT / path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def frame_path(number):
    return RAW / "frames_20261112_152304" / f"{STEM}_{number:06d}.jpg"


def grey(path):
    image = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (480, 640):
        raise ValueError(f"Expected a 640x480 image at {path}")
    return image


def luma_thumbnail(image):
    return cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)


def mean_abs_diff(left, right):
    return float(np.abs(np.subtract(left, right, dtype=np.float32)).mean())


def video_alignment(frames):
    radius = 15
    wanted = {index for number in frames for index in range(number - radius, number + radius + 1)}
    timeline = {}
    decoder = cv2.VideoCapture(str(ROOT / VIDEO))
    try:
        for number in range(1, max(wanted) + 1):
            ok, bgr = decoder.read()
            if not ok:
                break
            if number in wanted:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                timeline[number] = luma_thumbnail(gray), decoder.get(cv2.CAP_PROP_POS_MSEC) / 1000
    finally:
        decoder.release()

    matches = {}
    for number, gray in frames.items():
        small = luma_thumbnail(gray)
        ranked = sorted(
            (float(np.subtract(small, timeline[index][0], dtype=np.float32).std()), index)
            for index in range(number - radius, number + radius + 1)
            if index in timeline
        )
        if len(ranked) < 2:
            matches[number] = {"state": "unlocated", "reason": "missing_decode_candidates"}
            continue
        (score, match), (second_score, _) = ranked[:2]
        evidence = {
            "method": "160x120 luma residual RMS after offset removal; search JPEG ordinal +/-15",
            "best_video_frame_number": match,
            "rms": round(score, 5),
            "runner_up_margin": round(second_score - score, 5),
        }
        if match == number and score < 3.0 and second_score - score > 0.03:
            evidence.update(
                state="content_and_sequence_match",
                video_frame_index=match - 1,
                pts_seconds=round(timeline[match][1], 6),
            )
        else:
            evidence["state"] = "unlocated"
        matches[number] = evidence
    return matches


def waypoint_alignment():
    """Find capture counterparts beyond the person windows without transferring labels."""
    point_paths = sorted((ROOT / RAW).glob("P*.png"))
    frame_paths = sorted((ROOT / RAW / "frames_20261112_152304").glob("*.jpg"))
    stills = [grey(path) for path in point_paths]

    def descriptor(gray):
        return cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA).ravel().astype(np.float32)

    point_descriptors = np.stack([descriptor(gray) for gray in stills])
    frame_descriptors = np.stack([descriptor(grey(path)) for path in frame_paths])
    distance = (
        (point_descriptors ** 2).sum(axis=1)[:, None]
        + (frame_descriptors ** 2).sum(axis=1)[None, :]
        - 2 * point_descriptors @ frame_descriptors.T
    )
    segment_rows = read_json(SEGMENTS)["segments"]
    originals = source_copy_records()
    verified_jpegs, records = {}, []
    for path, still, scores in zip(point_paths, stills, distance):
        candidates = np.argsort(scores)[:64]
        ranked = sorted((mean_abs_diff(still, grey(frame_paths[index])), index) for index in candidates)
        (score, index), (second, _) = ranked[:2]
        jpeg = frame_paths[index].relative_to(ROOT)
        number = int(jpeg.stem.rsplit("_", 1)[1])
        image = path.relative_to(ROOT).as_posix()
        same_capture = score < 1.0 and second - score > 0.5
        row = {
            "id": f"{GROUP}:{path.stem}",
            "image": image,
            "point": path.stem,
            "source_group": GROUP,
            "parent_video": VIDEO.as_posix(),
            "image_size": [640, 480],
            "frame_index": None,
            "timestamp_seconds": None,
            "segment": None,
            "evidence": {
                "source_copy": originals[image],
                "point_match": {
                    "state": "same_capture_visual_match" if same_capture else "unlocated",
                    "method": "all-JPEG 80x60 luma search; full-resolution grayscale MAE on 64 nearest images",
                    "best_jpeg": jpeg.as_posix(),
                    "mae": round(score, 5),
                    "runner_up_mae": round(second, 5),
                },
            },
        }
        if same_capture:
            segment = next(item for item in segment_rows if item["start_frame"] <= number <= item["end_frame"])
            row.update(
                matched_image=jpeg.as_posix(),
                capture_identity=f"{GROUP}:video:{number}",
                frame_index=number,
                segment=segment["segment_id"],
            )
            row["evidence"]["sequence_source"] = {"manifest": SEGMENTS.as_posix(), "episode": segment["episode_group"]}
            verified_jpegs[number] = grey(jpeg)
        records.append(row)
    alignment = video_alignment(verified_jpegs) if verified_jpegs else {}
    for row in records:
        match = alignment.get(row["frame_index"], {"state": "unlocated"})
        row["evidence"]["mp4"] = match
        row["timestamp_seconds"] = match.get("pts_seconds")
    return {
        "schema_version": 1,
        "source_group": GROUP,
        "parent_video": VIDEO.as_posix(),
        "frame_index_basis": "One-based JPEG export ordinal; MP4 zero-based decode index and PTS in evidence.",
        "summary": {
            "records": len(records),
            "matched_waypoint_jpegs": sum("matched_image" in row for row in records),
            "matched_waypoint_timestamps": sum(row["timestamp_seconds"] is not None for row in records),
        },
        "records": records,
    }


def person_instances(row):
    return [{**box, "class_name": CLASSES[box["class_id"]]} for box in row["boxes"]]


def person_label(instances, state, origin, evidence, layer="auto_candidate", **extra):
    complete = state in {"complete_positive", "complete_negative"}
    return {
        "state": state,
        "label_level": layer,
        "class_names": CLASSES,
        "instances": instances,
        "uses": ["direct_sample", "development_diagnostic"] if complete else ["review_candidate"],
        "evidence": {"origin": origin, **evidence},
        **extra,
    }


def point_label(row, manifest_ref, direct_layer):
    instances, excluded = [], []
    for box in person_instances(row):
        if box["asset_id"] in {"C6", "C16"} and "P08_PS4" in row["image"]:
            excluded.append({**box, "reason": "full silhouette below PERSON-MIN-SCALE"})
        else:
            instances.append(box)
    evidence = {
        **manifest_ref,
        "review": row["review_record"],
        "checked_frames": row["checked_frames"],
        "background_observations": row["excluded_observations"],
        "former_positive_boxes_below_minimum": excluded,
    }
    if "P11_PS5" in row["image"]:
        evidence["occlusion"] = "C7 is a nearby front; scooter hides the lower legs and shortens the visible box."
    layer = "auto_candidate"
    if direct_layer is not None:
        layer = "auto_validated"
        evidence["existing_direct_annotation_layers"] = {
            "manifest": LAYERS.as_posix(),
            "id": direct_layer["id"],
            "status_layers": direct_layer["status_layers"],
        }
    return person_label(instances, "complete_positive", "existing_point_review", evidence, layer)


def register_instances(still, gray, instances):
    """Translate point boxes in an otherwise stable view; retain ambiguous pans for review."""
    error = mean_abs_diff(still, gray)
    if error > 4.0:
        return None
    warped, correlations = [], []
    for box in instances:
        x0, y0, x1, y1 = box["xyxy"]
        # Use a core inset so a box touching the image edge can still be checked.
        left, top, right, bottom = x0 + 5, y0 + 5, x1 - 5, y1 - 5
        patch = still[top:bottom, left:right]
        search = gray[top - 5:bottom + 5, left - 5:right + 5]
        response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, (dx, dy) = cv2.minMaxLoc(response)
        dx, dy = dx - 5, dy - 5
        if confidence < 0.93 or abs(dx) >= 5 or abs(dy) >= 5:
            return None
        xyxy = [max(0, x0 + dx), max(0, y0 + dy), min(640, x1 + dx), min(480, y1 + dy)]
        warped.append({**box, "xyxy": xyxy})
        correlations.append(round(confidence, 5))
    return warped, {"full_image_luma_mae": round(error, 5), "local_template_ncc": correlations}


def source_copy_records():
    return {
        row["path"]: {"manifest": COPY_MANIFEST.as_posix(), "line": line, "original": row["original_path"]}
        for line, row in jsonl(COPY_MANIFEST)
        if row.get("source_group") == GROUP
    }


def source_maps():
    field_labels = {}
    for filename in ("annotations_root.json", "annotations_frames.json"):
        source = FIELD / filename
        for index, row in enumerate(read_json(source)):
            if row["model_key"] == "person":
                field_labels[Path(row["image"]).name] = row, {"manifest": source.as_posix(), "record_index": index}
    observations = {}
    for filename in ("sequence_instances_part1_manifest.jsonl", "sequence_instances_part2_manifest.jsonl"):
        source = ANNOTATED / filename
        for line, row in jsonl(source):
            if Path(row["path"]).parent.name.startswith("frames_"):
                number = row.get("frame_number", row.get("source_frame_sequence_index"))
                review = row.get("task_review", row.get("task_observations", {}))
                instances = row.get("observed_instances", row.get("instances", []))
                observations[number] = {
                    "manifest": source.as_posix(),
                    "line": line,
                    "review_state": review.get("person"),
                    "instances": [box for box in instances if box["task"] == "person"],
                }
    return field_labels, observations


def build():
    field_labels, observations = source_maps()
    originals = source_copy_records()
    segment_rows = read_json(SEGMENTS)["segments"]
    segment_by_frame = {
        number: row
        for row in segment_rows
        for number in range(row["start_frame"], row["end_frame"] + 1)
    }
    layer_by_image = {row["canonical_image"]: row for row in read_json(LAYERS)["records"]}
    point_by_frame = {number: point for point, (first, last) in POINTS.items() for number in range(first, last + 1)}
    numbers = sorted(point_by_frame.keys() | {181, 316, 361})
    frames = {number: grey(frame_path(number)) for number in numbers}
    stills = {point: grey(RAW / f"{point}.png") for point in POINTS}
    with (ROOT / CSV).open(encoding="utf-8-sig", newline="") as stream:
        csv_frames = {int(row["frame"]): row for row in csv.DictReader(stream)}
    alignment = video_alignment(frames)

    def record(path, number, point):
        video_match = alignment.get(number, {"state": "unlocated"})
        segment = segment_by_frame.get(number)
        evidence = {
            "source_copy": originals[path.as_posix()],
            "parent_video_original": originals[VIDEO.as_posix()]["original"],
            "mp4": video_match,
        }
        if number in csv_frames:
            evidence["csv"] = {
                "path": CSV.as_posix(),
                "basis": "CSV frame ordinal; wall/ROS clocks preserved separately from media PTS",
                **csv_frames[number],
            }
        if segment:
            evidence["sequence_source"] = {"manifest": SEGMENTS.as_posix(), "episode": segment["episode_group"]}
        return {
            "id": f"{GROUP}:{path.stem}",
            "image": path.as_posix(),
            "source_group": GROUP,
            "image_size": [640, 480],
            "parent_video": VIDEO.as_posix(),
            "point": point,
            "segment": segment["segment_id"] if segment else None,
            "frame_index": number,
            "timestamp_seconds": video_match.get("pts_seconds"),
            "evidence": evidence,
        }

    point_rows, point_anchors, records = {}, {}, []
    for point, (first, last) in POINTS.items():
        filename = f"{point}.png"
        ranked = sorted((mean_abs_diff(stills[point], frames[n]), n) for n in range(first, last + 1))
        (score, best), (second, _) = ranked[:2]
        anchor = best if score < 1.0 and second - score > 0.5 else None
        item = record(RAW / filename, anchor, point)
        item["evidence"]["point_match"] = {
            "state": "same_capture_visual_match" if anchor else "unlocated",
            "method": "full-resolution grayscale MAE across the existing episode window",
            "best_jpeg": frame_path(best).as_posix(),
            "mae": round(score, 5),
            "runner_up_mae": round(second, 5),
        }
        if anchor:
            item["capture_identity"] = f"{GROUP}:video:{anchor}"
            point_anchors[anchor] = item
        item["tasks"] = {"person": point_label(*field_labels[filename], layer_by_image.get(item["image"]))}
        point_rows[point] = item
        records.append(item)

    for number in numbers:
        path = frame_path(number)
        point = point_by_frame.get(number)
        item = record(path, number, point)
        observed = observations.get(number)
        if observed:
            item["evidence"]["instance_observations"] = {k: v for k, v in observed.items() if k != "instances"}
        if point in point_rows:
            still = point_rows[point]
            item["evidence"]["point_match"] = {"image": still["image"], "basis": "continuous observation window"}
        else:
            still = None

        if number in point_anchors:
            canonical = point_anchors[number]
            item["duplicate_of"] = canonical["image"]
            item["capture_identity"] = canonical["capture_identity"]
            item["evidence"]["point_match"] = canonical["evidence"]["point_match"]
            label = {**canonical["tasks"]["person"], "uses": ["same_capture_reference", "development_diagnostic"]}
        elif path.name in field_labels:
            existing, ref = field_labels[path.name]
            if number == 361:
                label = person_label(
                    [], "partial", "existing_frame_review",
                    {**ref, "reason": "Foreground leg occlusion prevents resolving the scale boundary from visible height."},
                    candidate_instances=person_instances(existing),
                )
            else:
                label = person_label([], "complete_negative", "existing_frame_review", {**ref, "notes": existing["notes"]})
        elif (negative := next((note for first, last, note in NEGATIVE_SPANS if first <= number <= last), None)):
            label = person_label([], "complete_negative", "transition_contact_sheet", {"observation": negative})
        elif still and (registered := register_instances(stills[point], frames[number], still["tasks"]["person"]["instances"])):
            instances, check = registered
            label = person_label(instances, "complete_positive", "registered_point_review", {"image": still["image"], **check})
            item["near_duplicate_group"] = still["image"]
        else:
            candidates = observed["instances"] if observed else []
            seen = bool(candidates) or bool(point)
            label = person_label(
                [], "partial" if seen else "unknown", "continuous_frame_review",
                {"reason": "Camera motion or crop prevented transfer of a complete in-scope label."},
                candidate_instances=candidates,
            )
        item["tasks"] = {"person": label}
        records.append(item)

    return {
        "schema_version": 1,
        "task": "person",
        "source_group": GROUP,
        "source_manifest": SEGMENTS.as_posix(),
        "parent_video": VIDEO.as_posix(),
        "split_policy": "Unassigned development evidence. Source pairing contains a proposed test allocation only.",
        "frame_index_basis": "One-based export ordinal. MP4 zero-based decode index and PTS are recorded separately.",
        "point_windows": {point: {"first_frame": first, "last_frame": last} for point, (first, last) in POINTS.items()},
        "summary": {
            "records": len(records),
            "state_counts": dict(Counter(row["tasks"]["person"]["state"] for row in records)),
            "visually_matched_frame_timestamps": sum(row["timestamp_seconds"] is not None for row in records[6:]),
            "same_capture_representation_pairs": len(point_anchors),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--waypoints-only", action="store_true", help="Build all point-to-video links without person labels.")
    args = parser.parse_args()
    manifest = waypoint_alignment() if args.waypoints_only else build()
    output = args.output or ROOT / (WAYPOINT_OUTPUT if args.waypoints_only else OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(manifest["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
