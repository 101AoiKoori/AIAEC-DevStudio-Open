"""Batch spatial observations for the shared background pool.

The projection is a local, unitless XYZ frame for composition.  It is not a
metric reconstruction of the camera or the physical set.  Contact masks screen
visible floor, while task-specific labels and role decisions remain separate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from .frame_environment import DEFAULT_DATABASE, edit_database, workspace_path
from .paths import WORKSPACE
from .review.geometry import contact_reference_quad, estimate_ground_plane, ground_perspective_cues
from .review.ground_space import GroundSpace
from .review.placement import GroundPlacement, assess_placeable_ground
from .review.placement_record import decode_placement_record, encode_placement_record
from .review.relative_depth import RelativeDepthEstimator, ground_stripe_scale_cue


MANIFEST = WORKSPACE / "datasets/annotated/common/sources/background_pool_manifest.json"
OUTPUT = WORKSPACE / "datasets/annotated/common/perspective/background_pool_space_v4"
ALGORITHM = "shared_background_space_v4"
RELATIVE_ALGORITHM = "shared_background_space_v5_relative"
RELATIVE_OUTPUT = WORKSPACE / "datasets/annotated/common/perspective/background_pool_space_v5_relative"
EVIDENCE_KEY = "shared_background_space"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _frame_index(database):
    conn = sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True)
    try:
        return {
            row[0]: {"frame_id": row[1], "content_sha256": row[2], "width": row[3], "height": row[4]}
            for row in conn.execute(
                "SELECT p.path, f.frame_id, f.content_sha256, f.width, f.height "
                "FROM frame_paths p JOIN frames f USING(frame_id)"
            )
        }
    finally:
        conn.close()


def _valid_quad(points, width, height):
    quad = np.asarray(points, dtype=np.float64)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return False
    if np.any(quad < 0) or np.any(quad > (width - 1, height - 1)):
        return False
    if cv2.contourArea(quad.astype(np.float32)) < width * height * 0.005:
        return False
    # The far edge must really be above the near edge, and the patch must span depth.
    return bool(max(quad[0, 1], quad[1, 1]) + 0.08 * height < min(quad[2, 1], quad[3, 1]))


def _coarse_floor(width, height):
    """Near-field floor envelope only; it says nothing about object clearance."""
    far = round(0.68 * (height - 1))
    near = round(0.97 * (height - 1))
    return [[round(0.08 * (width - 1)), far], [round(0.92 * (width - 1)), far],
            [width - 1, near], [0, near]]


def _scene_mask(polygons, shape):
    if not polygons:
        return None
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or np.any(points < 0)
                or np.any(points > (shape[1] - 1, shape[0] - 1))
                or cv2.contourArea(points.astype(np.float32)) == 0):
            raise ValueError("scene evidence polygon must be nondegenerate and in source pixels")
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def _prompt_mask(model, image, prompts):
    """Apply one recorded SAM prompt in image space and verify its seed labels."""
    points = np.asarray(prompts["points_xy"], dtype=np.float64)
    labels = np.asarray(prompts["labels"])
    height, width = image.shape[:2]
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or labels.shape != (len(points),)
            or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > (width - 1, height - 1))
            or not np.isin(labels, (0, 1)).all() or not np.any(labels == 1)):
        raise ValueError("expected valid image-space SAM points and binary labels")
    # Keep a bounded, aspect-preserving inference input and restore the mask to source pixels.
    scale = min(1.0, 640 / max(height, width))
    small = image
    if scale < 1.0:
        small = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    result = model.predict(small, points=(points * (small.shape[1] / width, small.shape[0] / height)).tolist(),
                           labels=labels.tolist(), imgsz=640, verbose=False)[0]
    if result.masks is None or not len(result.masks.data):
        return None
    mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    x, y = np.rint(points).astype(np.intp).T
    return mask if np.array_equal(mask[y, x], labels) else None


def _automatic_floor_prompt(pool_row, shape):
    """Use existing source-pixel surface evidence to propose inset SAM seeds."""
    height, width = shape
    for task, entry in pool_row.get("tasks", {}).items():
        geometry = entry.get("geometry") or {}
        size = geometry.get("coordinate_size") or pool_row.get("image_size")
        if size != [width, height]:
            continue
        polygons = [region["polygon"] for region in geometry.get("regions", [])
                    if region.get("role") == "visible_ground_surface" and region.get("polygon")]
        # A fitted, task-cited crop is stronger evidence than a line-only floor observation.
        if (geometry.get("ground_polygon") and geometry.get("coordinate_size")
                and entry.get("evidence") and not geometry.get("ground_reason_codes")):
            polygons.append(geometry["ground_polygon"])
        if not polygons:
            continue
        # Some legacy crop records used the half-open image boundary as a corner.
        polygons = [np.clip(polygon, (0, 0), (width - 1, height - 1)).tolist() for polygon in polygons]
        try:
            mask = _scene_mask(polygons, shape)
        except ValueError:
            continue
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        rows = np.where(np.max(distance, axis=1) > 2)[0]
        if len(rows) < 3:
            continue
        probes = [int(rows[round(fraction * (len(rows) - 1))]) for fraction in (0.25, 0.50, 0.75)]
        return {"floor_polygons": polygons, "floor_prompts": {
            "points_xy": [[int(np.argmax(distance[y])), y] for y in probes], "labels": [1] * len(probes),
        }, "prompt_basis": "inset_points_in_existing_surface", "source_task": task,
            "reference": entry.get("evidence")}
    return None


def _refine_sparse_ground(image, placement, sam, *, source_kind=None):
    """Extend a sparse near-field observation using bounded automatic SAM prompts."""
    original = placement.candidate_mask
    pixels = int(original.sum())
    is_render = source_kind == "blender_render"
    if not 0 < pixels < (0.30 if is_render else 0.08) * original.size:
        return placement, None, None
    height, width = original.shape
    padded = np.pad(original.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1].copy()
    positive = []
    for _ in range(2):
        _, maximum, _, xy = cv2.minMaxLoc(distance)
        if maximum < 3:
            break
        positive.append(xy)
        cv2.circle(distance, xy, round(0.22 * width), 0, -1)
    if not positive:
        return placement, None, None
    # The upper background prior is explicitly part of this opt-in refinement.
    # It is a prompt exclusion, not recorded evidence of a physical obstacle.
    negative = [(round(fraction * (width - 1)), round(0.105 * (height - 1))) for fraction in (1 / 3, 0.70)]
    tested = []
    radius = max(1, round(0.004 * min(height, width)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    render_marker = None
    if is_render:
        # Bright lane blocks can be raised geometry above a dark rendered floor.
        # Keep them unknown unless the previous observation already blocked them.
        luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.percentile(luma[original], 95) < 80:
            render_marker = luma > np.percentile(luma[original], 99) + 18
    for count in range(len(positive), 0, -1):
        points = positive[:count]
        prompts = {"points_xy": [list(xy) for xy in (*points, *negative)], "labels": [1] * count + [0, 0]}
        tested.append(prompts)
        mask = _prompt_mask(sam, image, prompts) if sam is not None else None
        if mask is None or np.mean(mask[original] > 0) < 0.55:
            continue
        _, components = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        seeded = np.isin(components, [components[y, x] for x, y in points]) & (components != 0)
        clear = (seeded & ~placement.obstacle_mask).astype(np.uint8)
        if render_marker is not None:
            clear[render_marker] = 0
        accepted = cv2.erode(clear, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if int(accepted.sum()) <= 1.2 * pixels:
            continue
        return (GroundPlacement("candidate", "sam_contact_seeded_floor", accepted, placement.obstacle_mask),
                {"prompt_basis": "inset_contact_seeds_with_upper_background_prior",
                 "floor_prompts": prompts, "prompt_attempts": tested,
                 "bright_render_markers_excluded": render_marker is not None}, "supported")
    return (placement, {"prompt_basis": "inset_contact_seeds_with_upper_background_prior", "prompt_attempts": tested},
            "segmenter_unavailable" if sam is None else "prompt_unresolved")


def _space_record(image, pool_row, frame, *, index, scene_evidence=None, sam=None, model_evidence=None,
                  refine_sparse=False, prior_obstacle_mask=None):
    height, width = image.shape[:2]
    plane = estimate_ground_plane(image)
    found = plane.floor_region if plane.ok and _valid_quad(plane.floor_region, width, height) else None
    proposal = found or _coarse_floor(width, height)
    proposal_basis = "line_and_texture_floor" if found else "near_field_scene_prior"
    evidence = scene_evidence or {}
    floor = _scene_mask(evidence.get("floor_polygons"), (height, width))
    obstacles = _scene_mask(evidence.get("obstacle_polygons"), (height, width))
    prompt_outcomes = {}
    for name in ("floor", "obstacle"):
        if not (prompts := evidence.get(f"{name}_prompts")):
            continue
        if sam is None:
            prompt_outcomes[name] = "segmenter_unavailable"
            continue
        mask = _prompt_mask(sam, image, prompts)
        prompt_outcomes[name] = "supported" if mask is not None else "prompt_unresolved"
        if mask is not None:
            if name == "floor":
                floor = mask if floor is None else floor & mask
            else:
                obstacles = mask if obstacles is None else obstacles | mask
    if any(evidence.get(f"{name}_prompts") for name in ("floor", "obstacle")):
        evidence = {**evidence, **(model_evidence or {})}
    # Exact-image surface evidence can extend beyond a flawed projection proposal.
    search = proposal if floor is None else [[0, 0], [width - 1, 0],
                                             [width - 1, height - 1], [0, height - 1]]
    placement = assess_placeable_ground(image, search, floor_mask=floor, obstacle_mask=obstacles)
    if not evidence and sam is not None and placement.status == "unknown":
        if automatic := _automatic_floor_prompt(pool_row, (height, width)):
            # Keep the original finding when the model cannot honor its seeds.
            mask = _prompt_mask(sam, image, automatic["floor_prompts"])
            evidence = {**automatic, **(model_evidence or {})}
            prompt_outcomes["floor"] = "supported" if mask is not None else "prompt_unresolved"
            if mask is not None:
                floor = mask & _scene_mask(automatic["floor_polygons"], (height, width))
                search = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
                placement = assess_placeable_ground(image, search, floor_mask=floor, obstacle_mask=obstacles)
    if refine_sparse and not evidence and placement.status == "candidate":
        placement, refinement, outcome = _refine_sparse_ground(image, placement, sam,
                                                              source_kind=pool_row.get("source_kind"))
        if refinement is not None:
            evidence = {**refinement, **(model_evidence or {})}
            prompt_outcomes["floor_refinement"] = outcome
    if prior_obstacle_mask is not None:
        if floor is None or not evidence.get("floor_polygons"):
            raise ValueError("preserving current obstacles requires exact-image floor evidence")
        scoped_obstacle = (placement.obstacle_mask & (floor > 0)) | prior_obstacle_mask
        contact = placement.candidate_mask & ~scoped_obstacle
        evidence = {**evidence, "obstacle_scope": {
            "basis": "retain_prior_obstacles_and_limit_new_clutter_to_observed_floor",
            "retained_pixels": int(prior_obstacle_mask.sum()),
            "added_pixels": int((scoped_obstacle & ~prior_obstacle_mask).sum()),
        }}
        placement = GroundPlacement("candidate" if contact.any() else "unknown", placement.basis,
                                    contact, scoped_obstacle)
    polygon = contact_reference_quad(placement.candidate_mask)
    geometry_basis = "contact_supported_reference" if polygon else "no_contact_reference"
    support = placement.candidate_mask
    if floor is not None:
        support = support | ((floor > 0) & ~placement.obstacle_mask)
    perspective = ground_perspective_cues(image, support)
    # Preserve the old scalar field only as a center-row approximation of a
    # separately verified line. Callers needing scale should consume that line.
    horizon = None
    if perspective["scale"]["status"] == "estimated":
        a, b, c = perspective["scale"]["horizon_line_px"]
        horizon = -(a * (width - 1) / 2 + c) / b
    if not polygon or (horizon is not None and horizon >= min(p[1] for p in polygon) - 0.02 * height):
        horizon = None
    scale_basis = perspective["scale"]["basis"] if perspective["scale"]["status"] == "estimated" else None
    local_frame = height_reference = None
    if polygon:
        space = GroundSpace(polygon, image_shape=image.shape, horizon_y=horizon, far_scale=0.6)
        reference_xy = (0.5, 0.85)
        foot = space.footpoint(reference_xy)
        top = (foot[0], foot[1] - 0.30 * height)
        vertical_vp = (foot[0], -4.0 * height, 1.0)
        local_frame = {"origin": "far_left_reference_corner", "x": "right", "y": "toward_camera",
                       "z": "up", "extent_xyz": [1.0, 1.0, None], "far_scale": 0.6,
                       "basis": "contact_reference_grid"}
        height_reference = {"local_xyz": [0.5, 0.85, 1.0], "image_top_xy": list(top),
                            "vertical_vanishing_point": list(vertical_vp),
                            "basis": "configured_image_height_and_vp",
                            "meaning": "one local height unit spans 30 percent of image height at the anchor"}
    contact_pixels = int(np.count_nonzero(placement.candidate_mask))
    status = "contact_candidate_unverified" if contact_pixels else "space_only"
    search_bbox = None
    if contact_pixels:
        x, y, w, h = cv2.boundingRect(placement.candidate_mask.astype(np.uint8))
        search_bbox = [x, y, x + w, y + h]
    payload = {
        "status": status,
        "geometry_basis": geometry_basis,
        "scale_basis": scale_basis,
        "coordinate_system": "local_unitless_xyz" if polygon else "source_image_pixels",
        "image_size": [width, height],
        "ground_polygon": polygon,
        "proposal_polygon": proposal,
        "proposal_basis": proposal_basis,
        "scene_evidence": evidence or None,
        "prompt_outcomes": prompt_outcomes,
        "camera_calibration": None,
        "placement_patch": plane.placement_patch if found else [],
        "horizon_y": horizon,
        "estimated_global_horizon_y": plane.horizon_y,
        "vanishing_points": plane.vanishing_points if plane.ok else [],
        "line_residual": plane.residual,
        "ground_reason_codes": plane.reason_codes,
        "local_frame": local_frame,
        "projected_headings_deg": None,
        "perspective_cues": perspective,
        "height_reference": height_reference,
        "placement_screen": {"status": placement.status, "basis": placement.basis,
                             "contact_pixels": contact_pixels,
                             "contact_search_bbox_xyxy": search_bbox,
                             "contact_fraction": round(contact_pixels / (width * height), 6),
                             "obstacle_fraction": round(float(np.mean(placement.obstacle_mask)), 6)},
        "placement_record": encode_placement_record(
            placement, reference_quad_px=polygon, horizon_y=horizon
        ),
        "pool_indices": [index],
        "pool_paths": [pool_row["image"]],
        "source_kind": pool_row["source_kind"],
        "parent_group": pool_row["parent_group"],
        "limits": ["no_metric_scale", "single_view_vertical_reference_assumed",
                   "floor_like_pixels_can_include_existing_objects",
                   "contacts_are_machine_candidates_not_task_eligibility"],
    }
    return payload


def read_spatial(image, *, database=DEFAULT_DATABASE):
    """Read one current shared spatial observation without decoding its image."""
    path = workspace_path(image)
    conn = sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT f.frame_id, f.content_sha256, o.payload_json FROM frame_paths p "
            "JOIN frames f USING(frame_id) JOIN observations o USING(frame_id) "
            "WHERE p.path=? AND o.task='' AND o.kind='geometry' AND o.scope='shared_background' "
            "AND o.evidence_key=? AND o.current=1", (path, EVIDENCE_KEY)
        ).fetchone()
        if row is None:
            return None
        return {"frame_id": row[0], "content_sha256": row[1], **json.loads(row[2])}
    finally:
        conn.close()


def projection_from_spatial(record):
    """Reconstruct the configured XYZ grid for layout, independent of scene-axis cues."""
    reference = record["height_reference"]
    frame = record["local_frame"]
    if not reference or not frame:
        raise ValueError("shared spatial record has no supported local projection")
    space = GroundSpace(record["ground_polygon"], image_shape=record["image_size"][::-1],
                        horizon_y=record["horizon_y"], far_scale=frame["far_scale"])
    return space.with_height_reference(reference["local_xyz"], reference["image_top_xy"],
                                       vertical_vanishing_point=reference["vertical_vanishing_point"])


def _reuse_space_observations(paths):
    reused = {}
    sources = []
    for path in paths:
        relative = workspace_path(path)
        data = (WORKSPACE / relative).read_bytes()
        sources.append({"path": relative, "sha256": sha256(data).hexdigest()})
        for line in data.decode("utf-8").splitlines():
            previous = json.loads(line)
            frame_id = previous["frame_id"]
            if frame_id in reused:
                raise ValueError(f"multiple source spatial records for frame {frame_id}")
            reused[frame_id] = previous
    return reused, sources


def _current_space_payloads(database):
    conn = sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True)
    try:
        return dict(conn.execute(
            "SELECT frame_id, payload_json FROM observations "
            "WHERE task = '' AND kind = 'geometry' AND scope = 'shared_background' "
            "AND evidence_key = ? AND current = 1", (EVIDENCE_KEY,),
        ))
    finally:
        conn.close()


def audit(*, manifest=MANIFEST, output=None, database=DEFAULT_DATABASE, write=False,
          qa_only=False, indices=None, scene_evidence=None, sam_weights=None, refine_sparse=False,
          reuse_space_jsonl=(), relative_depth_model=None, relative_depth_model_source=None,
          preserve_current_obstacles=False):
    relative_cues = bool(reuse_space_jsonl or relative_depth_model is not None)
    if relative_cues and (not qa_only or write):
        raise ValueError("relative depth comparison requires a QA-only run")
    if reuse_space_jsonl and (scene_evidence or sam_weights or refine_sparse):
        raise ValueError("cannot regenerate floor evidence when replaying saved spatial records")
    if preserve_current_obstacles and (not scene_evidence or not qa_only or write or reuse_space_jsonl):
        raise ValueError("preserving current obstacles requires a QA-only run with exact scene evidence")
    if output is None:
        output = RELATIVE_OUTPUT if relative_cues else OUTPUT
    reused, reused_sources = _reuse_space_observations(reuse_space_jsonl)
    previous_observations = _current_space_payloads(database) if preserve_current_obstacles else {}
    depth = None
    if relative_depth_model is not None:
        depth = RelativeDepthEstimator(WORKSPACE / workspace_path(relative_depth_model),
                                       model_source=relative_depth_model_source)
    pool = json.loads(Path(manifest).read_text(encoding="utf-8"))
    rows = pool["records"]
    selected = set(range(len(rows))) if indices is None else set(indices)
    if not selected or not selected <= set(range(len(rows))):
        raise ValueError("select valid shared pool indices")
    evidence = json.loads(Path(scene_evidence).read_text(encoding="utf-8")) if scene_evidence else {}
    observations = evidence.get("frames_by_sha256", {})
    sam = None
    model_evidence = None
    if sam_weights is not None:
        weights = WORKSPACE / workspace_path(sam_weights)
        model_digest = sha256(weights.read_bytes()).hexdigest()
        if evidence.get("weights_sha256") and model_digest != evidence["weights_sha256"]:
            raise ValueError("SAM weights differ from the recorded prompt evidence")
        try:
            from ultralytics import SAM, __version__ as runtime_version
        except ModuleNotFoundError as error:
            if error.name != "ultralytics":
                raise
        else:
            sam = SAM(str(weights))
            model_evidence = {"weights_sha256": model_digest, "weights_name": weights.name,
                              "weights_source": evidence.get("weights_source"),
                              "runtime": f"ultralytics {runtime_version}"}
    indexed = _frame_index(database)
    result = []
    by_frame = {}
    counts = Counter()
    by_parent = defaultdict(Counter)
    for index, row in enumerate(rows):
        if index not in selected:
            continue
        path = workspace_path(row["image"])
        frame = indexed.get(path)
        if frame is None:
            raise ValueError(f"Pool frame not registered: {path}")
        data = (WORKSPACE / path).read_bytes()
        digest = sha256(data).hexdigest()
        if digest != frame["content_sha256"]:
            raise ValueError(f"Pool frame bytes differ from database: {path}")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != (frame["width"], frame["height"]):
            raise ValueError(f"Pool image decode or size failed: {path}")
        record = by_frame.get(frame["frame_id"])
        if record is None:
            if reuse_space_jsonl:
                previous = reused.get(frame["frame_id"])
                if previous is None or previous["content_sha256"] != digest:
                    raise ValueError(f"missing or mismatched saved spatial record: {path}")
                payload = previous["payload"].copy()
                if payload["image_size"] != [frame["width"], frame["height"]] or payload["parent_group"] != row["parent_group"]:
                    raise ValueError(f"saved spatial record coordinate or parent mismatch: {path}")
                payload.update(pool_indices=[index], pool_paths=[path])
            else:
                scene = observations.get(digest)
                prior_obstacle = None
                if preserve_current_obstacles:
                    previous = json.loads(previous_observations[frame["frame_id"]])
                    prior_obstacle = decode_placement_record(previous["placement_record"]).obstacle_mask
                payload = _space_record(image, row, frame, index=index, scene_evidence=scene, sam=sam,
                                        model_evidence=model_evidence, refine_sparse=refine_sparse,
                                        prior_obstacle_mask=prior_obstacle)
            if relative_cues:
                mask = decode_placement_record(payload["placement_record"]).candidate_mask
                cue = depth.observe(image, mask) if depth is not None else {"status": "unknown", "basis": "model_not_requested"}
                stripe = ground_stripe_scale_cue(image, mask)
                perspective = payload.get("perspective_cues") or ground_perspective_cues(image, mask)
                payload["perspective_cues"] = {**perspective,
                                               "relative_depth": cue, "local_stripe_scale": stripe}
                counts[f"direction:{perspective['direction']['status']}"] += 1
                counts[f"horizon_scale:{perspective['scale']['status']}"] += 1
                counts[f"relative_depth:{cue['basis']}"] += 1
                counts[f"local_stripe_scale:{stripe['basis']}"] += 1
            record = {"frame_id": frame["frame_id"], "content_sha256": digest,
                      "image": path, "payload": payload}
            if reuse_space_jsonl and previous.get("source_observation") is not None:
                record["source_observation"] = previous["source_observation"]
            result.append(record)
            by_frame[frame["frame_id"]] = record
        else:
            if record["content_sha256"] != digest or record["payload"]["parent_group"] != row["parent_group"]:
                raise ValueError(f"Pool aliases disagree on frame identity: {path}")
            record["payload"]["pool_indices"].append(index)
            record["payload"]["pool_paths"].append(path)
        payload = record["payload"]
        counts[payload["status"]] += 1
        counts[payload["geometry_basis"]] += 1
        counts[f"placement_basis:{payload['placement_screen']['basis']}"] += 1
        by_parent[row["parent_group"]][payload["status"]] += 1
    if write:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with edit_database(database) as writer:
            for record in result:
                writer.observe(
                    record["frame_id"], kind="geometry", scope="shared_background",
                    evidence_key=EVIDENCE_KEY, task=None, origin="machine",
                    algorithm_version=ALGORITHM,
                    reference={"manifest": workspace_path(manifest),
                               "pool_indices": record["payload"]["pool_indices"],
                               "content_sha256": record["content_sha256"]},
                    payload=record["payload"],
                )
    if write or qa_only:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "observations.jsonl").write_text(
            "\n".join(_json(record) for record in result) + "\n", encoding="utf-8"
        )
    summary = {"pool_records": len(rows), "selected_pool_paths": len(selected), "distinct_frames": len(result),
               "prompt_segmenter": "available" if sam is not None else "not_loaded",
               "counts": dict(sorted(counts.items())),
               "parents": {key: dict(value) for key, value in sorted(by_parent.items())},
               "database_written": bool(write), "algorithm_version": RELATIVE_ALGORITHM if relative_cues else ALGORITHM}
    if relative_cues:
        summary["reused_space_sources"] = reused_sources
        summary["relative_depth_model"] = depth.evidence if depth is not None else None
    if write or qa_only:
        (Path(output) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write observations and contact masks")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--qa-only", action="store_true", help="write the QA snapshot without writing the database")
    parser.add_argument("--index", type=int, action="append", help="limit to selected shared-pool indices")
    parser.add_argument("--scene-evidence", type=Path)
    parser.add_argument("--sam-weights", type=Path, help="local weights for recorded mask prompts")
    parser.add_argument("--refine-sparse", action="store_true",
                        help="try automatic prompts on a sparse existing contact mask")
    parser.add_argument("--reuse-space-jsonl", type=Path, action="append", default=[],
                        help="keep exact stored masks and append relative cues in a QA-only run")
    parser.add_argument("--relative-depth-model", type=Path, help="offline Depth Anything relative model directory")
    parser.add_argument("--relative-depth-model-source", help="model provenance URL recorded with the output")
    parser.add_argument("--preserve-current-obstacles", action="store_true",
                        help="keep current blockers when repairing floor with exact scene evidence")
    args = parser.parse_args()
    if args.write and args.qa_only:
        parser.error("use either --write or --qa-only")
    print(json.dumps(audit(manifest=args.manifest, output=args.output,
                           database=args.database, write=args.write, qa_only=args.qa_only, indices=args.index,
                           scene_evidence=args.scene_evidence, sam_weights=args.sam_weights,
                           refine_sparse=args.refine_sparse, reuse_space_jsonl=args.reuse_space_jsonl,
                           relative_depth_model=args.relative_depth_model,
                           relative_depth_model_source=args.relative_depth_model_source,
                           preserve_current_obstacles=args.preserve_current_obstacles),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
