"""Screen a monocular inverse-depth prediction inside observed ground pixels."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .geometry import _segments


class RelativeDepthEstimator:
    """Optional, offline Depth Anything V2 inference and horizontal-flip check."""

    def __init__(self, model_directory, *, model_source=None, device=None):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForDepthEstimation, __version__

        directory = Path(model_directory)
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        if config.model_type != "depth_anything" or config.depth_estimation_type != "relative":
            raise ValueError("expected a local Depth Anything relative-depth model")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            directory, config=config, local_files_only=True, use_safetensors=True,
        ).to(self.device).eval()
        digest = sha256()
        with (directory / "model.safetensors").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.evidence = {"model": config.model_type, "source": model_source,
                         "weights_sha256": digest.hexdigest(), "runtime": f"transformers {__version__}",
                         "torch": torch.__version__, "device": self.device,
                         "views": ["original", "horizontal_flip"]}

    def predict(self, image_bgr):
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=[rgb, np.ascontiguousarray(rgb[:, ::-1])], return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs.to(self.device))
        original, flipped = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[rgb.shape[:2]] * 2,
        )
        return np.stack((original["predicted_depth"].cpu().numpy(),
                         flipped["predicted_depth"].cpu().numpy()[:, ::-1]))

    def observe(self, image_bgr, surface_mask):
        """Return the quality-screened local depth order and weight provenance."""
        mask = np.asarray(surface_mask, dtype=bool)
        if mask.shape != image_bgr.shape[:2]:
            raise ValueError("surface_mask must match the source image")
        if not mask.any():
            return {"status": "unknown", "basis": "no_ground_support", "model_evidence": self.evidence}
        return {**fit_relative_ground_depth(self.predict(image_bgr), mask), "model_evidence": self.evidence}


def _robust_plane(design, values):
    """Fit a plane with residual-based downweighting of contaminated tiles."""
    weights = np.ones(len(values))
    for _ in range(3):
        weighted = np.sqrt(weights)
        coeff, _, rank, _ = np.linalg.lstsq(design * weighted[:, None], values * weighted, rcond=None)
        if rank != 3:
            return None
        residual = values - design @ coeff
        deviation = np.abs(residual - np.median(residual))
        tolerance = max(1e-7, 2.5 * 1.4826 * np.median(deviation))
        weights = np.minimum(1, tolerance / np.maximum(deviation, 1e-7))
    return coeff


def fit_relative_ground_depth(predictions, surface_mask):
    """Fit an affine inverse-depth *index*, with farther 0 and nearer 1 anchors.

    Predictions are two aligned inverse-depth maps: original and unflipped
    horizontal mirror. The output has arbitrary offset and scale. The index
    is useful for ordering locations inside the observed surface and cannot
    be divided to obtain a size ratio or used as a distance or elevation.
    """
    mask = np.asarray(surface_mask, dtype=bool)
    depth = np.asarray(predictions, dtype=np.float32)
    if mask.ndim != 2 or depth.shape != (2, *mask.shape) or not np.isfinite(depth).all():
        raise ValueError("expected two finite aligned depth maps and one matching surface mask")
    height, width = mask.shape
    inner = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    ys, xs = np.nonzero(inner >= max(2, 0.006 * min(height, width)))
    if not len(xs):
        return {"status": "unknown", "basis": "insufficient_ground_support"}
    nx, ny = 16, 12
    tile = (ys * ny // height) * nx + xs * nx // width
    samples = []
    sampled_values = depth[:, ys, xs].T
    for index in np.unique(tile):
        selected = tile == index
        if selected.sum() >= max(8, 0.025 * width * height / (nx * ny)):
            samples.append((index, np.median(xs[selected]), np.median(ys[selected]),
                            *np.median(sampled_values[selected], axis=0)))
    if len(samples) < 12:
        return {"status": "unknown", "basis": "insufficient_ground_support",
                "diagnostics": {"sampled_tiles": len(samples)}}
    tiles, x, y, *model_scores = np.asarray(samples).T
    xy = np.column_stack((x, y))
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    area = cv2.contourArea(hull) / (width * height)
    diagnostics = {"sampled_tiles": len(samples), "support_hull_fraction": round(area, 5)}

    def unknown(basis):
        return {"status": "unknown", "basis": basis, "diagnostics": diagnostics}

    if area < 0.009:
        return unknown("insufficient_ground_extent")
    design = np.column_stack((x / width, y / height, np.ones_like(x)))
    folds = (tiles.astype(int) % nx + tiles.astype(int) // nx) % 2 == 0
    if min(folds.sum(), (~folds).sum()) < 5:
        return unknown("insufficient_holdout_tiles")
    coefficients, fit_errors, signal = [], [], []
    for values, prediction in zip(model_scores, depth):
        coeff = _robust_plane(design, values)
        if coeff is None:
            return unknown("insufficient_ground_extent")
        fitted = design @ coeff
        low, high = np.percentile(fitted, [10, 90])
        dynamic_range = float(np.ptp(np.percentile(prediction[::4, ::4], [5, 95])))
        if min(high - low, dynamic_range) <= 1e-6:
            return unknown("unresolved_depth_gradient")
        coefficients.append((coeff - [0, 0, low]) / (high - low))
        signal.append(float((high - low) / dynamic_range))
        holdout = np.empty(len(values))
        for test in (folds, ~folds):
            train_coeff = _robust_plane(design[~test], values[~test])
            if train_coeff is None:
                return unknown("insufficient_holdout_tiles")
            holdout[test] = design[test] @ train_coeff - values[test]
        fit_errors.append(float(np.percentile(np.abs(holdout), 80) / (high - low)))
    first, second = coefficients
    gradients = np.array([first[:2], second[:2]]) / [width, height]
    agreement = float(gradients[0] @ gradients[1] / np.prod(np.linalg.norm(gradients, axis=1)))
    profile_error = float(np.percentile(np.abs(design @ (first - second)), 90))
    diagnostics.update({"relative_signal_min": round(min(signal), 4),
                        "holdout_p80_error_max": round(max(fit_errors), 4),
                        "flip_gradient_cosine": round(agreement, 4),
                        "flip_profile_p90_error": round(profile_error, 4)})
    if min(signal) < 0.07:
        return unknown("unresolved_depth_gradient")
    if agreement < 0.94 or profile_error > 0.22:
        return unknown("depth_predictions_disagree")
    if max(fit_errors) > 0.32:
        return unknown("depth_not_consistent_with_local_plane")
    return {"status": "estimated", "basis": "screened_monocular_inverse_depth_order",
            "confidence": "model_consistent", "coordinate_space": "source_image_pixels",
            "image_size": [width, height], "support_polygon_px": hull.tolist(),
            "index_coefficients_uv1": ((first + second) / 2).tolist(),
            "index_anchors": {"farther_p10": 0, "nearer_p90": 1},
            "diagnostics": diagnostics}


def ground_relative_depth_index(cue, foot_xy, surface_mask):
    """Evaluate one source-pixel point against the matching ground observation."""
    if cue.get("status") != "estimated" or cue.get("basis") != "screened_monocular_inverse_depth_order":
        return None
    width, height = cue["image_size"]
    mask = np.asarray(surface_mask, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError("surface_mask must match the observed image size")
    point = np.asarray(foot_xy, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("expected a finite source-pixel point")
    x, y = point
    if not (0 <= x <= width - 1 and 0 <= y <= height - 1 and mask[round(y), round(x)]):
        return None
    polygon = np.asarray(cue["support_polygon_px"], dtype=np.float32)
    if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) < 0:
        return None
    return float(np.dot(cue["index_coefficients_uv1"], [x / width, y / height, 1]))


def _image_samples(image, points):
    height, width = image.shape[:2]
    x, y = np.rint(points).astype(int).T
    valid = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    return image[y.clip(0, height - 1), x.clip(0, width - 1)], valid


def ground_stripe_scale_cue(image_bgr, surface_mask):
    """Screen scale along an observed, approximately straight constant-width stripe.

    The two image edges delimit horizontal cross-sections on the same stripe.
    Their width ratio gives the inverse camera-depth ratio along the center ray
    if the physical stripe has parallel edges. It has no validity off that ray.
    """
    mask = np.asarray(surface_mask, dtype=bool)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if mask.shape != gray.shape:
        raise ValueError("surface_mask must match the source image")
    if not mask.any():
        return {"status": "unknown", "basis": "no_ground_support"}
    height, width = mask.shape
    short = min(height, width)
    diagonal = float(np.hypot(width, height))
    luma = gray.astype(np.float32)
    bright = (luma >= 145) & (np.ptp(image_bgr, axis=2) <= 48)
    gap = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    edges = []
    for line in _segments(gray):
        if line["length"] < 0.085 * diagonal or abs(line["dir_y"]) < 0.23:
            continue
        ends = np.array([[line["x1"], line["y1"]], [line["x2"], line["y2"]]])
        ends = ends[np.argsort(ends[:, 1])]
        direction = (ends[1] - ends[0]) / line["length"]
        normal = np.array([direction[1], -direction[0]])
        points = ends[0] + np.linspace(0.17, 0.83, 17)[:, None] * (ends[1] - ends[0])
        distance, _ = _image_samples(gap, points)
        if np.mean(distance < 0.045 * short) < 0.27:
            continue
        offset = max(2, 0.006 * short) * normal
        right, right_valid = _image_samples(luma, points + offset)
        left, left_valid = _image_samples(luma, points - offset)
        both = right_valid & left_valid
        to_right = float(np.mean(both & _image_samples(bright, points + offset)[0] & (right - left > 26)))
        to_left = float(np.mean(both & _image_samples(bright, points - offset)[0] & (left - right > 26)))
        if max(to_right, to_left) >= 0.43:
            edges.append((ends, direction, normal, to_right - to_left, max(to_right, to_left)))
    best = None
    for first, second in combinations(edges, 2):
        if first[3] * second[3] >= 0:
            continue
        left, right = (first, second) if first[3] > 0 else (second, first)
        low = max(left[0][0, 1], right[0][0, 1])
        high = min(left[0][1, 1], right[0][1, 1])
        span = high - low
        if span < 0.085 * height:
            continue
        rows = np.linspace(low + 0.08 * span, high - 0.08 * span, 21)
        x1, x2 = (ends[0, 0] + (rows - ends[0, 1]) * direction[0] / direction[1]
                  for ends, direction, _, _, _ in (left, right))
        widths = x2 - x1
        normal_widths = widths * (left[1][1] + right[1][1]) / 2
        if normal_widths.min() < max(2, 0.005 * short) or normal_widths.max() > 0.073 * short:
            continue
        centers = np.column_stack(((x1 + x2) / 2, rows))
        if np.mean(_image_samples(bright, centers)[0]) < 0.78:
            continue
        outside = max(3, 0.012 * short)
        l_dist, l_valid = _image_samples(gap, np.column_stack((x1, rows)) - outside * left[2])
        r_dist, r_valid = _image_samples(gap, np.column_stack((x2, rows)) + outside * right[2])
        if min(np.mean(l_valid & (l_dist < 0.02 * short)), np.mean(r_valid & (r_dist < 0.02 * short))) < 0.31:
            continue
        anchors = [2, -3]
        far_width, near_width = widths[anchors]
        ratio = float(near_width / far_width)
        lower, upper = (near_width - 1) / (far_width + 1), (near_width + 1) / (far_width - 1)
        if not 1.12 < ratio < 3.5 or lower <= 1 or upper - lower > 0.6 * ratio:
            continue
        score = span / height * min(left[4], right[4])
        if best is None or score > best[0]:
            best = score, {
                "status": "estimated", "basis": "ground_supported_stripe_width_change",
                "confidence": "conditional_on_parallel_stripe_edges", "coordinate_space": "source_image_pixels",
                "center_anchors_px": centers[anchors].tolist(), "anchor_widths_px": widths[anchors].tolist(),
                "near_over_far": ratio, "width_error_sensitivity": [float(lower), float(upper)],
                "width_error_px": 1.0, "edge_segments_xyxy": [left[0].ravel().tolist(), right[0].ravel().tolist()],
            }
    if best is None:
        return {"status": "unknown", "basis": "no_supported_stripe_width_change"}
    return best[1]


def ground_stripe_scale_ratio(cue, foot_xy, reference_xy):
    """Compare two points on the observed stripe center between its anchors."""
    if cue.get("status") != "estimated" or cue.get("basis") != "ground_supported_stripe_width_change":
        return None
    centers = np.asarray(cue["center_anchors_px"], dtype=float)
    points = np.asarray([foot_xy, reference_xy], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("expected two finite source-pixel points")
    y = points[:, 1]
    if np.any(y < centers[0, 1]) or np.any(y > centers[1, 1]):
        return None
    if np.any(np.abs(points[:, 0] - np.interp(y, centers[:, 1], centers[:, 0])) > 2):
        return None
    widths = np.interp(y, centers[:, 1], cue["anchor_widths_px"])
    return float(widths[0] / widths[1])
