"""Ground plane, visible floor and placement patch for candidate review.

Three layers, deliberately separate:

* ``horizon_y`` / ``vanishing_points`` / ``homography`` describe the **ground
  plane**: where the floor's perspective structure is, plus a metric-free top-down
  reference frame for it.
* ``floor_region`` is the part of the **visible floor** in the image.
* ``placement_patch`` is the local ground area where a target of the declared
  apparent size can actually stand. Composition uses only this patch; the plane
  supplies projection and scale constraints.

Nothing here is metric: no camera height, no ground scale, no intrinsics recovery.
The plane is a projective reference frame, and the patch is expressed in apparent
(image) size because the declared requirement is an apparent target height.

``residual`` is the inlier-share deficit of the two vanishing-point families,
``1 - inlier_length / candidate_length``: ``0`` means every detected segment
supports one of the two directions, ``1`` means none does. It measures line
structure agreement, not a pixel fit error; ``1 - residual`` is the usual support
share.

This stays independent of the experimental estimator in
``lab/training_common/ground_placement.py``: that one returns an ad-hoc region for
scene placement, this one returns schema records for review.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .schema import GroundObservation, GroundPlaneObservation
from .texture import texture_floor_top

__all__ = [
    "DEFAULT_PARAMS",
    "MIN_IMAGE_SIDE",
    "estimate_ground_plane",
    "estimate_ground_region",
    "foot_from_polygon",
    "polygon_area_ratio",
]

MIN_IMAGE_SIDE = 32

# Callers pass this dict through ``params``; unknown keys are rejected so a typo
# cannot silently fall back to a default.
DEFAULT_PARAMS = {
    "clearance_win_px": 24,
    "clearance_edge_density_max": 0.06,
    "clearance_min_component_ratio": 0.002,
    "ground_band_fraction": 0.4,
    "ransac_iters": 400,
    "seed": 0,
    "target_height_fraction": 0.30,
    "target_aspect": 0.42,
    "grid_step_px": 8,
    "horizon_max_slope_deg": 1.5,
    "min_patch_depth_fraction": 0.10,
    "texture_fallback_confidence_min": 0.40,
    "texture_top_margin_fraction": 0.01,
    "texture_max_horizon_fraction": 0.85,
}

# Segment detection: the longest segments are kept, shorter ones are texture.
_MAX_SEGMENTS = 320
_MIN_SEGMENT_FRACTION = 0.02  # of the image diagonal
# Segments steeper than |dy| > this * |dx| are treated as vertical structure.
_MAX_VERTICAL_RATIO = 6.0
_CANNY_LOW = 50
_CANNY_HIGH = 150

# Vanishing points: a segment supports a hypothesis while the angle between its
# direction and the ray to the point stays inside this tolerance.
_VP_TOLERANCE_DEG = 4.0
_MIN_PAIR_ANGLE_DEG = 5.0
_MIN_FAMILY_SEGMENTS = 4
_MIN_FAMILY_SHARE = 0.08  # of the candidate segment length
_MIN_COMBINED_SHARE = 0.35  # of the candidate segment length
_MIN_DIRECTION_SEPARATION_DEG = 12.0
# A vanishing point beyond this many image diagonals is treated as being at
# infinity: only its direction is used.
_VP_FAR_LIMIT = 20.0

# Horizon: reject a line closer to vertical than this |cos| value. The camera is
# mounted level, so a horizon steeper than ``DEFAULT_PARAMS["horizon_max_slope_deg"]``
# is not accepted; the row then comes from the transverse family's intersections.
_MIN_HORIZON_COS = 0.2
_MAX_INTERSECTION_MEMBERS = 40

# Floor top edge: horizon-parallel scan lines. A line's rectified texture energy
# (median gradient x depth) should stay inside this band of the near-ground
# reference while the scan is still on the floor.
_TEXTURE_LOW = 0.4
_TEXTURE_HIGH = 2.5
_TEXTURE_SAMPLES = 96
_TEXTURE_POINTS = 64
_MIN_FLOOR_EXTENT = 0.03  # of the image height

# Placement grid caps keep the search bounded on any image size.
_MAX_GRID_CELLS = 40000
_MAX_GRID_AXIS = 420

# Ground/vertical-surface junction ("wall base"): a bright, low-saturation, long
# horizontal stripe (painted line or a lit kerb) in the lower part of the frame.
_WALL_SEARCH_BOTTOM = 0.7  # only the lower this fraction of the frame is searched
_WALL_BRIGHT_VALUE = 200  # HSV V floor
_WALL_BRIGHT_SATURATION = 60  # HSV S ceiling
# Painted lines in older captures are dimmer than fresh paint, so the same detector
# runs a second, documented relaxation when the strict thresholds find nothing.
_WALL_BRIGHT_STAGES = ((200, 60), (140, 90))
_WALL_ROW_COVERAGE = 0.30  # share of the row width covered by bright pixels
_WALL_JITTER = 1  # rows of tolerance for a hand-painted line
_WALL_MARGIN = 0.01  # the floor's top edge sits this far above the junction

# Placement patch acceptance, matching the judge's rule: a patch is the rectangle a
# composited target may stand in, so a crossed or thin quad is worse than none.
_PATCH_MIN_AREA = 1200.0
_PATCH_MIN_AREA_RATIO = 0.0025
_PATCH_SIDE_RATIO_LOW = 0.125
_PATCH_SIDE_RATIO_HIGH = 8.0
_PATCH_MIN_PERSPECTIVE = 1.05  # near edge over far edge
_PATCH_MAX_AREA_RATIO = 0.45  # share of the frame; the judge only sanity-checks this
_MAX_PATCH_DEPTH_SAMPLES = 400
# Depth windows to try, as a factor of the nearest accepted depth. A ground rectangle
# tapers like near/far depth, so the whole floor is tried first and the window is
# shortened until the projected quad passes the taper rule.
_PATCH_DEPTH_WINDOWS = (1.0e9, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5)


def _as_gray(value: np.ndarray) -> np.ndarray:
    """Return a contiguous 2-D uint8 grayscale image."""

    array = np.asarray(value)
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif channels >= 1:
            array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    if array.size == 0:
        raise ValueError("expected a non-empty image")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8) * 255)
    work = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
    if float(work.max()) <= 1.0:
        work = work * 255.0
    return np.ascontiguousarray(np.clip(work, 0.0, 255.0).astype(np.uint8))


def _resolve_params(params: Optional[dict]) -> Dict[str, Any]:
    """Merge ``params`` over :data:`DEFAULT_PARAMS`, rejecting unknown keys."""

    options: Dict[str, Any] = dict(DEFAULT_PARAMS)
    if params:
        for key, value in dict(params).items():
            if key not in options:
                raise ValueError("unknown ground parameter: %r" % (key,))
            options[key] = value
    if int(options["ransac_iters"]) < 0:
        raise ValueError("ransac_iters must be non-negative")
    for name in ("target_height_fraction", "target_aspect", "grid_step_px"):
        if float(options[name]) <= 0.0:
            raise ValueError("%s must be positive" % (name,))
    slope_limit = float(options["horizon_max_slope_deg"])
    if not 0.0 < slope_limit < 90.0:
        raise ValueError("horizon_max_slope_deg must be in (0, 90)")
    return options


# --------------------------------------------------------------- segmentation


def _hough_segments(gray: np.ndarray, min_length: float) -> List[Tuple[float, float, float, float]]:
    """Fallback segment list for OpenCV builds without the LSD implementation."""

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(24, int(min_length)),
        minLineLength=int(round(min_length)),
        maxLineGap=max(4, int(round(min_length / 3.0))),
    )
    if lines is None:
        return []
    return [tuple(float(value) for value in row) for row in np.asarray(lines).reshape(-1, 4)]


def _segments(gray: np.ndarray) -> List[dict]:
    """Detect long line segments, preferring LSD and falling back to HoughLinesP."""

    height, width = gray.shape[:2]
    min_length = max(12.0, _MIN_SEGMENT_FRACTION * math.hypot(height, width))
    found: List[Tuple[float, float, float, float]] = []
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except (AttributeError, cv2.error):
        detector = None
    if detector is not None:
        try:
            raw = detector.detect(gray)[0]
        except cv2.error:
            raw = None
        if raw is not None:
            found = [tuple(float(value) for value in row) for row in np.asarray(raw).reshape(-1, 4)]
    if not found:
        found = _hough_segments(gray, min_length)
    entries: List[dict] = []
    for x1, y1, x2, y2 in found:
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length or length <= 0.0:
            continue
        # A near-vertical segment is a wall, pole or standee edge, never a line lying
        # on the ground plane; keeping it would let vertical structure outvote the
        # two ground directions.
        if abs(dy) > _MAX_VERTICAL_RATIO * abs(dx):
            continue
        entries.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mid_x": (x1 + x2) / 2.0,
                "mid_y": (y1 + y2) / 2.0,
                "dir_x": dx / length,
                "dir_y": dy / length,
                "length": length,
                "angle": math.atan2(dy, dx) % math.pi,
            }
        )
    if len(entries) > _MAX_SEGMENTS:
        entries.sort(key=lambda item: (-item["length"], item["x1"], item["y1"]))
        entries = entries[:_MAX_SEGMENTS]
    return entries


# ----------------------------------------------------------- direction families


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean direction modulo pi, in [0, pi)."""

    doubled = 2.0 * angles
    sine = float((weights * np.sin(doubled)).sum())
    cosine = float((weights * np.cos(doubled)).sum())
    if float(np.hypot(sine, cosine)) <= 0.0:
        return float(angles[0])
    return (math.atan2(sine, cosine) / 2.0) % math.pi


def _angle_distance(left: float, right: float) -> float:
    difference = abs(left - right) % math.pi
    return min(difference, math.pi - difference)


def _cluster_directions(segments: Sequence[dict]) -> List[List[int]]:
    """Split segments into two direction families with a circular 2-means.

    The two peaks of a weighted 1-degree histogram seed the centres and the
    assignment is deterministic, so the same segments always yield the same
    families. A single family is returned when the directions do not separate, and
    the caller then rejects the frame.
    """

    if len(segments) < 2:
        return []
    angles = np.array([segment["angle"] for segment in segments], dtype=np.float64)
    weights = np.array([segment["length"] for segment in segments], dtype=np.float64)
    bins = 180
    indices = np.clip((angles / math.pi * bins).astype(int), 0, bins - 1)
    histogram = np.zeros(bins)
    np.add.at(histogram, indices, weights)
    smoothed = histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
    first = int(np.argmax(smoothed))
    distances = np.degrees(np.abs(((np.arange(bins) - first + bins // 2) % bins) - bins // 2))
    separated = np.where(distances >= _MIN_DIRECTION_SEPARATION_DEG, smoothed, -1.0)
    if float(separated.max()) <= 0.0:
        return [list(range(len(segments)))]
    second = int(np.argmax(separated))
    centers = [first * math.pi / bins, second * math.pi / bins]
    assignment = np.full(len(segments), -1, dtype=int)
    for _ in range(4):
        first_distance = np.abs([_angle_distance(angle, centers[0]) for angle in angles])
        second_distance = np.abs([_angle_distance(angle, centers[1]) for angle in angles])
        updated = (second_distance < first_distance).astype(int)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for label in (0, 1):
            members = assignment == label
            if members.any():
                centers[label] = _circular_mean(angles[members], weights[members])
    groups = [sorted(int(index) for index in np.where(assignment == 0)[0]),
              sorted(int(index) for index in np.where(assignment == 1)[0])]
    groups = [group for group in groups if group]
    if len(groups) == 2 and math.degrees(_angle_distance(centers[0], centers[1])) < _MIN_DIRECTION_SEPARATION_DEG:
        return [sorted(groups[0] + groups[1])]
    return groups


# ------------------------------------------------------------ vanishing points


def _intersect(point: np.ndarray, direction: np.ndarray,
               other_point: np.ndarray, other_direction: np.ndarray) -> Optional[np.ndarray]:
    """Intersection of two infinite lines, or ``None`` when nearly parallel."""

    denominator = float(direction[0] * other_direction[1] - direction[1] * other_direction[0])
    if abs(denominator) < 1e-9:
        return None
    delta = other_point - point
    scale = float(delta[0] * other_direction[1] - delta[1] * other_direction[0]) / denominator
    return point + scale * direction


def _support(point: np.ndarray, midpoints: np.ndarray, directions: np.ndarray,
             lengths: np.ndarray, sin_tolerance: float) -> Tuple[float, np.ndarray]:
    """Length-weighted support for ``point`` as a vanishing point."""

    offsets = point[None, :] - midpoints
    norms = np.hypot(offsets[:, 0], offsets[:, 1])
    cross = directions[:, 0] * offsets[:, 1] - directions[:, 1] * offsets[:, 0]
    sine = np.abs(cross) / np.maximum(norms, 1e-6)
    hits = (sine <= sin_tolerance) | (norms <= 1.0)
    return float(lengths[hits].sum()), hits


def _refine_point(midpoints: np.ndarray, normals: np.ndarray, weights: np.ndarray,
                  constants: np.ndarray) -> Optional[np.ndarray]:
    """Weighted least-squares point closest to the given supporting lines."""

    matrix = np.array(
        [
            [float((weights * normals[:, 0] * normals[:, 0]).sum()),
             float((weights * normals[:, 0] * normals[:, 1]).sum())],
            [float((weights * normals[:, 0] * normals[:, 1]).sum()),
             float((weights * normals[:, 1] * normals[:, 1]).sum())],
        ]
    )
    vector = -np.array(
        [
            float((weights * normals[:, 0] * constants).sum()),
            float((weights * normals[:, 1] * constants).sum()),
        ]
    )
    scale = float(np.abs(matrix).sum())
    if scale <= 0.0:
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-6 * scale * scale:
        return None
    try:
        solution = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(solution).all():
        return None
    return solution


def _canonical(direction: np.ndarray) -> np.ndarray:
    """Fix the sign of a line direction so reports are reproducible."""

    if direction[0] < 0.0 or (abs(direction[0]) <= 1e-12 and direction[1] < 0.0):
        return -direction
    return direction


def _vanishing_point(
    segments: Sequence[dict],
    members: Sequence[int],
    rng: np.random.Generator,
    iters: int,
    far_limit: float,
) -> Optional[dict]:
    """RANSAC in intersection space for one direction family.

    Two sampled segments intersect to give a hypothesis; the score is the
    length-weighted support of that point over the family, and the best hypothesis
    is refined by weighted least squares. A refined point beyond ``far_limit`` is
    reported as a direction only (``at_infinity``).
    """

    if len(members) < 2:
        return None
    midpoints = np.array([[segments[index]["mid_x"], segments[index]["mid_y"]] for index in members],
                         dtype=np.float64)
    directions = np.array([[segments[index]["dir_x"], segments[index]["dir_y"]] for index in members],
                          dtype=np.float64)
    lengths = np.array([segments[index]["length"] for index in members], dtype=np.float64)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    sin_tolerance = math.sin(math.radians(_VP_TOLERANCE_DEG))
    count = len(members)
    normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1)
    constants = -(normals[:, 0] * midpoints[:, 0] + normals[:, 1] * midpoints[:, 1])
    best: Optional[Tuple[float, np.ndarray]] = None
    for _ in range(int(iters)):
        first, second = rng.choice(count, size=2, replace=False)
        if math.degrees(_angle_distance(float(angles[int(first)]), float(angles[int(second)]))) < _MIN_PAIR_ANGLE_DEG:
            continue
        point = _intersect(midpoints[int(first)], directions[int(first)],
                           midpoints[int(second)], directions[int(second)])
        if point is None or not np.isfinite(point).all():
            continue
        support, _ = _support(point, midpoints, directions, lengths, sin_tolerance)
        if best is None or support > best[0]:
            best = (support, point)
    if best is None:
        point = _refine_point(midpoints, normals, lengths, constants)
        if point is None:
            return None
        best = (0.0, point)
    point = best[1]
    for _ in range(3):
        _, hits = _support(point, midpoints, directions, lengths, sin_tolerance)
        if int(hits.sum()) < 2:
            break
        refined = _refine_point(midpoints[hits], normals[hits], lengths[hits], constants[hits])
        if refined is None or np.allclose(refined, point, atol=1e-6):
            break
        point = refined
    _, hits = _support(point, midpoints, directions, lengths, sin_tolerance)
    inliers = [int(members[index]) for index in np.where(hits)[0]]
    if len(inliers) < _MIN_FAMILY_SEGMENTS:
        return None
    support = float(lengths[hits].sum())
    if not np.isfinite(point).all() or float(np.hypot(point[0], point[1])) > far_limit:
        inside = np.where(hits)[0]
        mean = np.array(
            [
                float((lengths[inside] * directions[inside, 0]).sum()),
                float((lengths[inside] * directions[inside, 1]).sum()),
            ]
        )
        norm = float(np.hypot(mean[0], mean[1]))
        if norm <= 0.0:
            return None
        return {
            "point": None,
            "direction": _canonical(mean / norm),
            "inliers": inliers,
            "support": support,
            "at_infinity": True,
        }
    return {
        "point": np.array([float(point[0]), float(point[1])]),
        "direction": _canonical(np.array([directions[hits, 0].mean(), directions[hits, 1].mean()])),
        "inliers": inliers,
        "support": support,
        "at_infinity": False,
    }


# ------------------------------------------------------------------- horizon


def _line_through(point: Sequence[float], direction: Sequence[float]) -> np.ndarray:
    """Normalized homogeneous line through ``point`` along ``direction``."""

    normal = np.array([-float(direction[1]), float(direction[0])], dtype=np.float64)
    norm = float(np.hypot(normal[0], normal[1]))
    if norm > 0.0:
        normal = normal / norm
    constant = -(normal[0] * float(point[0]) + normal[1] * float(point[1]))
    return np.array([normal[0], normal[1], constant], dtype=np.float64)


def _horizon_line(first: dict, second: dict) -> Optional[np.ndarray]:
    """Line through both vanishing points, or ``None`` when it is undetermined."""

    if first["at_infinity"] and second["at_infinity"]:
        return None
    if not first["at_infinity"] and not second["at_infinity"]:
        x1, y1 = float(first["point"][0]), float(first["point"][1])
        x2, y2 = float(second["point"][0]), float(second["point"][1])
        if math.hypot(x2 - x1, y2 - y1) < 1e-6:
            return None
        line = np.array([y1 - y2, x2 - x1, x1 * y2 - y1 * x2], dtype=np.float64)
    else:
        finite = second if first["at_infinity"] else first
        infinite = first if first["at_infinity"] else second
        line = _line_through(finite["point"], infinite["direction"])
    norm = float(np.hypot(line[0], line[1]))
    if norm <= 0.0:
        return None
    return line / norm


def _horizon_row(line: np.ndarray, columns) -> Any:
    """Row of the horizon line at ``columns`` (scalar or array)."""

    return -(line[0] * columns + line[2]) / line[1]


# ----------------------------------------------------------------- homography


def _ground_homography(line: np.ndarray, first: dict, second: dict,
                       shape: Tuple[int, int]) -> Optional[List[List[float]]]:
    """Metric-free top-down reference frame for the ground plane.

    The horizon is mapped to infinity and the two ground directions onto the
    axes, which rectifies the floor up to an unknown affine scale; the frame is
    then translated and scaled so the image's near edge spans roughly one image
    side. No metric meaning is implied and the returned matrix is 3x3 floats.
    """

    height, width = int(shape[0]), int(shape[1])

    def direction_of(family: dict) -> Optional[np.ndarray]:
        vector = (np.array(family["direction"], dtype=np.float64)
                  if family["at_infinity"] else np.array(family["point"], dtype=np.float64))
        norm = float(np.hypot(vector[0], vector[1]))
        return vector / norm if norm > 0.0 else None

    left = direction_of(first)
    right = direction_of(second)
    if left is None or right is None:
        return None
    determinant = float(left[0] * right[1] - right[0] * left[1])
    if abs(determinant) < 0.15:
        return None
    inverse = np.array(
        [[right[1] / determinant, -right[0] / determinant],
         [-left[1] / determinant, left[0] / determinant]],
        dtype=np.float64,
    )
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    offset = float(line[0] * center_x + line[1] * center_y + line[2])
    if abs(offset) < 1e-3 * math.hypot(height, width):
        return None
    translate = np.array([[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]])
    rectify = np.array(
        [[inverse[0, 0], inverse[0, 1], 0.0],
         [inverse[1, 0], inverse[1, 1], 0.0],
         [line[0], line[1], line[2]]],
        dtype=np.float64,
    )
    rectified = rectify @ translate
    near_x = np.linspace(0.0, float(width - 1), 5)
    near_points = np.stack([near_x, np.full_like(near_x, float(height - 1))], axis=1)
    mapped = cv2.perspectiveTransform(near_points.reshape(-1, 1, 2), rectified).reshape(-1, 2)
    finite = np.isfinite(mapped).all(axis=1)
    if int(finite.sum()) >= 2:
        valid = mapped[finite]
        span = max(float(valid[:, 0].max() - valid[:, 0].min()),
                   float(valid[:, 1].max() - valid[:, 1].min()),
                   1e-6)
        scale = min(float(width), float(height)) / span
        shift_x = width / 2.0 - scale * float(valid[:, 0].min() + valid[:, 0].max()) / 2.0
        shift_y = height / 2.0 - scale * float(valid[:, 1].min() + valid[:, 1].max()) / 2.0
        normalize = np.array([[scale, 0.0, shift_x], [0.0, scale, shift_y], [0.0, 0.0, 1.0]])
        rectified = normalize @ rectified
    if not np.isfinite(rectified).all():
        return None
    try:
        np.linalg.inv(rectified)
    except np.linalg.LinAlgError:
        return None
    return [[float(value) for value in row] for row in rectified]


# --------------------------------------------------------------- floor region


def _gradient_map(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    energy_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    energy_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(energy_x, energy_y)


def _gradient_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edges of the image, used for structural boundaries."""

    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), _CANNY_LOW, _CANNY_HIGH)


def _texture_profile(line: np.ndarray, gradient: np.ndarray,
                     shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Rectified texture energy along horizon-parallel scan lines.

    Each line is sampled at a constant offset from the horizon; the median gradient
    along the line is multiplied by that offset, which cancels the 1/depth
    foreshortening of a ground texture, so a uniform floor gives a flat profile.
    """

    height, width = int(shape[0]), int(shape[1])
    xs = np.linspace(0.02 * (width - 1), 0.98 * (width - 1), _TEXTURE_POINTS)
    bottom_depth = float(line[0] * ((width - 1) / 2.0) + line[1] * (height - 1) + line[2])
    if bottom_depth <= 0.0:
        return None
    fractions = np.linspace(1.0, 0.02, _TEXTURE_SAMPLES)
    energies = np.full(_TEXTURE_SAMPLES, np.nan)
    for index, fraction in enumerate(fractions):
        depth = float(fraction) * bottom_depth
        rows = (depth - line[2] - line[0] * xs) / line[1]
        keep = (rows >= 0.0) & (rows <= height - 1)
        if int(keep.sum()) < max(8, _TEXTURE_POINTS // 4):
            continue
        columns = np.clip(np.rint(xs[keep]).astype(int), 0, width - 1)
        sampled_rows = np.clip(np.rint(rows[keep]).astype(int), 0, height - 1)
        energies[index] = float(np.median(gradient[sampled_rows, columns])) * depth
    return fractions, energies, bottom_depth


def _floor_top_depth(line: np.ndarray, gradient: np.ndarray,
                     shape: Tuple[int, int]) -> Optional[float]:
    """Depth offset at which the floor texture stops matching the near ground."""

    profile = _texture_profile(line, gradient, shape)
    if profile is None:
        return None
    fractions, energies, bottom_depth = profile
    finite = np.isfinite(energies)
    near_slice = slice(0, max(4, int(round(0.35 * _TEXTURE_SAMPLES))))
    near_values = energies[near_slice][finite[near_slice]]
    if near_values.size < 4:
        return None
    reference = float(np.median(near_values))
    if not np.isfinite(reference) or reference <= 0.0:
        return None
    accepted = 0
    for index in range(_TEXTURE_SAMPLES):
        if not finite[index]:
            break
        ratio = float(energies[index]) / reference
        if not (_TEXTURE_LOW <= ratio <= _TEXTURE_HIGH):
            break
        accepted = index
    if accepted <= 0:
        return None
    return float(fractions[accepted]) * bottom_depth


def _wall_base_line(image: np.ndarray, gray: np.ndarray, horizon: np.ndarray,
                    max_slope_deg: float = 1.5) -> Optional[np.ndarray]:
    """Bright, low-saturation stripe where the ground meets a vertical surface.

    The junction is usually painted (a white line along the wall foot) or lit, so the
    detector looks for bright pixels with low saturation in the lower 70% of the frame,
    tolerates a one-pixel wobble, and requires the stripe to cover at least
    ``_WALL_ROW_COVERAGE`` of the row width. The candidate closest to the horizon (but
    below it) wins. Its tilt is fitted from the stripe's own pixels and is clamped to
    ``max_slope_deg``: the camera is mounted level, so the junction is treated as
    horizontal rather than as a skewed line.
    """

    height, width = int(gray.shape[0]), int(gray.shape[1])
    array = np.asarray(image)
    search_top = int(round((1.0 - _WALL_SEARCH_BOTTOM) * height))
    horizon_row = float(_horizon_row(horizon, (width - 1) / 2.0))
    tolerant = None
    candidates = np.empty(0, dtype=int)
    for value_floor, saturation_ceiling in _WALL_BRIGHT_STAGES:
        if array.ndim == 3 and array.shape[2] >= 3:
            hsv = cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2HSV)
            bright = ((hsv[:, :, 2] >= value_floor) & (hsv[:, :, 1] <= saturation_ceiling))
        else:
            bright = _as_gray(array) >= value_floor
        mask = bright.astype(np.uint8)
        mask[:search_top] = 0
        if not mask.any():
            continue
        # A hand-painted line wobbles by a pixel or two, so widen the mask vertically.
        tolerant = cv2.dilate(mask, np.ones((2 * _WALL_JITTER + 1, 1), np.uint8))
        coverage = tolerant.mean(axis=1)
        candidates = np.where(coverage >= _WALL_ROW_COVERAGE)[0]
        candidates = candidates[candidates >= horizon_row + 2]
        if candidates.size:
            break
    if tolerant is None or candidates.size == 0:
        return None
    row = int(candidates.min())
    low = max(0, row - _WALL_JITTER)
    high = min(height, row + _WALL_JITTER + 1)
    rows, columns = np.nonzero(tolerant[low:high])
    if columns.size >= 16:
        rows = rows.astype(np.float64) + low
        slope, intercept = np.polyfit(columns.astype(np.float64), rows, 1)
        slope, intercept = float(slope), float(intercept)
    else:
        slope, intercept = 0.0, float(row)
    if abs(slope) > math.tan(math.radians(float(max_slope_deg))):
        slope = 0.0
        intercept = float(np.median(rows)) if rows.size else float(row)
    return _line_through((0.0, intercept), (1.0, slope))


def _horizontal_horizon_row(segments: Sequence[dict], families: Sequence[dict]) -> Optional[float]:
    """Horizon row from one family's line intersections, when the VP pair is tilted.

    Two vanishing points that both belong to ground directions must sit on the horizon;
    if their connecting line is too steep, the estimate is unusable. The intersections
    inside the more horizon-parallel family all lie at its own vanishing point, so the
    length-weighted median of their rows recovers a level horizon instead of a skewed
    one. Fewer than three intersections means the row is not measurable.
    """

    rows: List[float] = []
    weights: List[float] = []
    # A finite vanishing point of either family lies on the horizon, and it is far
    # better conditioned than the intersections of nearly parallel lines.
    for family in families:
        if family["point"] is None:
            continue
        rows.append(float(family["point"][1]))
        weights.append(float(family["support"]))
    best: Optional[Tuple[float, Sequence[int]]] = None
    for family in families:
        members = list(family["inliers"])
        if len(members) < 3:
            continue
        tilt = float(np.mean([abs(math.sin(segments[index]["angle"])) for index in members]))
        if best is None or tilt < best[0]:
            best = (tilt, members)
    members = list(best[1])[:_MAX_INTERSECTION_MEMBERS] if best is not None else []
    for first in range(len(members)):
        left = segments[members[first]]
        for second in range(first + 1, len(members)):
            right = segments[members[second]]
            point = _intersect(
                np.array([left["mid_x"], left["mid_y"]], dtype=np.float64),
                np.array([left["dir_x"], left["dir_y"]], dtype=np.float64),
                np.array([right["mid_x"], right["mid_y"]], dtype=np.float64),
                np.array([right["dir_x"], right["dir_y"]], dtype=np.float64),
            )
            if point is None or not np.isfinite(point).all():
                continue
            rows.append(float(point[1]))
            weights.append(float(left["length"] * right["length"]))
    if len(rows) < 3:
        return None
    order = np.argsort(np.array(rows))
    ordered_rows = np.array(rows)[order]
    ordered_weights = np.array(weights)[order]
    cumulative = np.cumsum(ordered_weights)
    if float(cumulative[-1]) <= 0.0:
        return None
    index = int(np.searchsorted(cumulative, float(cumulative[-1]) / 2.0))
    return float(ordered_rows[min(index, ordered_rows.size - 1)])


def _floor_region(line: np.ndarray, gradient: np.ndarray,
                  shape: Tuple[int, int],
                  wall_base: Optional[np.ndarray] = None) -> List[List[int]]:
    """Visible floor quad: measured top edge, image border elsewhere.

    Left, right and bottom edges are the image border: the floor is bounded by the
    frame, not by the estimator. The top edge prefers the ground/vertical-surface
    junction, then the horizon-parallel texture scan, then just below the horizon, so
    the region never claims wall surface as floor unless no junction exists at all.
    """

    height, width = int(shape[0]), int(shape[1])
    center_depth = float(line[0] * ((width - 1) / 2.0) + line[1] * (height - 1) + line[2])
    columns = np.array([0.0, float(width - 1)])
    horizon_rows = np.asarray(_horizon_row(line, columns), dtype=np.float64)
    margin = max(2.0, _WALL_MARGIN * height)
    if wall_base is not None:
        rows = np.asarray(_horizon_row(wall_base, columns), dtype=np.float64) - margin
    else:
        top_depth = _floor_top_depth(line, gradient, shape)
        if top_depth is None:
            top_depth = 0.02 * center_depth
        rows = (top_depth - line[2] - line[0] * columns) / line[1]
    minimum_extent = max(2.0, _MIN_FLOOR_EXTENT * height)
    top = []
    for index in range(2):
        limit = float(height - 1) - minimum_extent
        top.append(min(max(float(rows[index]), float(horizon_rows[index]) + 2.0), limit))
    if float(height - 1) - max(top) < minimum_extent:
        return []
    polygon = [
        [0, int(round(top[0]))],
        [width - 1, int(round(top[1]))],
        [width - 1, height - 1],
        [0, height - 1],
    ]
    if polygon[0][1] >= polygon[3][1] or polygon[1][1] >= polygon[2][1]:
        return []
    return polygon


def _floor_contains(columns: np.ndarray, rows: np.ndarray,
                    polygon: Sequence[Sequence[int]]) -> np.ndarray:
    """Vectorized containment test for a floor quad bounded by the image frame."""

    points = np.array([[float(point[0]), float(point[1])] for point in polygon])
    left = min(points[0][0], points[3][0])
    right = max(points[1][0], points[2][0])
    bottom = max(points[2][1], points[3][1])
    span = right - left
    if span <= 0.0:
        return np.zeros(np.shape(columns), dtype=bool)
    ratio = np.clip((columns - left) / span, 0.0, 1.0)
    top = points[0][1] + ratio * (points[1][1] - points[0][1])
    return (columns >= left) & (columns <= right) & (rows >= top) & (rows <= bottom)


# ----------------------------------------------------------- placement patch


def _maximal_rectangle(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Largest all-accepted axis-aligned rectangle in ``mask``.

    Returns ``(top, bottom, left, right)`` as cell indices with exclusive ends, or
    ``None``. Histogram-maximum search, so the cost is linear in the cell count.
    """

    if mask.size == 0 or not mask.any():
        return None
    heights = np.zeros(mask.shape[1], dtype=np.int64)
    best: Optional[Tuple[int, int, int, int]] = None
    best_area = 0
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack: List[Tuple[int, int]] = []
        for column in range(mask.shape[1] + 1):
            current = int(heights[column]) if column < mask.shape[1] else 0
            start = column
            while stack and stack[-1][1] >= current:
                index, height = stack.pop()
                area = height * (column - index)
                if area > best_area:
                    best_area = area
                    best = (row - height + 1, row + 1, index, column)
                start = index
            stack.append((start, current))
    return best


def _cross(origin: Sequence[float], first: Sequence[float], second: Sequence[float]) -> float:
    return ((float(first[0]) - float(origin[0])) * (float(second[1]) - float(first[1]))
            - (float(first[1]) - float(origin[1])) * (float(second[0]) - float(first[0])))


def _segments_cross(first: Sequence[float], second: Sequence[float],
                    third: Sequence[float], fourth: Sequence[float]) -> bool:
    """True when segment ``first->second`` properly intersects ``third->fourth``."""

    def orientation(origin, a, b):
        value = ((float(a[0]) - float(origin[0])) * (float(b[1]) - float(origin[1]))
                 - (float(a[1]) - float(origin[1])) * (float(b[0]) - float(origin[0])))
        return 1 if value > 1e-9 else (-1 if value < -1e-9 else 0)

    left = orientation(first, second, third)
    right = orientation(first, second, fourth)
    up = orientation(third, fourth, first)
    down = orientation(third, fourth, second)
    return left * right < 0 and up * down < 0


def _clockwise_quad(points: Sequence[Sequence[float]]) -> List[List[int]]:
    """Order four projected corners clockwise, starting at the far-left corner.

    Sorting by angle around the centroid turns any convex quad into a simple cyclic
    order, which removes the crossed ("bow tie") shapes a per-axis extreme ordering
    produces. In image coordinates, ascending angle is clockwise on screen.
    """

    values = [(float(point[0]), float(point[1])) for point in points]
    centre_x = sum(value[0] for value in values) / 4.0
    centre_y = sum(value[1] for value in values) / 4.0
    ordered = sorted(values, key=lambda value: math.atan2(value[1] - centre_y, value[0] - centre_x))
    start = min(range(4), key=lambda index: (ordered[index][1], ordered[index][0]))
    ordered = ordered[start:] + ordered[:start]
    return [[int(round(value[0])), int(round(value[1]))] for value in ordered]


def _patch_problems(patch: Sequence[Sequence[int]], floor: Sequence[Sequence[int]],
                    line: np.ndarray, shape: Tuple[int, int]) -> List[str]:
    """Why ``patch`` is not an acceptable placement rectangle (empty when it is).

    The checks mirror the judge's rule so that a patch accepted here is not rejected
    downstream: convex and simple, large enough, not a thin strip, and every corner on
    the floor, below the horizon and inside the frame.
    """

    height, width = int(shape[0]), int(shape[1])
    if len(patch) != 4:
        return ["patch_not_quad"]
    points = [(float(point[0]), float(point[1])) for point in patch]
    signs = set()
    for index in range(4):
        value = _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        if abs(value) <= 1e-9:
            return ["patch_collinear"]
        signs.add(1 if value > 0 else -1)
    if len(signs) != 1:
        return ["patch_not_convex"]
    if (_segments_cross(points[0], points[1], points[2], points[3])
            or _segments_cross(points[1], points[2], points[3], points[0])):
        return ["patch_self_intersecting"]
    area = abs(sum(points[index][0] * points[(index + 1) % 4][1]
                   - points[(index + 1) % 4][0] * points[index][1] for index in range(4))) / 2.0
    if area < max(_PATCH_MIN_AREA, _PATCH_MIN_AREA_RATIO * height * width):
        return ["patch_too_small"]

    def length(left, right):
        return math.hypot(left[0] - right[0], left[1] - right[1])

    far_edge = length(points[0], points[1])
    near_edge = length(points[2], points[3])
    left_edge = length(points[3], points[0])
    right_edge = length(points[1], points[2])
    for first, second in ((far_edge, near_edge), (left_edge, right_edge)):
        if first <= 0.0 or second <= 0.0:
            return ["patch_degenerate"]
        ratio = first / second
        if ratio < _PATCH_SIDE_RATIO_LOW or ratio > _PATCH_SIDE_RATIO_HIGH:
            return ["patch_thin"]
    for point in points:
        if not (0.0 <= point[0] <= width - 1.0 and 0.0 <= point[1] <= height - 1.0):
            return ["patch_outside_frame"]
        if point[1] < float(_horizon_row(line, point[0])):
            return ["patch_above_horizon"]
        if not bool(_floor_contains(np.array([point[0]]), np.array([point[1]]), floor)[0]):
            return ["patch_off_floor"]
    return []


def _maximal_rectangle(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Largest all-accepted axis-aligned rectangle in ``mask``.

    Returns ``(top, bottom, left, right)`` as cell indices with exclusive ends, or
    ``None``. Histogram-maximum search, so the cost is linear in the cell count.
    """

    if mask.size == 0 or not mask.any():
        return None
    heights = np.zeros(mask.shape[1], dtype=np.int64)
    best: Optional[Tuple[int, int, int, int]] = None
    best_area = 0
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack: List[Tuple[int, int]] = []
        for column in range(mask.shape[1] + 1):
            current = int(heights[column]) if column < mask.shape[1] else 0
            start = column
            while stack and stack[-1][1] >= current:
                index, height = stack.pop()
                area = height * (column - index)
                if area > best_area:
                    best_area = area
                    best = (row - height + 1, row + 1, index, column)
                start = index
            stack.append((start, current))
    return best


def _ground_lateral_interval(v: float, rectified: np.ndarray) -> Optional[Tuple[float, float]]:
    """Ground lateral interval visible at rectified depth ``v``.

    The floor quad is projected into the rectified frame, so its four edges give the
    lateral span at every depth by linear interpolation. Returns ``None`` outside the
    quad's depth range.
    """

    quad = np.asarray(rectified, dtype=np.float64)
    depths = quad[:, 1]
    low, high = float(depths.min()), float(depths.max())
    if not (low - 1e-9 <= v <= high + 1e-9):
        return None
    intersections: List[float] = []
    for index in range(4):
        first = quad[index]
        second = quad[(index + 1) % 4]
        span = second[1] - first[1]
        if abs(span) < 1e-12:
            if abs(v - first[1]) < 1e-9:
                intersections.extend([float(first[0]), float(second[0])])
            continue
        ratio = (v - first[1]) / span
        if -1e-9 <= ratio <= 1.0 + 1e-9:
            intersections.append(float(first[0] + ratio * (second[0] - first[0])))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def _pixels_per_ground_unit(inverse: np.ndarray, u: float, v: float,
                            epsilon: float = 1.0) -> float:
    """Image pixels spanned by one ground unit along the lateral axis at ``(u, v)``.

    This is the local Jacobian of the ground-to-image homography, evaluated
    numerically, which is what converts the target's pixel width into the ground width
    the placement rectangle has to reserve.
    """

    sample = np.array([[[u, v]], [[u + epsilon, v]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(sample, inverse).reshape(2, 2)
    if not np.isfinite(mapped).all():
        return 0.0
    return float(math.hypot(mapped[1][0] - mapped[0][0], mapped[1][1] - mapped[0][1])) / epsilon


def _box_fits(image_xy: Sequence[Sequence[float]], shape: Tuple[int, int],
              box_height: float, box_width: float) -> bool:
    """True when a target box standing at ``image_xy`` sits entirely inside the frame."""

    height, width = int(shape[0]), int(shape[1])
    for x, y in image_xy:
        if not (x - box_width / 2.0 >= 0.0 and x + box_width / 2.0 <= width - 1.0):
            return False
        if not (y - box_height >= 0.0 and y <= height - 1.0):
            return False
    return True


def _patch_perspective(patch: Sequence[Sequence[float]],
                       vanishing_points: Sequence[Sequence[float]],
                       depth_span: float,
                       shape: Tuple[int, int],
                       tolerance_deg: float = 5.0) -> str:
    """Why the quad is not a perspective-consistent ground rectangle (empty when it is)."""

    points = [(float(point[0]), float(point[1])) for point in patch]
    ordered = sorted(points, key=lambda value: value[1])
    far_pair = sorted(ordered[:2], key=lambda value: value[0])
    near_pair = sorted(ordered[2:], key=lambda value: value[0])

    def length(left, right):
        return math.hypot(left[0] - right[0], left[1] - right[1])

    far_edge = length(far_pair[0], far_pair[1])
    near_edge = length(near_pair[0], near_pair[1])
    if far_edge <= 0.0 or near_edge <= 0.0:
        return "patch_degenerate"
    if depth_span > 0.05 * float(shape[0]) and near_edge / far_edge < _PATCH_MIN_PERSPECTIVE:
        return "patch_no_perspective"
    if not vanishing_points:
        return "patch_not_ground_aligned"
    best = 180.0
    for candidate in vanishing_points:
        angles = []
        for start, end in ((far_pair[0], near_pair[0]), (far_pair[1], near_pair[1])):
            edge = (end[0] - start[0], end[1] - start[1])
            aim = (float(candidate[0]) - start[0], float(candidate[1]) - start[1])
            denominator = math.hypot(*edge) * math.hypot(*aim)
            if denominator <= 0.0:
                return "patch_degenerate"
            cosine = (edge[0] * aim[0] + edge[1] * aim[1]) / denominator
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, abs(cosine))))))
        best = min(best, max(angles))
    if best > tolerance_deg:
        return "patch_not_ground_aligned"
    return ""


def _floor_depth_point(floor: np.ndarray, horizon: float, height: int,
                       width: int) -> Tuple[float, float]:
    """Where the floor polygon's side edges meet, or a point on the horizon above centre."""

    def line(point_a, point_b):
        (x1, y1), (x2, y2) = point_a, point_b
        return np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)

    count = len(floor)
    left_edge = line(tuple(floor[count - 1]), tuple(floor[0]))
    right_edge = line(tuple(floor[1]), tuple(floor[2]))
    point = np.cross(left_edge, right_edge)
    if abs(point[2]) > 1e-9:
        x, y = float(point[0] / point[2]), float(point[1] / point[2])
        if np.isfinite(x) and np.isfinite(y) and abs(y) <= 50.0 * height and abs(x) <= 50.0 * width:
            return (x, y)
    return (width / 2.0, horizon)


def _row_span(floor: np.ndarray, y: float) -> Tuple[float, float]:
    """Horizontal floor span at one image row, from the polygon edges."""

    xs: List[float] = []
    count = len(floor)
    for index in range(count):
        x1, y1 = float(floor[index][0]), float(floor[index][1])
        x2, y2 = float(floor[(index + 1) % count][0]), float(floor[(index + 1) % count][1])
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if abs(y2 - y1) < 1e-9:
                xs.extend([x1, x2])
            else:
                t = (y - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2:
        return (-1.0, -1.0)
    return (min(xs), max(xs))


def _patch_depth_ok(patch: Sequence[Sequence[int]], shape: Tuple[int, int],
                    options: Mapping[str, Any]) -> bool:
    """Require a minimum vertical patch extent before accepting a fallback."""

    depth = max(float(point[1]) for point in patch) - min(float(point[1]) for point in patch)
    return depth >= float(options.get("min_patch_depth_fraction", 0.10)) * shape[0]


def placement_patch_report(homography: Sequence[Sequence[float]], floor: Sequence[Sequence[int]],
                           line: np.ndarray, shape: Tuple[int, int],
                           options: Dict[str, Any],
                           vanishing_points: Sequence[Sequence[float]] = ()
                           ) -> Tuple[List[List[int]], str]:
    """Ground rectangle for the declared target, expressed as a perspective trapezoid.

    A rectangle lying on the ground plane projects to a trapezoid whose side edges meet
    at the horizon: the horizontal scale at image row ``y`` is proportional to
    ``y - y_h``. The patch is therefore built directly in image space from the horizon
    row and the floor span, without rectification:

    * near edge at the bottom row, far edge at ``y_h + 0.35 * (y_near - y_h)``;
    * half width ``a`` at the near row, shrinking as ``(y - y_h) / (y_near - y_h)``;
    * ``a`` is the largest value whose target box still fits inside the floor span at
      every sampled row, so each foot point in the patch can carry the target.

    The second return value is empty on success, otherwise it names the binding
    constraint (``depth_range_empty``, ``lateral_interval_empty``, ``patch_no_perspective``).
    """

    height, width = int(shape[0]), int(shape[1])
    box_height = float(options["target_height_fraction"]) * height
    box_width = float(options["target_aspect"]) * box_height
    if box_width >= width or box_height >= height:
        return [], "box_taller_than_frame"

    quad = np.array([[float(point[0]), float(point[1])] for point in floor], dtype=np.float64)
    if len(quad) < 3:
        return [], "floor_missing"
    floor_top = float(quad[:, 1].min())
    floor_bottom = float(quad[:, 1].max())

    a_coef, b_coef, c_coef = float(line[0]), float(line[1]), float(line[2])
    horizon = None
    if abs(b_coef) > 1e-9:
        horizon = -(a_coef * (width / 2.0) + c_coef) / b_coef
    if horizon is None or not np.isfinite(horizon) or horizon >= floor_top - 0.02 * height:
        horizon = max(0.0, floor_top - 0.06 * height)

    y_near = min(height - 3.0, floor_bottom - 1.0)
    y_far = horizon + 0.35 * (y_near - horizon)
    if y_far <= floor_top + 2.0 or y_far >= y_near - 0.05 * height:
        return [], "depth_range_empty"
    if y_near - box_height < 0.0:
        return [], "box_does_not_fit"

    span_near = _row_span(quad, y_near)
    if span_near[1] - span_near[0] < box_width * 1.2:
        return [], "lateral_interval_empty"

    def _line(point_a, point_b):
        (x1, y1), (x2, y2) = point_a, point_b
        return np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)

    def _intersect(line_a, line_b):
        point = np.cross(line_a, line_b)
        if abs(point[2]) < 1e-12:
            return None
        return (float(point[0] / point[2]), float(point[1] / point[2]))

    limit = 50.0 * max(height, width)
    on_horizon = [
        (float(point[0]), float(point[1]))
        for point in (vanishing_points or [])
        if len(point) >= 2
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and abs(float(point[0])) <= limit
        and abs(float(point[1])) <= limit
        and abs(float(point[1]) - horizon) <= 0.05 * height
    ]
    # A ground direction always vanishes on the horizon. Points elsewhere are vertical
    # or spurious directions, so they must not shape a ground rectangle.
    finite_points = on_horizon
    if len(finite_points) < 2:
        finite_points = []

    ratio = (y_far - horizon) / (y_near - horizon)
    detail = "patch_no_perspective"
    center_near = (span_near[0] + span_near[1]) / 2.0
    left_seed = span_near[0] + box_width / 2.0
    right_seed = span_near[1] - box_width / 2.0

    if not finite_points:
        # Level-camera fallback: the transverse direction is treated as parallel to the
        # image plane, so near and far edges stay horizontal and the side edges converge
        # on the depth vanishing point taken from the floor's own side edges.
        depth_point = _floor_depth_point(quad, horizon, height, width)
        for _ in range(80):
            near_left = (left_seed, y_near)
            near_right = (right_seed, y_near)
            far_left = (depth_point[0] + ratio * (near_left[0] - depth_point[0]),
                        depth_point[1] + ratio * (near_left[1] - depth_point[1]))
            far_right = (depth_point[0] + ratio * (near_right[0] - depth_point[0]),
                         depth_point[1] + ratio * (near_right[1] - depth_point[1]))
            feasible = True
            for step in np.linspace(0.0, 1.0, 7):
                for near_corner, far_corner in ((near_left, far_left), (near_right, far_right)):
                    foot_x = near_corner[0] + step * (far_corner[0] - near_corner[0])
                    foot_y = near_corner[1] + step * (far_corner[1] - near_corner[1])
                    row_span = _row_span(quad, float(foot_y))
                    if (foot_x - box_width / 2.0 < row_span[0] + 1.0
                            or foot_x + box_width / 2.0 > row_span[1] - 1.0):
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                break
            shrink = 0.04 * (right_seed - left_seed)
            left_seed += shrink
            right_seed -= shrink
            if right_seed - left_seed < box_width:
                return [], "lateral_interval_empty"
        patch = [
            [int(round(near_left[0])), int(round(near_left[1]))],
            [int(round(near_right[0])), int(round(near_right[1]))],
            [int(round(far_right[0])), int(round(far_right[1]))],
            [int(round(far_left[0])), int(round(far_left[1]))],
        ]
        problems = _patch_problems(patch, floor, line, shape)
        if problems:
            return [], problems[0]
        if polygon_area_ratio(patch, (height, width)) > _PATCH_MAX_AREA_RATIO:
            return [], "patch_too_large"
        return patch, ""

    for depth_point in finite_points:
        transverse_point = next((p for p in finite_points if p != depth_point), None)
        if transverse_point is None:
            continue
        left = left_seed
        right = right_seed
        for _ in range(80):
            near_left = (left, y_near)
            near_right = (right, y_near)
            # The far edge must converge on the transverse vanishing point, so the far
            # corners are the intersections of the side rays with one transverse ray.
            far_line = _line(transverse_point, (center_near, y_far))
            far_left = _intersect(_line(depth_point, near_left), far_line)
            far_right = _intersect(_line(depth_point, near_right), far_line)
            if far_left is None or far_right is None:
                break
            feasible = True
            for step in np.linspace(0.0, 1.0, 7):
                for side in (near_left, near_right):
                    foot_x = side[0] + step * ((far_left[0] if side is near_left else far_right[0]) - side[0])
                    foot_y = side[1] + step * ((far_left[1] if side is near_left else far_right[1]) - side[1])
                    row_span = _row_span(quad, float(foot_y))
                    if foot_x - box_width / 2.0 < row_span[0] + 1.0 or foot_x + box_width / 2.0 > row_span[1] - 1.0:
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                break
            shrink = 0.04 * (right - left)
            left += shrink
            right -= shrink
            if right - left < box_width:
                feasible = False
                break
        if not feasible:
            detail = "lateral_interval_empty"
            continue
        patch = [
            [int(round(near_left[0])), int(round(near_left[1]))],
            [int(round(near_right[0])), int(round(near_right[1]))],
            [int(round(far_right[0])), int(round(far_right[1]))],
            [int(round(far_left[0])), int(round(far_left[1]))],
        ]
        problems = _patch_problems(patch, floor, line, shape)
        if problems:
            detail = problems[0]
            continue
        if polygon_area_ratio(patch, (height, width)) > _PATCH_MAX_AREA_RATIO:
            detail = "patch_too_large"
            continue
        return patch, ""
    return [], detail


def _placement_patch(homography: Sequence[Sequence[float]], floor: Sequence[Sequence[int]],
                     line: np.ndarray, shape: Tuple[int, int],
                     options: Dict[str, Any],
                     vanishing_points: Sequence[Sequence[float]] = ()) -> List[List[int]]:
    """Ground-plane placement rectangle (see :func:`placement_patch_report`)."""

    patch, _ = placement_patch_report(homography, floor, line, shape, options, vanishing_points)
    return patch


def _placement_patch_with_fallback(homography, floor, line, shape, options, finite_points):
    """Try the VP-shaped patch, then the level-camera patch.

    A missing homography or a thin VP patch does not prove that no target can be
    placed. The level-camera path keeps near/far edges horizontal and converges
    the side edges on the floor's own depth point; it ignores the homography, so
    it can run with ``None``. The caller must still accept the returned patch
    through the normal geometry checks.
    """

    if homography is not None and finite_points:
        patch, detail = placement_patch_report(
            homography, floor, line, shape, options, finite_points
        )
        if patch:
            return patch, detail, "vanishing_points"
    patch, detail = placement_patch_report(None, floor, line, shape, options, ())
    if patch:
        return patch, detail, "level_camera"
    return [], detail, None


def _texture_floor_fallback(image_bgr: np.ndarray, shape: Tuple[int, int],
                            horizon_y: float, options: Dict[str, Any]):
    """Rebuild a floor/patch candidate from the Gabor texture boundary."""

    texture = texture_floor_top(
        image_bgr,
        horizon_y=horizon_y,
        params={
            "max_width": 640,
            "min_score": 1.0,
            "min_floor_fraction": 0.06,
        },
    )
    if texture.get("row") is None:
        return None
    if float(texture.get("confidence", 0.0)) < float(
        options.get("texture_fallback_confidence_min", 0.40)
    ):
        return None
    height, width = int(shape[0]), int(shape[1])
    row = int(texture["row"])
    if row <= 0 or row >= height - 2:
        return None
    # A texture boundary far above the line-based horizon is more likely a wall,
    # ceiling or reflection edge than the floor. Keep the fallback conservative:
    # it may raise the floor top slightly above the horizon, not replace the whole
    # scene's horizon from far above it.
    if row < float(horizon_y) - 0.05 * height:
        return None
    margin = max(2, int(round(float(options.get("texture_top_margin_fraction", 0.01)) * height)))
    texture_horizon = max(0.0, float(row - margin))
    if texture_horizon > float(options.get("texture_max_horizon_fraction", 0.85)) * height:
        return None
    line = np.array([0.0, 1.0, -texture_horizon], dtype=np.float64)
    floor = [[0, row], [width - 1, row], [width - 1, height - 1], [0, height - 1]]
    patch, detail, mode = _placement_patch_with_fallback(
        None, floor, line, shape, options, ()
    )
    if not patch:
        return None
    if not _patch_depth_ok(patch, (height, width), options):
        return None
    return {
        "patch": patch,
        "line": line,
        "floor": floor,
        "horizon_y": texture_horizon,
        "texture": texture,
        "detail": detail,
        "mode": mode,
    }


# ------------------------------------------------------------- public surface


def estimate_ground_plane(
    image_bgr: np.ndarray,
    *,
    params: Optional[dict] = None,
) -> GroundPlaneObservation:
    """Estimate the ground plane, the visible floor and the placement patch.

    Pipeline: line segments (``cv2.createLineSegmentDetector``, ``HoughLinesP`` when
    LSD is unavailable) are clustered into two directions; each family gets a RANSAC
    vanishing point; the horizon is the line through both points, and the floor's top
    edge comes from a horizon-parallel texture scan. The placement patch is then
    enumerated on the rectified ground grid.

    ``params`` accepts exactly the keys of :data:`DEFAULT_PARAMS`; anything else
    raises ``ValueError``. On success the reason codes are ``ground_visible``,
    ``ground_polygon_fit`` and ``horizon_estimable``, plus ``placement_patch_fit``
    when a patch was found or ``geometry_rejected`` when it was not (a plane can be
    valid for a frame with no room for a target). The estimate is rejected with
    ``ok=False`` and ``geometry_rejected`` when no two-direction ground structure
    exists, when both vanishing points are at infinity, when the horizon cannot be
    reduced to a row, or when the floor region comes out empty.

    Known limits: a frame whose dominant lines are vertical structures rather than
    ground directions can yield a plausible but wrong plane; a horizon through the
    image centre makes the rectification degenerate and is rejected; nothing here is
    metric, so no real distance or camera height is recovered.
    """

    options = _resolve_params(params)
    gray = _as_gray(image_bgr)
    height, width = gray.shape[:2]
    if min(height, width) < MIN_IMAGE_SIDE:
        return GroundPlaneObservation(ok=False, reason_codes=["geometry_rejected"])
    segments = _segments(gray)
    total_length = float(sum(segment["length"] for segment in segments))
    if len(segments) < 2 * _MIN_FAMILY_SEGMENTS or total_length <= 0.0:
        return GroundPlaneObservation(ok=False, reason_codes=["geometry_rejected"])
    groups = _cluster_directions(segments)
    if len(groups) != 2:
        return GroundPlaneObservation(ok=False, reason_codes=["geometry_rejected"])
    diagonal = math.hypot(height, width)
    rng = np.random.default_rng(int(options["seed"]))
    families = []
    for members in groups:
        family = _vanishing_point(segments, members, rng, int(options["ransac_iters"]),
                                  _VP_FAR_LIMIT * diagonal)
        if family is None or family["support"] < _MIN_FAMILY_SHARE * total_length:
            return GroundPlaneObservation(ok=False, reason_codes=["geometry_rejected"])
        families.append(family)
    share = sum(family["support"] for family in families) / total_length
    residual = float(min(1.0, max(0.0, 1.0 - share)))
    if residual > 1.0 - _MIN_COMBINED_SHARE:
        return GroundPlaneObservation(ok=False, residual=residual,
                                      reason_codes=["geometry_rejected"])
    line = _horizon_line(families[0], families[1])
    if line is None or abs(float(line[1])) < _MIN_HORIZON_COS:
        return GroundPlaneObservation(ok=False, residual=residual,
                                      reason_codes=["geometry_rejected"])
    center_x = (width - 1) / 2.0
    # The camera is mounted level, so the horizon is level too. A steep line through the
    # two vanishing points means the pair is unreliable: fall back to the level row given
    # by the transverse family's intersections instead of reporting a skewed horizon.
    forced = False
    slope = abs(float(-line[0] / line[1]))
    if slope > math.tan(math.radians(float(options["horizon_max_slope_deg"]))):
        row = _horizontal_horizon_row(segments, families)
        if row is None or not (-0.02 * height <= row <= 1.02 * height):
            return GroundPlaneObservation(ok=False, residual=residual,
                                          reason_codes=["geometry_rejected"])
        line = np.array([0.0, 1.0, -float(row)], dtype=np.float64)
        forced = True
    raw_horizon = float(_horizon_row(line, center_x))
    if not (-0.02 * height <= raw_horizon <= 1.02 * height):
        return GroundPlaneObservation(ok=False, residual=residual,
                                      reason_codes=["geometry_rejected"])
    horizon_y = float(np.clip(raw_horizon, 0.0, float(height - 1)))
    homography = _ground_homography(line, families[0], families[1], (height, width))
    wall_base = _wall_base_line(image_bgr, gray, line,
                                float(options["horizon_max_slope_deg"]))
    floor = _floor_region(line, _gradient_map(gray), (height, width), wall_base)
    if len(floor) != 4:
        return GroundPlaneObservation(ok=False, horizon_y=horizon_y, residual=residual,
                                      reason_codes=["geometry_rejected"])
    finite_points = [np.array(family["point"], dtype=np.float64)
                     for family in families if not family["at_infinity"]]
    patch = []
    patch_mode = None
    if floor:
        patch, _, patch_mode = _placement_patch_with_fallback(
            homography, floor, line, (height, width), options, finite_points
        )
    texture_fallback = None
    if not patch and floor:
        texture_fallback = _texture_floor_fallback(
            image_bgr, (height, width), horizon_y, options
        )
        if texture_fallback is not None:
            patch = texture_fallback["patch"]
            line = texture_fallback["line"]
            floor = texture_fallback["floor"]
            horizon_y = float(texture_fallback["horizon_y"])
            patch_mode = "texture_fallback"
            homography = None
    points = [[float(point[0]), float(point[1])] for point in finite_points]
    points.sort(key=lambda point: point[0])
    reasons = ["ground_visible", "ground_polygon_fit", "horizon_estimable"]
    if forced:
        reasons.append("horizon_forced_horizontal")
    if texture_fallback is not None:
        reasons.append("horizon_texture_fallback")
    if patch_mode == "level_camera":
        reasons.append("placement_patch_level_camera")
    elif patch_mode == "texture_fallback":
        reasons.append("placement_patch_texture_fallback")
    reasons.append("placement_patch_fit" if patch else "geometry_rejected")
    return GroundPlaneObservation(
        ok=True,
        horizon_y=horizon_y,
        vanishing_points=points,
        homography=homography,
        floor_region=floor,
        placement_patch=patch,
        residual=residual,
        reason_codes=reasons,
    )


def estimate_ground_region(
    image_bgr: np.ndarray,
    *,
    params: Optional[dict] = None,
) -> GroundObservation:
    """Compatibility entry: the plane observation projected onto the older record.

    ``polygon`` carries ``floor_region``, ``foot_xy`` is the midpoint of the
    placement patch's bottom edge and ``residual`` keeps the vanishing-point inlier
    deficit, so existing consumers (judge thresholds, overlays, metrics) read the
    same fields without a second set of threshold semantics.
    """

    plane = estimate_ground_plane(image_bgr, params=params)
    patch = plane.placement_patch if plane.ok else []
    return GroundObservation(
        ok=bool(plane.ok),
        polygon=[list(point) for point in plane.floor_region] if plane.ok else [],
        foot_xy=foot_from_polygon(patch) if patch else None,
        horizon_y=plane.horizon_y,
        residual=plane.residual,
        reason_codes=list(plane.reason_codes),
        placement_patch=[list(point) for point in patch],
    )


def foot_from_polygon(
    polygon: Sequence[Sequence[int]],
    x_fraction: float = 0.5,
) -> Optional[List[int]]:
    """Foot point on the bottom edge of a ``TL, TR, BR, BL`` polygon."""

    if polygon is None:
        return None
    try:
        points = [[float(value) for value in point[:2]] for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) < 4:
        return None
    fraction = float(x_fraction)
    if not math.isfinite(fraction):
        return None
    fraction = min(1.0, max(0.0, fraction))
    bottom_left = points[3]
    bottom_right = points[2]
    x = bottom_left[0] + fraction * (bottom_right[0] - bottom_left[0])
    y = bottom_left[1] + fraction * (bottom_right[1] - bottom_left[1])
    return [int(round(x)), int(round(y))]


def polygon_area_ratio(polygon: Sequence[Sequence[int]], shape: Tuple[int, int]) -> float:
    """Shoelace area of a polygon as a fraction of the image area, clamped to 1."""

    height = int(shape[0])
    width = int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape must be (height, width) with positive values")
    if polygon is None:
        return 0.0
    try:
        points = [[float(value) for value in point[:2]] for point in polygon]
    except (TypeError, ValueError, IndexError):
        return 0.0
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += point[0] * following[1] - following[0] * point[1]
    area = abs(total) / 2.0
    return float(min(1.0, area / float(height * width)))

def ground_area_ratio(floor: Sequence[Sequence[int]], shape: Tuple[int, int]) -> float:
    """Floor area as a share of the frame."""

    return polygon_area_ratio(floor, shape)


def ground_clearance(image_bgr: np.ndarray, patch: Sequence[Sequence[int]], *,
                     params: Optional[dict] = None) -> Dict[str, Any]:
    """How much of the placement patch is empty floor.

    Pure image measurement: a local edge-density map (box filter over Canny edges) marks
    textured pixels as obstacles; the share they cover is the clutter. No annotation is
    consulted, so an unlabelled e-bike still counts.

    Returns ``free_ratio``, ``obstacle_ratio``, ``components`` (obstacle blobs of at least
    ``clearance_min_component_ratio`` of the frame), ``patch_area_ratio`` and ``method``.
    """

    options = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError("unknown clearance parameters: %s" % ", ".join(sorted(unknown)))
        options.update(params)

    height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if len(patch) < 3:
        return {"free_ratio": 0.0, "obstacle_ratio": 1.0, "components": 0,
                "patch_area_ratio": 0.0, "method": "patch_missing"}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    window = int(options["clearance_win_px"])
    density = cv2.boxFilter(edges, -1, (window, window), normalize=True)
    patch_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(patch_mask, [np.asarray(patch, dtype=np.int32).reshape(-1, 1, 2)], 255)
    patch_pixels = int(np.count_nonzero(patch_mask))
    if patch_pixels == 0:
        return {"free_ratio": 0.0, "obstacle_ratio": 1.0, "components": 0,
                "patch_area_ratio": 0.0, "method": "empty_mask"}
    obstacle = ((density >= float(options["clearance_edge_density_max"])) & (patch_mask > 0)).astype(np.uint8)
    obstacle_pixels = int(np.count_nonzero(obstacle))
    count, _, stats, _ = cv2.connectedComponentsWithStats(obstacle, connectivity=4)
    minimum = float(options["clearance_min_component_ratio"]) * height * width
    components = sum(1 for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= minimum)
    free_ratio = 1.0 - obstacle_pixels / float(patch_pixels)
    return {
        "free_ratio": round(max(0.0, min(1.0, free_ratio)), 4),
        "obstacle_ratio": round(min(1.0, obstacle_pixels / float(patch_pixels)), 4),
        "components": int(components),
        "patch_area_ratio": round(polygon_area_ratio(patch, (height, width)), 4),
        "method": "canny+boxfilter(win=%d, density>=%.3f), 4-connect, area>=%.4f*frame"
                  % (window, float(options["clearance_edge_density_max"]),
                     float(options["clearance_min_component_ratio"])),
    }
