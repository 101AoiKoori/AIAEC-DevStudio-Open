"""Project ground contacts and estimate candidate regions for offline synthesis."""

from dataclasses import dataclass

import cv2
import numpy as np


def project(camera, point, *, image_size=(640, 480), sensor_width_mm=36.0):
    """Project a world point for a level-roll Blender camera with horizontal sensor fit."""
    eye = np.asarray(camera["camera_location_m"], dtype=float)
    forward = np.asarray(camera["camera_target_m"], dtype=float) - eye
    forward_length = np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right_length = np.linalg.norm(right)
    if not np.isfinite(forward_length) or forward_length == 0 or right_length == 0:
        raise ValueError("Camera must point in a finite, nonvertical direction")
    forward /= forward_length
    right /= right_length
    up = np.cross(right, forward)
    offset = np.asarray(point, dtype=float) - eye
    depth = float(np.dot(offset, forward))
    if not np.isfinite(depth) or depth <= 0.05:
        raise ValueError("World point lies at or behind the camera's near plane")
    width, height = image_size
    focal_px = width * float(camera["camera_lens_mm"]) / sensor_width_mm
    return np.array((
        width / 2 + focal_px * np.dot(offset, right) / depth,
        height / 2 - focal_px * np.dot(offset, up) / depth,
    ))


def project_blender_ground(camera, ground_xy, **projection_options):
    """Project one ground contact in metres using the declared Blender floor height."""
    x, y = ground_xy
    return project(camera, (x, y, camera["floor_top_m"]), **projection_options)


def fit_ground_homography(ground_xy, image_xy):
    """Fit a fixed-view plane from four corresponding corners in convex cyclic order.

    Supply ground coordinates in metres and pixel coordinates at the current
    resolution. Use a separate fit for each camera pose and image transform.
    """
    corners = []
    for points in (ground_xy, image_xy):
        quad = np.asarray(points, dtype=np.float64)
        if (quad.shape != (4, 2) or not np.isfinite(quad).all()
                or not cv2.isContourConvex(quad.astype(np.float32))):
            raise ValueError("Calibration needs four distinct convex corners in cyclic order")
        corners.append(quad)
    matrix, _ = cv2.findHomography(*corners, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Calibration points do not determine an invertible homography")
    return matrix


def project_calibrated_ground(matrix, ground_xy):
    """Project one or more ground contacts, preserving the input array shape.

    Interpolate within the observed control region when choosing placement.
    The plane fit supplies neither obstacle masks nor out-of-region validation.
    """
    points = np.asarray(ground_xy, dtype=float)
    if points.ndim not in (1, 2) or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Ground contacts must be finite (x, y) pairs")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Ground projection needs a finite 3×3 homography")
    flat = points.reshape(-1, 2)
    inputs = np.column_stack((flat, np.ones(len(flat))))
    homogeneous = inputs @ matrix.T
    depth = homogeneous[:, 2]
    tolerance = 1e-10 * (np.abs(inputs) @ np.abs(matrix[2]))
    if not np.isfinite(homogeneous).all() or np.any(np.abs(depth) <= tolerance):
        raise ValueError("Ground contact has no finite pixel projection")
    return (homogeneous[:, :2] / depth[:, None]).reshape(points.shape)


@dataclass(frozen=True)
class GroundRegionEstimate:
    """Image-space placement candidate without semantic or obstacle clearance."""

    polygon_px: tuple[tuple[int, int], ...]
    method: str
    support_segments_px: tuple[tuple[int, int, int, int], ...]


def _x_at_y(segment, y):
    x_top, y_top, x_bottom, y_bottom = segment
    return x_top + (x_bottom - x_top) * (y - y_top) / (y_bottom - y_top)


def _region(top, bottom, left_top, right_top, left_bottom, right_bottom, width, method, segments):
    horizontal = np.clip((left_top, right_top, left_bottom, right_bottom), width * .06, width * .94)
    left_top, right_top, left_bottom, right_bottom = np.rint(horizontal).astype(int)
    if min(right_top - left_top, right_bottom - left_bottom) < width * .18:
        return None
    polygon = ((int(left_top), top), (int(right_top), top),
               (int(right_bottom), bottom), (int(left_bottom), bottom))
    return GroundRegionEstimate(polygon, method, segments)


def _converging_region(diagonals, width, height):
    for left in diagonals:
        if left[2] >= left[0]:
            continue
        for right in diagonals:
            if right[2] <= right[0]:
                continue
            top = max(left[1], right[1], int(height * .50))
            bottom = min(left[3], right[3], int(height * .92))
            if bottom - top < height * .20:
                continue
            top += round((bottom - top) * .22)
            x_left, x_right = _x_at_y(left, top), _x_at_y(right, top)
            bottom_left, bottom_right = _x_at_y(left, bottom), _x_at_y(right, bottom)
            width_top = x_right - x_left
            width_bottom = bottom_right - bottom_left
            if width_top < width * .18 or width_bottom < width_top * 1.12:
                continue
            if not width * .30 <= (bottom_left + bottom_right) / 2 <= width * .70:
                continue
            inset = width * .04
            result = _region(top, bottom, x_left + inset, x_right - inset,
                             bottom_left + inset, bottom_right - inset, width,
                             "converging_lower_edges", (left, right))
            if result is not None:
                return result
    return None


def _wall_foot_region(horizontals, diagonals, width, height):
    # Require a wide aligned edge and a second, lower, outward-running cue.
    x_center = width / 2
    for first in horizontals:
        x1, y1, x2, y2 = first
        slope = (y2 - y1) / (x2 - x1)
        y_center = y1 + slope * (x_center - x1)
        if not height * .30 <= y_center <= height * .64:
            continue
        support = tuple(
            segment for segment in horizontals
            if max(abs(y - (y1 + slope * (x - x1)))
                   for x, y in (segment[:2], segment[2:])) <= height * .025
        )
        covered = np.zeros(width, dtype=bool)
        for x_start, _, x_end, _ in support:
            covered[x_start:x_end + 1] = True
        if covered.sum() < width * .58:
            continue
        y_limit = max(max(segment[1], segment[3]) for segment in support)
        for cue in diagonals:
            cue_x, cue_y, near_x, near_y = cue
            moves_left = near_x < cue_x < x_center
            moves_right = near_x > cue_x > x_center
            if cue_y < y_limit + height * .06 or not (moves_left or moves_right):
                continue
            top = max(cue_y, int(y_limit + height * .20))
            bottom = min(near_y, int(height * .90))
            if bottom - top < height * .13:
                continue
            left, right = width * .20, width * .80
            if moves_left:
                left = max(left, cue_x + width * .08)
            else:
                right = min(right, cue_x - width * .08)
            result = _region(top, bottom, left, right, left, right, width,
                             "wide_edge_with_lower_diagonal", (*support, cue))
            if result is not None:
                return result
    return None


def estimate_ground_region(bgr):
    """Estimate a conservative candidate from lower-image edges, or return None.

    Input is an OpenCV uint8 BGR image. This cannot identify the ground by
    appearance, recover its scale, or tell whether an object covers it.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Ground estimation expects a uint8 BGR image")
    height, width = bgr.shape[:2]
    if min(height, width) < 64:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 115)
    edges[:int(height * .28)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 36, minLineLength=round(width * .22),
                            maxLineGap=round(width * .04))
    if lines is None:
        return None
    segments = sorted(
        (tuple(map(int, segment)) for segment in lines[:, 0]),
        key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]), reverse=True,
    )
    horizontals, diagonals = [], []
    for x1, y1, x2, y2 in segments:
        if abs(y2 - y1) <= .16 * abs(x2 - x1):
            horizontals.append((x1, y1, x2, y2) if x1 < x2 else (x2, y2, x1, y1))
        if abs(y2 - y1) < max(height * .13, .30 * abs(x2 - x1)):
            continue
        if abs(x2 - x1) >= width * .08 and max(y1, y2) >= height * .82:
            diagonals.append((x1, y1, x2, y2) if y1 < y2 else (x2, y2, x1, y1))
    return (_converging_region(diagonals, width, height)
            or _wall_foot_region(horizontals, diagonals, width, height))
