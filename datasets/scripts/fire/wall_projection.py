"""Project rectangular fire papers through the recorded facade homographies."""
import cv2
import numpy as np


def wall_transform(surface):
    width, height = surface['rectified_size']
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32(surface['reference_quad_px'])
    matrix = cv2.getPerspectiveTransform(corners, destination)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError(f'Invalid wall homography: {surface["wall_id"]}')
    return matrix


def transform_points(points, matrix):
    return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), matrix)[0]


def place_on_wall(surface, anchor, fraction, facade_height, jitter_fraction, aspect, margin, rng):
    """Return a complete planar paper, or None when this bounded attempt does not fit."""
    matrix = wall_transform(surface)
    inverse = np.linalg.inv(matrix)
    rectified = transform_points(anchor, inverse)
    center = rectified.mean(axis=0)
    extent = np.asarray(surface['rectified_size'], dtype=np.float32)
    # Keep the previous accepted image-size reference at the plane's calibration centre.
    endpoints = transform_points([extent / 2 - [0, .5], extent / 2 + [0, .5]], matrix)
    vertical_scale = float(np.linalg.norm(endpoints[1]-endpoints[0]))
    reference_height = facade_height / vertical_scale
    paper_height = reference_height * fraction
    half = np.array([paper_height * aspect, paper_height], dtype=np.float32) / 2
    jitter = reference_height * jitter_fraction
    center += [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)]
    boundary = np.float32(surface['boundary_px'])
    rectified_boundary = transform_points(boundary, inverse)
    padding = max(margin / vertical_scale, float(min(extent)) * .01)
    low = rectified_boundary.min(axis=0) + half + padding
    high = rectified_boundary.max(axis=0) - half - padding
    if np.any(low >= high):
        return None
    center = np.clip(center, low, high)
    paper = center + np.array([[-half[0], -half[1]], [half[0], -half[1]],
                               [half[0], half[1]], [-half[0], half[1]]], dtype=np.float32)
    quad = transform_points(paper, matrix)
    if (not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or
            any(cv2.pointPolygonTest(boundary, (float(x), float(y)), True) < margin for x, y in quad)):
        return None
    anchor_height = (np.linalg.norm(rectified[3]-rectified[0]) +
                     np.linalg.norm(rectified[2]-rectified[1])) / 2
    return {'quad_source_px': quad.tolist(), 'quad_wall_xy': paper.tolist(),
            'wall_id': surface['wall_id'], 'wall_homography': matrix.tolist(),
            'wall_boundary_px': boundary.tolist(), 'height_fraction': fraction,
            'paper_aspect_ratio': aspect,
            'reference_facade_height_wall_units': reference_height,
            'scale_from_reference': float(paper_height / anchor_height)}
