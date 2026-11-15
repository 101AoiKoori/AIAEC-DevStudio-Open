"""Shared plane projection and perspective warp primitives for synthesis."""
import math

import cv2
import numpy as np

from lab.training_common.ground_placement import (
    fit_ground_homography,
    project_calibrated_ground,
)


def project_corners(width, height, yaw, pitch, roll):
    """Project a flat image with bounded yaw, pitch and roll."""
    if width < 2 or height < 2 or abs(yaw) > 15 or abs(pitch) > 8 or abs(roll) > 4:
        raise ValueError('invalid image or projection angles')
    yr, pr, rr = map(math.radians, (yaw, pitch, roll))
    result = []
    for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        depth = 1.0 + 0.35 * (x * math.sin(yr) + y * math.sin(pr))
        px, py = x * math.cos(yr) / depth, y * math.cos(pr) / depth
        px, py = px * width, py * height
        result.append((px * math.cos(rr) - py * math.sin(rr),
                       px * math.sin(rr) + py * math.cos(rr)))
    points = np.asarray(result, dtype=np.float32)
    first, second = points[1] - points[0], points[2] - points[1]
    if first[0] * second[1] - first[1] * second[0] <= 0:
        raise ValueError('projection reflects or folds the foreground')
    return points


def planar_warp(image, yaw, pitch, roll):
    """Warp an RGBA sprite and crop to its visible alpha bounds."""
    height, width = image.shape[:2]
    source = np.float32(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)))
    destination = project_corners(width, height, yaw, pitch, roll)
    minimum, maximum = destination.min(axis=0), destination.max(axis=0)
    destination -= minimum
    out_width, out_height = (int(math.ceil(x)) + 2 for x in maximum - minimum)
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image, matrix, (out_width, out_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    y, x = np.nonzero(warped[:, :, 3] > 8)
    if not len(x):
        raise ValueError('empty projected sprite')
    return warped[y.min():y.max() + 1, x.min():x.max() + 1]


def warp_patch(patch, destination_quad, output_size):
    """Warp a rectangular patch and its visible mask into an image.

    ``destination_quad`` lists four corners in cyclic order; ``output_size``
    is ``(width, height)``. Return the warped image, uint8 mask, half-open
    bounding box and homography. This does not decide semantic placement or
    blend the warped patch into a particular task's background.
    """
    image = np.asarray(patch)
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 2:
        raise ValueError("Patch must be a two-dimensional image of at least 2x2")
    quad = np.asarray(destination_quad, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        raise ValueError("Destination must contain four finite convex corners")
    if len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size):
        raise ValueError("Output size must contain positive integer width and height")
    h, w = image.shape[:2]
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(source, quad)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Destination does not define an invertible perspective transform")
    warped = cv2.warpPerspective(image, matrix, output_size)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, output_size)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Warped patch is outside the output image")
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return warped, mask, box, matrix


__all__ = ["fit_ground_homography", "project_calibrated_ground", "project_corners", "planar_warp", "warp_patch"]
