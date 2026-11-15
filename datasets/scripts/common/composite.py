"""Image compositing and visibility primitives shared by dataset tasks."""
import math
import cv2
import numpy as np
from . import images as assets
from .perspective import planar_warp


def alpha_place(canvas, rgba, center_x, foot_y):
    """Composite an entirely in-frame sprite and return its binary occupancy."""
    height, width = canvas.shape[:2]
    sh, sw = rgba.shape[:2]
    left, top = round(center_x - sw / 2), round(foot_y - sh)
    right, bottom = left + sw, top + sh
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError('sprite outside frame')
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    canvas[top:bottom, left:right] = np.clip(
        rgba[:, :, :3] * alpha + canvas[top:bottom, left:right] * (1 - alpha), 0, 255).astype(np.uint8)
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[top:bottom, left:right] = rgba[:, :, 3] > 127
    if occupancy.sum() < 100:
        raise ValueError('sprite too small')
    return occupancy


def visible_masks(masks):
    """Input masks are ordered back to front; return same-order visible masks."""
    if not masks:
        return []
    if any(mask.shape != masks[0].shape for mask in masks):
        raise ValueError('mask shape mismatch')
    front = np.zeros(masks[0].shape, dtype=bool)
    result = [None] * len(masks)
    for i in range(len(masks) - 1, -1, -1):
        mask = np.asarray(masks[i], dtype=bool)
        result[i] = mask & ~front
        front |= mask
    return result


def truncated_gauss(rng, mean, std, low, high):
    """Sample inside bounds by rejection, never clamping values to the edges."""
    if not (math.isfinite(mean) and math.isfinite(std) and math.isfinite(low)
            and math.isfinite(high) and std > 0 and low < high and low <= mean <= high):
        raise ValueError('invalid truncated Gaussian parameters')
    for _ in range(100):
        value = rng.gauss(mean, std)
        if low <= value <= high:
            return value
    raise RuntimeError('truncated Gaussian sampling exceeded retry limit')


def random_actor(rng, kind, category, image, source, foot_x, foot_y, person_height):
    angles = {'yaw': truncated_gauss(rng, 0, 5, -12, 12),
              'pitch': truncated_gauss(rng, 0, 2.5, -6, 6),
              'roll': truncated_gauss(rng, 0, 1, -3, 3)}
    projected = planar_warp(image, **angles)
    if kind in ('person', 'person_back'):
        target_h = round(person_height)
        sprite = assets.resized(projected, height=target_h)
    elif category == 'upright':
        target_h = round(person_height * rng.uniform(.68, .87))
        sprite = assets.resized(projected, height=target_h)
    else:
        target_w = round(person_height * rng.uniform(.85, 1.12))
        sprite = assets.resized(projected, width=target_w)
    relative_exposure = truncated_gauss(rng, 1.0, .035, .90, 1.08)
    sprite = sprite.copy()
    sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.float32) * relative_exposure, 0, 255).astype(np.uint8)
    alpha = sprite[:, :, 3] > 127
    if not alpha.any() or float(cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_BGR2GRAY)[alpha].mean()) < 30:
        raise ValueError('foreground is too dark')
    return {'kind': kind, 'category': category, 'source': source, 'sprite': sprite,
            'foot_x': float(foot_x), 'foot_y': float(foot_y), 'angles_degrees': angles,
            'relative_exposure': relative_exposure, 'person_reference_height_px': person_height,
            'sprite_width_px': int(sprite.shape[1]), 'sprite_height_px': int(sprite.shape[0])}


def draw_shadow(canvas, actor):
    sh, sw = actor['sprite'].shape[:2]
    layer = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(layer, (round(actor['foot_x']), round(actor['foot_y'])),
                (max(3, round(sw*.28)), max(2, round(sh*.018))), 0, 0, 360, 255, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(2, sh*.015)).astype(np.float32) / 255
    factor = (1 - .22*layer)[:, :, None]
    canvas[:] = (canvas.astype(np.float32) * factor).astype(np.uint8)

