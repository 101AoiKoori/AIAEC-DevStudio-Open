"""Pure helpers for Blender fire-scene scheduling and visible-instance labels."""
import numpy as np


def srgb_to_linear(rgb):
    """Convert PNG sRGB encoded floats to linear light for emission ID decode."""
    data = np.asarray(rgb, dtype=np.float32)
    if not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1):
        raise ValueError('sRGB input must contain finite floats between zero and one')
    return np.where(data <= .04045, data / 12.92, ((data + .055) / 1.055) ** 2.4)


def scenario(index):
    """Cycle through a balanced ten-scene coverage plan, independent of chance."""
    variants = (
        {'active': ('A',), 'counts': (0,), 'kind': 'empty'},
        {'active': ('A',), 'counts': (1,), 'kind': 'single'},
        {'active': ('B',), 'counts': (3,), 'kind': 'multi'},
        {'active': ('C',), 'counts': (5,), 'kind': 'dense'},
        {'active': ('A', 'B'), 'counts': (1, 2), 'kind': 'two_buildings'},
        {'active': ('A', 'B', 'C'), 'counts': (1, 1, 1), 'kind': 'three_buildings'},
        {'active': ('C',), 'counts': (3,), 'kind': 'occluder'},
        {'active': ('A',), 'counts': (2,), 'kind': 'truncated'},
        {'active': ('A', 'B', 'C'), 'counts': (2, 0, 2), 'kind': 'mixed_zero'},
        {'active': ('B',), 'counts': (4,), 'kind': 'truncated'},
    )
    if index < 0:
        raise ValueError('scene index must be non-negative')
    return dict(variants[index % len(variants)])


def decode_ids(rgb, allowed_ids):
    """Decode object-color emission passes; black background has id zero.

    Colors encode R=1, G=id/32, B=0.25 in linear space. Ratios make
    anti-aliased edges against black less likely to look like another ID.
    The caller supplies linear RGB floats in image order, top row first.
    """
    data = np.asarray(rgb, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] < 3 or not np.isfinite(data).all():
        raise ValueError('RGB pass must be finite H×W×3 or H×W×4')
    ids = set(int(value) for value in allowed_ids)
    if any(value < 1 or value > 30 for value in ids):
        raise ValueError('object identifiers must be between 1 and 30')
    red = data[:, :, 0]
    denominator = np.maximum(red, 1e-7)
    green_ratio = data[:, :, 1] / denominator
    blue_ratio = data[:, :, 2] / denominator
    candidates = np.rint(green_ratio * 32).astype(np.int32)
    residual = np.abs(green_ratio - candidates / 32.0)
    valid = ((red >= .45) & (np.abs(blue_ratio - .25) < .035)
             & (residual < .012) & np.isin(candidates, list(ids)))
    result = np.zeros(red.shape, dtype=np.uint8)
    result[valid] = candidates[valid]
    return result


def visible_bbox(mask):
    """Return exclusive xyxy pixel coordinates and area, or None and zero."""
    y, x = np.nonzero(mask)
    if not len(x):
        return None, 0
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1], int(len(x))


def label_state(projected, bbox, pixels, minimum=12):
    """Do not reinterpret clipped, occluded, or subpixel targets as negatives."""
    if not projected['in_front']:
        return 'unknown_projection'
    if not projected['intersects']:
        return 'outside_frame'
    if pixels < minimum or bbox is None:
        return 'unknown_too_small_or_occluded'
    if projected['truncated']:
        return 'visible_truncated'
    return 'visible'


def fire_label_suppression(states, has_occluder):
    """Return why a whole-image fire label cannot safely be exported.

    A visible paper fragment does not establish that its printed flame is
    visible. Occlusion and frame clipping therefore suppress the entire image
    until an instance-level graphic visibility assessment is available.
    """
    if has_occluder:
        return 'occlusion_graphic_visibility_unverified'
    if any(state == 'visible_truncated' for state in states):
        return 'truncated_graphic_visibility_unverified'
    if any(state not in ('visible', 'outside_frame') for state in states):
        return 'unknown_target_visibility'
    return None


def ocr_status(state, has_occluder=False):
    """Mark generated full text as truth, never as an accepted OCR label."""
    if has_occluder:
        return 'ineligible_occlusion_unverified'
    if state == 'visible':
        return 'candidate_requires_readability_review'
    if state == 'visible_truncated':
        return 'ineligible_truncated'
    return 'ineligible_not_fully_visible'


def yolo_row(bbox, width, height):
    if width <= 0 or height <= 0 or len(bbox) != 4:
        raise ValueError('invalid image or box')
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('box must have positive area inside the image')
    return f'0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}'
