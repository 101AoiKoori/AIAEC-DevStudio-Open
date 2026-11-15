"""Pure helpers for Blender fire-scene scheduling and visible-instance labels."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/fire/. Production remains paused.")

import math
from pathlib import Path
import sys

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[3]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
from datasets.scripts.common.annotations import yolo_row


def placement_violations(boxes, placement, width=640, height=480):
    """Check projected building boxes against reviewed image-space constraints."""
    if not boxes:
        return ['no_building_projection']
    x0, y0, x1, y1 = placement['ground_anchor_xyxy']
    low, high = placement['building_height_px']
    union = [min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes)]
    anchor = ((union[0] + union[2]) / 2, union[3])
    issues = []
    if not (x0 <= anchor[0] <= x1 and y0 <= anchor[1] <= y1):
        issues.append('ground_anchor_outside_reviewed_region')
    for box in boxes:
        a, b, c, d = box
        if c <= a or d <= b:
            issues.append('degenerate_projection')
            continue
        if not low <= d-b <= high:
            issues.append('building_height_outside_reviewed_range')
        if not placement.get('allow_truncated', False) and (a < 0 or b < 0 or c > width or d > height):
            issues.append('building_clipped_without_approval')
        for left, top, right, bottom in placement['forbidden_rects_xyxy']:
            overlap = max(0, min(c, right) - max(a, left)) * max(0, min(d, bottom) - max(b, top))
            if overlap / ((c-a)*(d-b)) > placement['max_overlap_fraction']:
                issues.append('foreground_overlap_exceeds_limit')
                break
    return list(dict.fromkeys(issues))


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


def explicit_scenario(target):
    """Validate the renderer's bounded, front-facade scene combinations."""
    fields = {'scenario', 'active_buildings', 'fire_counts', 'view', 'placement', 'visibility'}
    if not isinstance(target, dict) or set(target) != fields or target['scenario'] != 'explicit':
        raise ValueError('invalid explicit scenario fields')
    active, counts = target['active_buildings'], target['fire_counts']
    if (not isinstance(active, list) or not active or
            any(key not in ('A', 'B', 'C') for key in active) or
            active != [key for key in ('A', 'B', 'C') if key in active]):
        raise ValueError('active buildings must be unique and ordered A, B, C')
    if not isinstance(counts, dict) or set(counts) != set(active) or any(
            type(counts[key]) is not int or not 0 <= counts[key] <= 5 for key in active):
        raise ValueError('each active building needs a fire count from 0 to 5')
    if target['view'] not in ('near_frontal', 'oblique_left', 'oblique_right'):
        raise ValueError('unsupported camera view')
    if target['placement'] not in ('all', 'left', 'right', 'upper', 'lower',
                                   'c_supported_left', 'c_supported_right'):
        raise ValueError('unsupported front-facade placement zone')
    visibility = target['visibility']
    if visibility not in ('no_added_occluder', 'occluder', 'truncated'):
        raise ValueError('unsupported visibility state')
    if visibility == 'occluder' and counts[active[0]] == 0:
        raise ValueError('occluder requires a sticker on the first active building')
    if target['placement'] in ('c_supported_left', 'c_supported_right') and (
            active != ['C'] or counts != {'C': 1} or visibility != 'no_added_occluder'):
        raise ValueError('supported wall placement requires a single C sticker and no added occluder')
    return {'active': tuple(active), 'counts': tuple(counts[key] for key in active),
            'kind': 'normal' if visibility == 'no_added_occluder' else visibility,
            'view': target['view'], 'placement': target['placement']}


def front_facade_slots(placement):
    """Return the existing non-overlapping front-facade grid, in metres."""
    columns = {'all': (-.065, -.025, .022, .061), 'left': (-.065, -.025),
               'right': (.022, .061), 'upper': (-.065, -.025, .022, .061),
               'lower': (-.065, -.025, .022, .061),
               'c_supported_left': (-.0465,), 'c_supported_right': (.0375,)}
    rows = {'all': (.115, .165, .215, .265, .315), 'left': (.115, .165, .215, .265, .315),
            'right': (.115, .165, .215, .265, .315), 'upper': (.265, .315),
            'lower': (.115, .165), 'c_supported_left': (.115, .165, .215, .265, .315),
            'c_supported_right': (.115, .165, .215, .265, .315)}
    if placement not in columns:
        raise ValueError('unsupported front-facade placement zone')
    return [(x, z) for x in columns[placement] for z in rows[placement]]


def rotated_paper_envelope(half_x, half_z, jitter, max_rotation):
    """Conservative x/z half-extents for in-plane rotation and centre jitter."""
    if (not all(math.isfinite(v) and v >= 0 for v in (half_x, half_z, jitter, max_rotation))
            or max_rotation > math.pi / 2):
        raise ValueError('invalid paper envelope inputs')
    ax = min(max_rotation, math.atan2(half_z, half_x))
    az = min(max_rotation, math.atan2(half_x, half_z))
    return (half_x * math.cos(ax) + half_z * math.sin(ax) + jitter,
            half_z * math.cos(az) + half_x * math.sin(az) + jitter)


def paper_depth_supported(front_y, paper_min_y, paper_max_y, padding, ripple):
    """Bound paper depth, including modifier displacement and surface ripple."""
    if (not all(math.isfinite(v) for v in (front_y, paper_min_y, paper_max_y, padding, ripple)) or
            paper_min_y > paper_max_y or padding < 0 or ripple < 0):
        raise ValueError('invalid paper depth inputs')
    near_y = paper_max_y + padding + ripple
    far_y = paper_min_y - padding - ripple
    return near_y <= front_y + .00003 and far_y >= front_y - .002


def rectangle_uncovered_area(rectangles, box):
    """Measure the area of an x/z footprint not covered by planar rectangles."""
    x0, z0, x1, z1 = box
    if not all(math.isfinite(v) for v in box) or x1 <= x0 or z1 <= z0:
        raise ValueError('invalid footprint')
    if any(len(r) != 4 or not all(math.isfinite(v) for v in r) or r[2] <= r[0] or r[3] <= r[1]
           for r in rectangles):
        raise ValueError('invalid support rectangle')
    cuts = sorted({x0, x1, *(max(x0, min(x1, x)) for r in rectangles for x in (r[0], r[2]))})
    missing = 0.0
    for left, right in zip(cuts, cuts[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2
        intervals = sorted((max(z0, r[1]), min(z1, r[3])) for r in rectangles
                           if r[0] <= mid <= r[2] and r[3] > z0 and r[1] < z1)
        covered = 0.0
        lo = hi = None
        for start, end in intervals:
            if end <= start:
                continue
            if lo is None:
                lo, hi = start, end
            elif start <= hi:
                hi = max(hi, end)
            else:
                covered += hi - lo
                lo, hi = start, end
        if lo is not None:
            covered += hi - lo
        missing += (right - left) * max(0.0, z1 - z0 - covered)
    return missing


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


def fire_scale_suppression(instances, min_width=14, min_height=20):
    """Keep tiny flame graphics unknown when the pixel evidence is insufficient."""
    for obj in instances:
        bbox = obj.get('pixel_bbox_xyxy')
        if obj.get('kind') == 'fire_sticker' and bbox is not None:
            if bbox[2] - bbox[0] < min_width or bbox[3] - bbox[1] < min_height:
                return 'fire_graphic_below_reviewable_pixel_scale'
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
