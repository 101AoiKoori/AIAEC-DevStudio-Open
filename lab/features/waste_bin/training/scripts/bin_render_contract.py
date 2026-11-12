"""Pure checks for bounded, source-aware bin-lid render candidates."""
import math

import numpy as np


BIN_NAMES = ('blue', 'red', 'green', 'gray')
MASK_COLORS = {
    'blue': (1, 0, 0),
    'red': (0, 1, 0),
    'green': (0, 0, 1),
    'gray': (1, 1, 0),
}
OPEN_POSES = {'hanging_back': -270, 'upright': -90}
STATE_CONTRACT = 'whole_bin_state_with_mouth_or_lid_evidence_v1'
BALANCED_ORDER = (0, 15, 5, 10, 3, 12, 6, 9, 1, 14, 2, 13, 4, 11, 7, 8)


def capture_material_baseline(materials):
    """Take immutable node values once; never substitute viewport display colors."""
    result = {}
    for name in BIN_NAMES:
        node = materials[name + ' adjustable body color'].node_tree.nodes.get('Principled BSDF')
        if node is None:
            raise ValueError(f'{name}: Principled BSDF is missing')
        color, roughness = node.inputs['Base Color'], node.inputs['Roughness']
        if color.is_linked or roughness.is_linked:
            raise ValueError(f'{name}: linked color/roughness requires an explicit sampling strategy')
        rgb = tuple(float(v) for v in color.default_value[:3])
        rough = float(roughness.default_value)
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (*rgb, rough)):
            raise ValueError(f'{name}: nonfinite or invalid material value')
        result[name] = {'base_color': rgb, 'roughness': rough}
    return result


def vary_material(base, color_factor, roughness_delta):
    if not math.isfinite(color_factor) or color_factor <= 0 or not math.isfinite(roughness_delta):
        raise ValueError('invalid material variation')
    rgb = tuple(min(1.0, max(0.0, v * color_factor)) for v in base['base_color'])
    roughness = min(1.0, max(0.0, base['roughness'] + roughness_delta))
    return rgb, roughness


def combinations(count):
    if not 1 <= count <= 16:
        raise ValueError('state combinations must be 1..16')
    return list(range(16)) if count == 16 else list(BALANCED_ORDER[:count])


def decode_mask(rgba, name):
    """Decode solid emission colors, conservatively excluding antialiased edges."""
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4 or not np.isfinite(array).all():
        raise ValueError('mask image must be finite H×W×4')
    if name not in MASK_COLORS:
        raise ValueError('unknown bin')
    mask = array[:, :, 3] >= .75
    for channel, value in enumerate(MASK_COLORS[name]):
        mask &= array[:, :, channel] >= .75 if value else array[:, :, channel] <= .25
    return mask


def measure_visibility(visible, solo, inside_frame, min_visibility=.70, min_pixels=24):
    visible = np.asarray(visible, dtype=bool)
    solo = np.asarray(solo, dtype=bool)
    if visible.shape != solo.shape or visible.ndim != 2:
        raise ValueError('visible and isolated masks must share H×W')
    if not 0 < min_visibility <= 1 or min_pixels < 1:
        raise ValueError('invalid visibility limits')
    visible_count, solo_count = int(visible.sum()), int(solo.sum())
    # Antialiasing at mask boundaries can differ slightly across the two passes.
    if visible_count > solo_count + max(4, int(solo_count * .03)):
        raise ValueError('visible mask exceeds isolated mask; mask passes are inconsistent')
    fraction = min(1.0, visible_count / solo_count) if solo_count else None
    reasons = []
    if not inside_frame:
        reasons.append('clipped_or_behind_camera')
    if not solo_count:
        reasons.append('no_isolated_pixels')
    if visible_count < min_pixels:
        reasons.append('too_few_visible_pixels')
    if fraction is not None and fraction < min_visibility:
        reasons.append('occluded_below_threshold')
    box = None
    if visible_count:
        y, x = np.nonzero(visible)
        height, width = visible.shape
        box = [int(x.min()) / width, int(y.min()) / height,
               (int(x.max()) + 1) / width, (int(y.max()) + 1) / height]
    return {'candidate_xyxy_normalized': box, 'visible_pixels': visible_count,
            'isolated_pixels': solo_count, 'visible_fraction': fraction,
            'in_frame': bool(inside_frame), 'label_eligible': not reasons,
            'rejection_reasons': reasons, 'visibility_verified': 'pixel_mask_estimate'}


def projected_points_fit(points, margin=0.0):
    """Reject geometry outside the camera, including behind-camera corners."""
    if not 0 <= margin < .5:
        raise ValueError('margin must be in [0, .5)')
    points = list(points)
    return bool(points) and all(math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
                                and z > 0 and margin <= x <= 1 - margin
                                and margin <= y <= 1 - margin for x, y, z in points)


def required_hinge_shift(back_y, panel_front_y, clearance=.015, max_shift=.15):
    """Move the hinge rearward just enough to clear the bin's back wall."""
    if not all(math.isfinite(x) for x in (back_y, panel_front_y, clearance, max_shift)):
        raise ValueError('hinge geometry must be finite')
    if clearance < 0 or max_shift <= 0:
        raise ValueError('invalid hinge clearance bounds')
    shift = max(0.0, back_y + clearance - panel_front_y)
    if shift > max_shift:
        raise ValueError('required hinge shift exceeds safe construction limit')
    return shift


def attach_state_evidence(target, evidence, state):
    """Require both a visible bin and its open-mouth or closed-lid evidence."""
    if state not in ('open', 'closed'):
        raise ValueError('unknown bin state')
    result = dict(target)
    result['target_eligible'] = target['label_eligible']
    result['target_rejection_reasons'] = list(target['rejection_reasons'])
    result['state_evidence'] = dict(evidence)
    result['state_evidence']['kind'] = 'mouth_proxy' if state == 'open' else 'lid'
    result['label_eligible'] = target['label_eligible'] and evidence['label_eligible']
    result['rejection_reasons'] = (list(target['rejection_reasons']) +
        ['state_evidence:' + reason for reason in evidence['rejection_reasons']])
    result['state'] = state
    return result
