"""Pure checks for bounded, source-aware bin-lid render candidates."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/waste_bin_state/. Production remains paused.")

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


def arena_planned_jobs(plan, scene_path, asset_path):
    """Validate the exact state vocabulary implemented by the arena renderer."""
    if plan['model_key'] != 'waste_bin_state':
        raise ValueError('arena renderer requires a waste_bin_state plan')
    if plan['constraints'] != {'resolution': [640, 480], 'label_mode': 'candidate'}:
        raise ValueError('arena renderer only supports 640x480 candidate labels')
    jobs = []
    seen = []
    for state in plan['strata']:
        if state['conditions']:
            raise ValueError('arena renderer does not implement additional conditions')
        if state['background'] != {'scene': scene_path}:
            raise ValueError('planned background differs from loaded arena')
        if state['sources'] != [scene_path, asset_path]:
            raise ValueError('planned sources must be the loaded arena and bin asset')
        target = state['target']
        if set(target) != {'bins', 'open_pose'} or not isinstance(target['bins'], dict):
            raise ValueError('target must specify the selected bins and states')
        if target['open_pose'] != 'upright':
            raise ValueError('arena renderer only supports upright open lids')
        bins = target['bins']
        if any(name not in BIN_NAMES or value not in ('open', 'closed') for name, value in bins.items()):
            raise ValueError('invalid bin name or state')
        if state['labels'] != {'mode': 'candidate', 'open': 0, 'closed': 1}:
            raise ValueError('unsupported label contract')
        aug = state['augmentation']
        if set(aug) != {'exposure_ev', 'saturation'}:
            raise ValueError('unsupported arena augmentation')
        for key, low, high in (('exposure_ev', -4, 4), ('saturation', 0, 2)):
            value = aug[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('invalid ' + key)
        params = state['parameters']
        if set(params) != {'bin_location', 'camera_location', 'camera_aim'}:
            raise ValueError('unsupported arena parameters')
        for key in params:
            vector = params[key]
            if (not isinstance(vector, list) or len(vector) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 100 for v in vector)):
                raise ValueError('invalid ' + key)
        if params['camera_location'] == params['camera_aim']:
            raise ValueError('camera location and aim must differ')
        if state['planned_count'] > 1:
            raise ValueError('arena renderer permits at most one image per exact state')
        identity = {key: state[key] for key in
                    ('background', 'target', 'labels', 'augmentation', 'parameters', 'sources')}
        if not bins:
            identity['parameters'] = {key: value for key, value in params.items() if key != 'bin_location'}
        if state['planned_count'] and identity in seen:
            raise ValueError('duplicate exact arena state in plan')
        if state['planned_count']:
            seen.append(identity)
        jobs.extend([state] * state['planned_count'])
    if len(jobs) != plan['budget']:
        raise ValueError('arena allocation does not match budget')
    return jobs


def standalone_planned_jobs(plan, scene_path, width, height, min_visibility, min_pixels):
    """Accept only standalone renderer states that can be set explicitly."""
    if plan['model_key'] != 'waste_bin_state' or plan['constraints'] != {
            'resolution': [width, height], 'label_mode': 'candidate',
            'min_visibility': min_visibility, 'min_visible_pixels': min_pixels}:
        raise ValueError('unsupported model or render constraints')
    jobs, seen = [], []
    for state in plan['strata']:
        if state['conditions'] or state['sources'] != [scene_path]:
            raise ValueError('unsupported conditions or scene source')
        bg, target, aug, params = (state[key] for key in ('background', 'target', 'augmentation', 'parameters'))
        if set(bg) != {'scene', 'floor_gray', 'backdrop_gray'} or bg['scene'] != scene_path:
            raise ValueError('background must identify the loaded scene and two material grays')
        if set(target) != {'bins', 'open_pose'} or not isinstance(target['bins'], dict):
            raise ValueError('target must specify bins and open pose')
        if target['open_pose'] not in OPEN_POSES or any(
                name not in BIN_NAMES or value not in ('open', 'closed') for name, value in target['bins'].items()):
            raise ValueError('unsupported bin state or pose')
        if state['labels'] != {'mode': 'candidate', 'open': 0, 'closed': 1} or aug != {'mode': 'none'}:
            raise ValueError('unsupported labels or augmentation')
        required = {'camera_location', 'camera_aim', 'yaw_degrees', 'lens_mm', 'key_energy', 'fill_energy',
                    'color_factor', 'roughness_delta'}
        if set(params) != required:
            raise ValueError('unsupported renderer parameters')
        for key in ('camera_location', 'camera_aim'):
            value = params[key]
            if (not isinstance(value, list) or len(value) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 100 for v in value)):
                raise ValueError('invalid ' + key)
        if params['camera_location'] == params['camera_aim']:
            raise ValueError('camera location and aim must differ')
        for obj, key, low, high in ((bg, 'floor_gray', 0, 1), (bg, 'backdrop_gray', 0, 1),
                                     (params, 'yaw_degrees', -180, 180), (params, 'lens_mm', 18, 200),
                                     (params, 'key_energy', 0, 10000), (params, 'fill_energy', 0, 10000),
                                     (params, 'color_factor', .01, 2), (params, 'roughness_delta', -1, 1)):
            value = obj[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('invalid ' + key)
        identity = {key: state[key] for key in ('background', 'target', 'labels', 'augmentation', 'parameters', 'sources')}
        identity['target'] = dict(target)
        if 'open' not in target['bins'].values():
            identity['target']['open_pose'] = None
        if not target['bins']:
            identity['parameters'] = {key: value for key, value in params.items()
                                      if key not in ('color_factor', 'roughness_delta')}
        if state['planned_count'] > 1 or (state['planned_count'] and identity in seen):
            raise ValueError('standalone renderer permits one image per distinct exact state')
        if state['planned_count']:
            seen.append(identity)
            jobs.append(state)
    if len(jobs) != plan['budget']:
        raise ValueError('standalone allocation does not match budget')
    return jobs


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


def mouth_pixel_contrast(rgb, mouth, body, min_pixels=24, max_ratio=.75):
    """Conservative image-space screen, not a substitute for label review."""
    rgb = np.asarray(rgb, dtype=float)
    mouth, body = np.asarray(mouth, dtype=bool), np.asarray(body, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[:2] != mouth.shape or body.shape != mouth.shape:
        raise ValueError('image and masks must share dimensions')
    if not np.isfinite(rgb).all() or min_pixels < 1 or not 0 < max_ratio < 1:
        raise ValueError('invalid image or contrast threshold')
    reference = body & ~mouth
    luminance = rgb[..., :3] @ np.array([.2126, .7152, .0722])
    n_mouth, n_body = int(mouth.sum()), int(reference.sum())
    mouth_median = float(np.median(luminance[mouth])) if n_mouth else None
    body_median = float(np.median(luminance[reference])) if n_body else None
    ratio = mouth_median / body_median if body_median and mouth_median is not None else None
    passed = n_mouth >= min_pixels and n_body >= min_pixels and ratio is not None and ratio <= max_ratio
    return {'method': 'visible_mouth_to_other_visible_bin_median_luminance',
            'mouth_pixels': n_mouth, 'reference_pixels': n_body, 'mouth_median': mouth_median,
            'reference_median': body_median, 'ratio': ratio, 'max_ratio': max_ratio,
            'passed': bool(passed), 'interpretation': 'heuristic_candidate_screen_not_label_review'}
