"""Create a bounded 640x480 sign-composition review batch from reviewed backgrounds.

The targets and their alpha masks originate from one physical sign video. Images
are correlated compositions, not independent acquisitions. Bounds are computed
from the final visible alpha mask; visual and source review remain required.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[3]
_TASK_HELPERS = _WORKSPACE / 'datasets/scripts/sign'
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
if str(_TASK_HELPERS) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from datasets.scripts.sign import compose_sign_scene as base
from datasets.scripts.sign import compose_sign_vehicle_pilot as pilot
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.paths import sample_dir


ROOT = base.ROOT
BACKGROUNDS = source_path('sign', 'background_manifest').parent
OUTPUT_ROOT = sample_dir('sign')
VIEWS = (('train_0.png', 'back'), ('train_8.png', 'side'),
         ('train_14.png', 'oblique'), ('train_20.png', 'front'), ('train_26.png', 'oblique'))
STATES = ('far_full', 'mid_full', 'near_full', 'top_clipped', 'bottom_clipped', 'placed')


def _polygon(value, name):
    """Validate a polygon in final-image pixel coordinates."""
    if (not isinstance(value, list) or len(value) < 3 or
            any(not isinstance(p, list) or len(p) != 2 or
                any(type(v) is not int or abs(v) > 10000 for v in p) for p in value)):
        raise ValueError(f'{name} must contain at least three integer pixel points')
    points = np.asarray(value, dtype=np.int32)
    if cv2.contourArea(points) <= 0:
        raise ValueError(f'{name} must have nonzero area')
    return points


def _placed_parameters(parameters):
    required = {'height', 'left', 'top', 'ground_polygon', 'occlusion_polygons', 'visible_kind',
                'visibility_evidence'}
    if not isinstance(parameters, dict) or not required.issubset(parameters) or set(parameters) - required - {'shadow_strength', 'retain_lower_half'}:
        raise ValueError('Placed sign needs geometry, ground region, occlusion polygons and visible kind')
    if 'retain_lower_half' in parameters and (parameters['retain_lower_half'] is not True or parameters['visible_kind'] != 'identifiable_pole'):
        raise ValueError('Lower-half crop must declare the designed identifiable pole state')
    if 'shadow_strength' in parameters and (type(parameters['shadow_strength']) not in (int, float)
            or not 0 <= parameters['shadow_strength'] <= .35):
        raise ValueError('Shadow strength must be between 0 and 0.35')
    if any(type(parameters[k]) is not int for k in ('height', 'left', 'top')):
        raise ValueError('Sign placement coordinates must be integers')
    if not 1 <= parameters['height'] <= 4000 or any(abs(parameters[k]) > 10000 for k in ('left', 'top')):
        raise ValueError('Sign placement is outside supported bounds')
    _polygon(parameters['ground_polygon'], 'Ground polygon')
    occlusions = parameters['occlusion_polygons']
    if not isinstance(occlusions, list):
        raise ValueError('Occlusion polygons must be a list')
    for polygon in occlusions:
        _polygon(polygon, 'Occlusion polygon')
    if parameters['visible_kind'] == 'unknown':
        raise ValueError('Unknown sign visibility cannot be written as a training label')
    if parameters['visible_kind'] not in ('fixture', 'identifiable_pole'):
        raise ValueError('Visible kind must identify a fixture or an identifiable pole')
    if not isinstance(parameters['visibility_evidence'], str) or not parameters['visibility_evidence'].strip():
        raise ValueError('Visible kind requires a documented identifying cue')


def validate_planned_stratum(stratum, frames):
    """Check the exact states this compositor can carry out before writing."""
    if not isinstance(stratum, dict):
        raise ValueError('Sign stratum must be an object')
    background = stratum.get('background')
    target = stratum.get('target')
    labels = stratum.get('labels')
    augmentation = stratum.get('augmentation')
    parameters = stratum.get('parameters')
    if not isinstance(background, dict) or set(background) != {'image'}:
        raise ValueError('Sign background must specify exactly image')
    matches = [row for row in frames if row.get('image') == background['image']]
    if len(matches) != 1:
        raise ValueError('Sign background is absent or ambiguous')
    if not isinstance(target, dict) or set(target) != {'state', 'mask'}:
        raise ValueError('Sign target must specify exactly state and mask')
    state, mask = target['state'], target['mask']
    negative = state == 'negative' and mask == 'none'
    if not negative and (state not in STATES or mask not in dict(VIEWS)):
        raise ValueError('Unsupported sign scale state or view mask')
    if labels != {'class_id': 0, 'kind': 'empty_candidate' if negative else 'visible_fixture_box'}:
        raise ValueError('Unsupported sign label contract')
    if not isinstance(augmentation, dict) or set(augmentation) != {'tone'} or augmentation['tone'] not in ('normal', 'dark_low_saturation'):
        raise ValueError('Unsupported sign augmentation')
    if negative:
        if parameters != {'kind': 'empty_background'}:
            raise ValueError('Negative sign state requires empty_background parameters')
        if 'no identifiable sign' not in matches[0].get('review_status', '').lower():
            raise ValueError('Background lacks a sign-absence candidate observation')
        return matches[0]
    if state == 'placed':
        _placed_parameters(parameters)
        if (matches[0].get('sign_absence') != 'confirmed' or
                not isinstance(matches[0].get('sign_absence_evidence'), str) or
                not matches[0]['sign_absence_evidence'].strip()):
            raise ValueError('Placed sign requires documented background sign absence')
        return matches[0]
    if not isinstance(parameters, dict) or set(parameters) != {'height', 'left', 'top'}:
        raise ValueError('Sign placement must specify exactly height, left and top')
    if any(type(parameters[k]) is not int for k in ('height', 'left', 'top')):
        raise ValueError('Sign placement coordinates must be integers')
    height, left, top = (parameters[k] for k in ('height', 'left', 'top'))
    bottom = top + height
    feasible = {
        'far_full': 160 <= height <= 239 and 390 <= bottom <= 465 and top >= 0,
        'mid_full': 240 <= height <= 365 and 390 <= bottom <= 465 and top >= 0,
        'near_full': 390 <= height <= 465 and 0 <= top and bottom <= 480,
        'top_clipped': 500 <= height <= 550 and 420 <= bottom <= 475 and top < 0,
        'bottom_clipped': 500 <= height <= 550 and 5 <= top <= 25 and bottom > 480,
    }
    if not feasible[state] or not 0 <= left < 640:
        raise ValueError('Infeasible sign placement for selected state')
    anchor = matches[0].get('road_anchor_center_x')
    if not isinstance(anchor, list) or len(anchor) != 2 or any(type(v) is not int for v in anchor) or anchor[0] > anchor[1]:
        raise ValueError('Background lacks a valid road anchor range')
    return matches[0]


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Could not encode {path}')
    encoded.tofile(str(path))


def compose(background, sprite, state, tone, rng, anchor_range, placement=None):
    if state == 'placed':
        return _compose_placed(background, sprite, tone, placement)
    if placement is not None:
        height, left, top = (placement[k] for k in ('height', 'left', 'top'))
        bottom = top + height
    elif state == 'far_full':
        height = rng.randint(160, 239)
        bottom = rng.randint(390, 465)
        top = bottom - height
    elif state == 'mid_full':
        height = rng.randint(240, 365)
        bottom = rng.randint(max(height, 390), 465)
        top = bottom - height
    elif state == 'near_full':
        height = rng.randint(390, 465)
        bottom = rng.randint(height, 480)
        top = bottom - height
    elif state == 'top_clipped':
        height = rng.randint(500, 550)
        bottom = rng.randint(420, 475)
        top = bottom - height
    elif state == 'bottom_clipped':
        height = rng.randint(500, 550)
        top = rng.randint(5, 25)
        bottom = top + height
    else:
        raise ValueError('Unknown scale state')
    width = max(2, round(sprite.shape[1] * height / sprite.shape[0]))
    if width >= 600:
        raise ValueError('Sign mask is too wide for the intended camera view')
    resized = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
    # The sign base must be placed over the road region of each real frame.
    # In particular the traffic-light frame has a raised wall at the right.
    min_left = max(15, anchor_range[0] - width // 2)
    max_left = min(625 - width, anchor_range[1] - width // 2)
    if min_left > max_left:
        raise ValueError('Sign cannot be placed within the reviewed road region')
    if placement is None:
        left = rng.randint(min_left, max_left)
    elif not min_left <= left <= max_left:
        raise ValueError('Planned sign is outside the reviewed road region')
    x1, y1, x2, y2 = max(0, left), max(0, top), min(640, left+width), min(480, bottom)
    src = resized[y1-top:y2-top, x1-left:x2-left]
    alpha = src[:, :, 3].astype(np.float32) / 255.0
    ys, xs = np.where(alpha >= .5)
    if not len(xs) or (state.endswith('clipped') and len(xs) < 300):
        raise ValueError('Visible sign alpha is empty or too small')
    image = background.copy().astype(np.float32)
    if 0 <= bottom < 480:
        shadow = np.zeros((480, 640), np.uint8)
        cv2.ellipse(shadow, (left+width//2, min(479, bottom)),
                    (max(4, width//4), max(2, height//70)), 0, 0, 360, 55, -1)
        image *= (1 - cv2.GaussianBlur(shadow, (0, 0), 4)[:, :, None] / 255.0)
    a = alpha[:, :, None]
    image[y1:y2, x1:x2] = src[:, :, :3] * a + image[y1:y2, x1:x2] * (1-a)
    image = np.clip(image, 0, 255).astype(np.uint8)
    gains = {'normal': (1.0, 1.0), 'dark_low_saturation': (.72, .65)}[tone]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= gains[0]
    hsv[:, :, 1] *= gains[1]
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    box = [x1+int(xs.min()), y1+int(ys.min()), x1+int(xs.max())+1, y1+int(ys.max())+1]
    return image, box, {'sprite_height_px': height, 'sprite_width_px': width,
                        'left': left, 'top': top, 'bottom': bottom,
                        'visible_alpha_pixels': int(len(xs)), 'value_gain': gains[0],
                        'saturation_gain': gains[1], 'shadow': 'approximate, not measured'}


def _compose_placed(background, sprite, tone, placement):
    """Compose with a declared ground region and foreground occluders.

    Polygons use the final 640x480 frame; off-frame ground coordinates allow a
    cropped base. Occlusion polygons describe objects already in the background.
    The caller declares whether visible pixels identify the fixture or its pole.
    """
    _placed_parameters(placement)
    if (background.shape != (480, 640, 3) or sprite.ndim != 3 or sprite.shape[2] != 4 or
            sprite.shape[0] == 0 or sprite.shape[1] == 0):
        raise ValueError('Expected 640x480 BGR background and BGRA sprite')
    height, left, top = (placement[k] for k in ('height', 'left', 'top'))
    width = max(1, round(sprite.shape[1] * height / sprite.shape[0]))
    if width > 4000:
        raise ValueError('Sign sprite width exceeds supported bounds')
    bottom = top + height
    anchor = (left + width / 2.0, float(bottom))
    if cv2.pointPolygonTest(_polygon(placement['ground_polygon'], 'Ground polygon'), anchor, False) < 0:
        raise ValueError('Sign contact point is outside the declared ground region')
    x1, y1, x2, y2 = max(0, left), max(0, top), min(640, left + width), min(480, bottom)
    if x1 >= x2 or y1 >= y2:
        raise ValueError('Sign has no visible image intersection')
    resized = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
    src = resized[y1-top:y2-top, x1-left:x2-left]
    alpha = src[:, :, 3].astype(np.float32) / 255.0
    occlusion = np.zeros((480, 640), dtype=np.uint8)
    if placement.get('retain_lower_half'):
        alpha[:max(0, min(alpha.shape[0], top + height // 2 - y1))] = 0
    for polygon in placement['occlusion_polygons']:
        cv2.fillPoly(occlusion, [_polygon(polygon, 'Occlusion polygon')], 1)
    alpha[occlusion[y1:y2, x1:x2] != 0] = 0
    ys, xs = np.where(alpha >= .5)
    if not len(xs):
        raise ValueError('No visible sign pixels remain after clipping and occlusion')
    image = background.copy().astype(np.float32)
    shadow_strength = placement.get('shadow_strength', 0)
    if shadow_strength and 0 <= bottom < 480:
        shadow = np.zeros((480, 640), np.uint8)
        cv2.ellipse(shadow, (round(anchor[0]), bottom),
                    (max(3, width // 3), max(2, height // 60)), 0, 0, 360, 255, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 3).astype(np.float32) / 255
        image *= 1 - shadow[:, :, None] * shadow_strength
    a = alpha[:, :, None]
    image[y1:y2, x1:x2] = src[:, :, :3] * a + image[y1:y2, x1:x2] * (1 - a)
    gains = {'normal': (1.0, 1.0), 'dark_low_saturation': (.72, .65)}[tone]
    hsv = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= gains[0]
    hsv[:, :, 1] *= gains[1]
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    box = [x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1]
    return image, box, {'sprite_height_px': height, 'sprite_width_px': width,
                        'left': left, 'top': top, 'bottom': bottom,
                        'contact_point_xy': list(anchor), 'ground_polygon': placement['ground_polygon'],
                        'occlusion_polygons': placement['occlusion_polygons'],
                        'visible_kind': placement['visible_kind'],
                        'retain_lower_half': placement.get('retain_lower_half', False),
                        'visibility_evidence': placement['visibility_evidence'],
                        'visible_alpha_pixels': int(len(xs)),
                        'value_gain': gains[0], 'saturation_gain': gains[1],
                        'shadow': 'approximate_ellipse' if shadow_strength else 'none',
                        'shadow_strength': shadow_strength}


def _prepare(declaration, manifest, frames, mask_data, background_root=None):
    if declaration.get('model_key') != 'sign' or declaration.get('constraints') != {
            'image_size': [640, 480], 'label_kind': 'visible_fixture_box'}:
        raise ValueError('Unsupported sign model or generation constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('A sign production plan with states is required')
    prepared, identities = {}, set()
    for state in states:
        row = validate_planned_stratum(state, frames)
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('Each exact-placement sign state permits at most one sample')
        mask = state['target']['mask']
        negative = state['target']['state'] == 'negative'
        sources = [] if negative else [r for r in mask_data['records'] if 'index' in r and f"train_{r['index']}.png" == mask]
        if not negative and len(sources) != 1:
            raise ValueError('Foreground mask source is absent or ambiguous')
        source_root = BACKGROUNDS if background_root is None else background_root
        expected = [(source_root / row['image']).relative_to(ROOT).as_posix(), row['parent_image']]
        if manifest.get('video'):
            expected.append(manifest['video'])
        if row.get('parent_video'):
            expected.append(row['parent_video'])
        if not negative:
            expected += [(base.ANNOTATED / mask).relative_to(ROOT).as_posix(),
                         sources[0]['source'], base.SIGN_VIDEO]
        expected = list(dict.fromkeys(expected))
        if state.get('sources') != expected:
            raise ValueError('Declared sources do not match selected sign inputs')
        view = 'none' if negative else dict(VIEWS)[mask]
        if set(state.get('conditions', [])) != {state['target']['state'], view, state['augmentation']['tone']}:
            raise ValueError('Sign conditions must describe state, view and tone')
        identity = json.dumps([state['background'], state['target'], state['augmentation'],
                               state['parameters']], sort_keys=True)
        if state['id'] in prepared or identity in identities:
            raise ValueError('Duplicate sign composition state')
        identities.add(identity)
        prepared[state['id']] = (row, None if negative else sources[0])
    return prepared


def _generate(output, plan, run, prepared, sampling_seed=None, background_root=None):
    output = Path(output).resolve()
    if output != OUTPUT_ROOT.resolve() or (output.exists() and any(output.iterdir())):
        raise ValueError('Candidate output must be the empty current sign sample directory')
    # Geometry is generated only after the shared guard allocates the plan.
    rendered = {}
    for state in plan['strata']:
        row, foreground = prepared[state['id']]
        if state['planned_count'] == 0:
            continue
        if state['target']['state'] == 'placed':
            validate_planned_stratum(state, [row])
        source_root = ROOT if row.get('workspace_image') is True else (background_root or BACKGROUNDS)
        frame = base.read_image(source_root / row['image'], cv2.IMREAD_COLOR)
        if row.get('crop_xyxy') is not None:
            x1, y1, x2, y2 = row['crop_xyxy']
            frame = cv2.resize(frame[y1:y2, x1:x2], (640, 480), interpolation=cv2.INTER_LINEAR)
        if frame.shape != (480, 640, 3):
            raise ValueError('Expected 640x480 BGR background')
        mask = state['target']['mask']
        if state['target']['state'] == 'negative':
            gains = {'normal': (1.0, 1.0), 'dark_low_saturation': (.72, .65)}[state['augmentation']['tone']]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= gains[0]
            hsv[:, :, 1] *= gains[1]
            image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            box = None
            geometry = {'value_gain': gains[0], 'saturation_gain': gains[1]}
        else:
            sprite = pilot.repaired_back(None, write_artifacts=False) if mask == 'train_0.png' else base.cutout(mask)
            image, box, geometry = compose(frame, sprite, state['target']['state'],
                                            state['augmentation']['tone'], None,
                                            row.get('road_anchor_center_x'), state['parameters'])
        rendered[state['id']] = (row, foreground, image, box, geometry)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir()
    (output / 'labels_candidate').mkdir()
    result = {'status': 'running', 'image_size': [640, 480],
              'target_split': plan['target_split'], 'foreground_source': base.SIGN_VIDEO,
              'scope': 'shared parent sources; no independent capture or model quality claim', 'samples': []}
    try:
        for state in plan['strata']:
            if state['planned_count'] == 0:
                continue
            row, foreground, image, box, geometry = rendered[state['id']]
            for _ in range(state['planned_count']):
                name = f"sign_{len(result['samples']):05d}"
                image_path = output / 'images' / (name + '.png')
                label_path = output / 'labels_candidate' / (name + '.txt')
                label = ''
                if box is not None:
                    x1, y1, x2, y2 = box
                    label = f'0 {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n'
                write_png(image_path, image)
                label_path.write_text(label, encoding='utf8')
                actual = {'geometry': geometry, 'bbox_xyxy': box, 'background': state['sources'][0],
                          'background_parent': row['parent_image'], 'background_time_seconds': row['pts_seconds'],
                          'foreground_mask': None if foreground is None else state['target']['mask'],
                          'foreground_source': None if foreground is None else foreground['source'],
                          'foreground_video': None if foreground is None else base.SIGN_VIDEO,
                          'view': 'none' if foreground is None else dict(VIEWS)[state['target']['mask']],
                          'state': state['target']['state'], 'tone': state['augmentation']['tone'],
                          'sampling_seed': sampling_seed}
                if state['target']['state'] == 'placed':
                    actual['background_sign_absence'] = row['sign_absence']
                    actual['background_sign_absence_evidence'] = row['sign_absence_evidence']
                    if row.get('crop_xyxy') is not None:
                        actual['background_crop_xyxy'] = row['crop_xyxy']
                        actual['background_output_size'] = [640, 480]
                    if row.get('parent_video'):
                        actual['background_video'] = row['parent_video']
                run.record_sample(state['id'], output=image_path, sources=state['sources'],
                                  parameters=state['parameters'], actual=actual)
                result['samples'].append({'id': name, 'state_id': state['id'],
                    'image': image_path.relative_to(output).as_posix(),
                    'label': label_path.relative_to(output).as_posix(),
                    'sources': state['sources'], 'declared_parameters': state['parameters'],
                    'actual': actual, 'review_status': 'candidate'})
        result['status'] = 'complete_candidates'
    except BaseException as exc:
        result.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def generate(output, seed, pilot_only, declaration=None, plan_path=None):
    from datasets.scripts.sign.sign_production_guard import guarded, existing_result
    if not isinstance(declaration, dict):
        raise ValueError('Explicit sign production plan is required')
    if any(isinstance(state, dict) and isinstance(state.get('id'), str) and
           state['id'].startswith(('photo-full_', 'photo-clip_'))
           for state in declaration.get('strata', [])):
        raise ValueError('Retired single-background fixed-quota sign plan cannot be produced')
    if 'sampling_seed' in declaration and declaration['sampling_seed'] != seed:
        raise ValueError('CLI seed does not match the production plan')
    design, source_config = load_task_config('sign')
    selection = source_config.get('current_selection')
    if selection:
        selected_images = set(selection['backgrounds'])
        selected_masks = set(selection['foreground_masks'])
        if any(state.get('background', {}).get('image') not in selected_images or
               state.get('target', {}).get('mask') not in selected_masks | {'none'}
               for state in declaration.get('strata', [])):
            raise ValueError('sign plan uses sources excluded from the current source.yaml selection')
    if design.get('sample_output') != 'datasets/samples/sign':
        raise ValueError('Sign sample output configuration is inconsistent')
    records = source_config.get('records')
    parent_roots = source_config.get('allowed_parent_roots')
    if not isinstance(records, list) or source_config.get('background_manifest') not in records or not isinstance(parent_roots, list) or not parent_roots:
        raise ValueError('Sign source configuration needs a background manifest and parent roots')
    allowed_roots = [(ROOT / item).resolve() for item in parent_roots]
    if any(not root.is_relative_to(ROOT) for root in allowed_roots):
        raise ValueError('Sign parent roots must stay inside the workspace')
    output_path = Path(output).resolve()
    if output_path != OUTPUT_ROOT.resolve():
        raise ValueError('Candidate output must be datasets/samples/sign')
    if pilot_only:
        raise ValueError('Legacy pilot mode does not accept production plans')
    custom = 'background_inputs' in declaration
    if custom:
        frames = declaration['background_inputs']
        if not isinstance(frames, list) or not frames:
            raise ValueError('Explicit background inputs must be a nonempty list')
        if any(state.get('target', {}).get('state') != 'placed' for state in declaration.get('strata', [])):
            raise ValueError('Explicit background inputs support placed states only')
        for row in frames:
            if not isinstance(row, dict) or not isinstance(row.get('image'), str):
                raise ValueError('Background input needs a workspace-relative image')
            image = (ROOT / row['image']).resolve()
            if not image.is_relative_to(ROOT) or image.relative_to(ROOT).as_posix() != row['image'] or not image.is_file():
                raise ValueError('Background image must be an existing canonical workspace path')
            if row.get('workspace_image') is not True or row.get('parent_image') != row['image'] or 'pts_seconds' not in row:
                raise ValueError('Direct background must record its original image and capture time')
            parent_video = row.get('parent_video')
            if parent_video is not None:
                if not isinstance(parent_video, str):
                    raise ValueError('Parent video must be a workspace-relative path')
                parent_path = (ROOT / parent_video).resolve()
                if (not parent_path.is_relative_to(ROOT) or
                        parent_path.relative_to(ROOT).as_posix() != parent_video or not parent_path.is_file()):
                    raise ValueError('Parent video must be an existing canonical workspace path')
                if not any(parent_path.is_relative_to(root) for root in allowed_roots):
                    raise ValueError('Parent video is outside the configured source roots')
            elif not any(image.is_relative_to(root) for root in allowed_roots):
                raise ValueError('Background without a parent video must be an allowed original source')
            crop = row.get('crop_xyxy')
            if crop is not None:
                if (not isinstance(crop, list) or len(crop) != 4 or any(type(n) is not int for n in crop)):
                    raise ValueError('Background crop must contain four integer pixel coordinates')
                source = base.read_image(image, cv2.IMREAD_COLOR)
                x1, y1, x2, y2 = crop
                if (not 0 <= x1 < x2 <= source.shape[1] or not 0 <= y1 < y2 <= source.shape[0] or
                        (x2 - x1) * 3 != (y2 - y1) * 4):
                    raise ValueError('Background crop must fit the image and have 4:3 aspect ratio')
        manifest, path, background_root = {}, None, ROOT
    else:
        path = source_path('sign', 'background_manifest')
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Configured sign background manifest is unavailable')
        manifest = json.loads(path.read_text(encoding='utf8'))
        if manifest.get('status') != 'AI_reviewed_sign_empty_candidates':
            raise ValueError('Backgrounds must be reviewed for sign targets first')
        frames = [row for row in manifest['frames'] if row.get('usage') != 'regression']
        background_root = path.parent
    mask_data = json.loads((base.ANNOTATED / 'manifest.json').read_text(encoding='utf8'))
    prepared = _prepare(declaration, manifest, frames, mask_data, background_root)
    media = [background_root / row['image'] for row in frames]
    media += [ROOT / row['parent_image'] for row in frames]
    media += [ROOT / row['parent_video'] for row in frames if row.get('parent_video')]
    names = ({state['target']['mask'] for state in declaration['strata']} if custom else
             {name for name, _ in VIEWS})
    media += [base.ANNOTATED / name for name in sorted(names)]
    media += [ROOT / row['source'] for row in mask_data['records']
              if 'index' in row and f"train_{row['index']}.png" in names]
    media += [ROOT / base.SIGN_VIDEO]
    if manifest.get('video'):
        background_parent = (ROOT / manifest['video']).resolve()
        if (not background_parent.is_file() or
                not any(background_parent.is_relative_to(root) for root in allowed_roots)):
            raise ValueError('Background video is outside the configured source roots or missing')
        media.append(background_parent)
    sign_parent = (ROOT / base.SIGN_VIDEO).resolve()
    if not sign_parent.is_file() or not any(sign_parent.is_relative_to(root) for root in allowed_roots):
        raise ValueError('Sign parent is outside the configured source roots or missing')
    procedural = [base.ANNOTATED / 'manifest.json',
                  ROOT / 'datasets/scripts/sign/config/dataset.yaml',
                  ROOT / 'datasets/scripts/sign/config/source.yaml']
    if path is not None:
        procedural.append(path)
    if plan_path is not None:
        plan_path = Path(plan_path).resolve()
        if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
            raise ValueError('Plan must be an existing file inside the workspace')
        procedural.append(plan_path)
    with guarded('compose_v4_measured', output, media, procedural,
                 {'seed': seed, 'pilot_only': pilot_only},
                 [Path(__file__), Path(base.__file__), Path(pilot.__file__),
                  ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py'],
                 production_plan=declaration, sample_output_paths=[Path(output) / 'images']) as run:
        if run.reused:
            return existing_result(output)
        return _generate(output, run.plan, run, prepared, sampling_seed=seed, background_root=background_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=4901)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan_path = (ROOT / args.plan).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('Plan must be an existing file inside the workspace')
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    result = generate(ROOT/args.output, args.seed, args.pilot, declaration, plan_path=plan_path)
    print(json.dumps({'status': result['status'], 'count': len(result['samples'])}, ensure_ascii=False))
