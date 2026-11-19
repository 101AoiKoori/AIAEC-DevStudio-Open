"""Create a bounded 640x480 sign-composition review batch from reviewed backgrounds.

The targets and their alpha masks originate from one physical sign video. Images
are correlated compositions, not independent acquisitions. Bounds are computed
from the final visible alpha mask; visual and source review remain required.
"""
import argparse
import json
from functools import lru_cache
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
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.review.placement_record import (
    contact_status_in_view, decode_placement_record,
)
from datasets.scripts.common.working_inputs import CODE as WORKING_INPUTS_CODE, WorkingInputs
from datasets.scripts.common.paths import sample_dir


ROOT = base.ROOT
BACKGROUNDS = source_path('sign', 'background_manifest').parent
OUTPUT_ROOT = sample_dir('sign')
VIEWS = (('train_0.png', 'back'), ('train_8.png', 'side'),
         ('train_14.png', 'oblique'), ('train_20.png', 'front'), ('train_26.png', 'oblique'))
STATES = ('far_full', 'mid_full', 'near_full', 'top_clipped', 'bottom_clipped', 'placed')
TONE_GAINS = {'normal': (1.0, 1.0), 'dark': (.72, 1.0),
              'low_saturation': (1.0, .65), 'dark_low_saturation': (.72, .65)}


def background_review_reference(review):
    return {'frame_id': review['frame_id'], 'record_id': review['record_id'],
            'evidence_ids': sorted(item['evidence_id'] for item in review['evidence']),
            'placement_algorithm': review['geometry']['placeable_ground']['algorithm_version']}


def existing_sign_boxes(review, crop=None):
    """Map complete source-pixel sign boxes into the 640x480 composition view."""
    label = review['label']
    instances = label.get('instances') or []
    if label['presence'] == 'absent':
        if instances:
            raise ValueError('empty sign background contains sign instances')
        return []
    if label['presence'] != 'present' or not instances:
        raise ValueError('positive sign background needs complete visible instances')
    width, height = review['image_size']
    left, top, right, bottom = crop or [0, 0, width, height]
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError('existing sign crop is outside the source image')
    content = output_content_rectangle(review['image_size'], crop)
    dx0, dy0, dx1, dy1 = content
    mapped = []
    for item in instances:
        box = item.get('xyxy')
        if (item.get('class_id') != 0 or not isinstance(box, list) or len(box) != 4 or
                any(type(value) is not int for value in box) or
                not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError('existing sign needs an exact class and source-pixel box')
        x1, y1, x2, y2 = max(left, box[0]), max(top, box[1]), min(right, box[2]), min(bottom, box[3])
        if x1 >= x2 or y1 >= y2:
            continue
        output = [round(dx0+(x1-left)*(dx1-dx0)/(right-left)),
                  round(dy0+(y1-top)*(dy1-dy0)/(bottom-top)),
                  round(dx0+(x2-left)*(dx1-dx0)/(right-left)),
                  round(dy0+(y2-top)*(dy1-dy0)/(bottom-top))]
        if output[0] >= output[2] or output[1] >= output[3]:
            raise ValueError('existing sign disappears after crop and resize')
        mapped.append({'class_id': 0, 'source_box_xyxy': box, 'output_box_xyxy': output,
                       'object_id': item.get('object_id')})
    return mapped


def output_content_rectangle(source_size, crop=None):
    width, height = source_size
    if crop is not None or width * 3 == height * 4:
        return [0, 0, 640, 480]
    if width * 9 == height * 16:
        return [0, 60, 640, 420]
    scale = min(640/width,480/height)
    scaled_w,scaled_h = round(width*scale),round(height*scale)
    left,top = (640-scaled_w)//2,(480-scaled_h)//2
    return [left,top,left+scaled_w,top+scaled_h]


def pool_ground_rectangle(review):
    geometry = review['geometry']
    size = review['image_size']
    if geometry.get('coordinate_size') != size:
        raise ValueError('sign pool ground coordinates differ from the source image')
    if geometry.get('support_rectangle_xyxy'):
        return geometry['support_rectangle_xyxy'], geometry.get('protected_rectangles_xyxy', [])
    polygon = geometry.get('ground_polygon')
    if (not isinstance(polygon, list) or len(polygon) < 3 or
            any(not isinstance(point, list) or len(point) != 2 for point in polygon)):
        raise ValueError('sign pool has no usable ground polygon')
    crop = geometry.get('crop_xyxy')
    content = output_content_rectangle(size, crop)
    if crop is not None:
        raise ValueError('ground polygon without a support rectangle cannot use a crop')
    x1 = round(min(point[0] for point in polygon) * 640 / size[0])
    x2 = min(content[2] - 1, round(max(point[0] for point in polygon) * 640 / size[0]))
    y1 = round(content[1] + min(point[1] for point in polygon) * (content[3]-content[1]) / size[1])
    y2 = min(content[3] - 1, round(content[1] + max(point[1] for point in polygon) * (content[3]-content[1]) / size[1]))
    if not 0 <= x1 < x2 < 640 or not 0 <= y1 < y2 < 480:
        raise ValueError('sign pool ground polygon exceeds the output image')
    return [int(x1), int(y1), int(x2), int(y2)], []


def bind_background_reviews(frames, source_config, split='train'):
    """Require current task and crop-specific ingest decisions for selected frames."""
    database = ROOT / source_config['databases']['frame_environment']
    reviews = query_frames(task='sign', role='composite_background', split=split,
                           status=None, database=database)
    reviews += query_frames(task='sign', role='direct_target', split=split,status=None,database=database)
    by_path = {(review['role'],path): review for review in reviews for path in review['paths']}
    bound, missing = [], []
    for row in frames:
        role = row.get('use_role','composite_background')
        review = by_path.get((role,row['image']))
        label = review.get('label', {}) if review else {}
        scope = label.get('scope', {})
        geometry = review.get('geometry', {}) if review else {}
        placement = geometry.get('placeable_ground')
        same_crop = scope.get('kind') == 'frame' or (
            scope.get('kind') == 'crop' and scope.get('crop_xyxy') == row.get('crop_xyxy'))
        if role == 'direct_target':
            if (not review or review['status']!='eligible' or label.get('presence')!='present' or
                label.get('completeness')!='complete' or scope.get('kind')!='frame' or
                image_digest(row['image'])!=review['content_sha256']):
                raise ValueError('Direct sign source lacks current complete frame labels: '+row['image'])
            refs = {'frame_id':review['frame_id'],'record_id':review['record_id'],
                'evidence_ids':sorted(item['evidence_id'] for item in review['evidence']),
                'placement_algorithm':'direct_frame_labels'}
            bound.append({**row,**refs,'sign_absence':'existing_labelled',
                'sign_absence_evidence':f"frame_environment:{review['record_id']}",
                'existing_sign_boxes':existing_sign_boxes(review,row.get('crop_xyxy'))})
            continue
        if (not review or review['status'] != 'eligible' or
                label.get('presence') not in ('absent', 'present') or
                label.get('completeness') != 'complete' or
                not same_crop or not placement):
            reason = ','.join(review['reason_codes']) if review else 'frame_not_registered'
            missing.append(f"{row['image']}:{reason}")
            continue
        if (placement.get('image_shape') != [review['image_size'][1], review['image_size'][0]] or
                image_digest(row['image']) != review['content_sha256']):
            raise ValueError(f"Reviewed sign frame pixels or coordinates changed: {row['image']}")
        decode_placement_record(placement)
        existing = existing_sign_boxes(review, row.get('crop_xyxy'))
        if label['presence'] == 'present' and scope.get('kind') == 'crop' and not existing:
            raise ValueError('positive crop sign labels have no visible box in the selected crop')
        positive = bool(existing)
        bound.append({**row, **background_review_reference(review),
                      'sign_absence': 'existing_labelled' if positive else 'confirmed',
                      'sign_absence_evidence': f"frame_environment:{review['record_id']}",
                      **({'existing_sign_boxes': existing} if positive else {})})
    if missing:
        raise ValueError('Sign backgrounds lack ingest eligibility: ' + '; '.join(missing))
    return bound


@lru_cache(maxsize=256)
def _stored_sign_review(record_id, database):
    records = query_frames(task='sign', role='composite_background', record_id=record_id,
                           database=database)
    if len(records) != 1 or not records[0]['current']:
        raise ValueError('Sign plan background decision is no longer current')
    review = records[0]
    return review, decode_placement_record(review['geometry']['placeable_ground'])


def recorded_contact_status(row, foot_xy, *, contact_width):
    """Map the planned output contact back into the saved source-pixel mask."""
    _, source_config = load_task_config('sign')
    database = ROOT / source_config['databases']['frame_environment']
    review, placement = _stored_sign_review(row['record_id'], database)
    if review['frame_id'] != row['frame_id']:
        raise ValueError('Sign plan background identity has changed')
    if background_review_reference(review) != {key: row[key] for key in (
            'frame_id', 'record_id', 'evidence_ids', 'placement_algorithm')}:
        raise ValueError('Sign plan background evidence has changed')
    width, height = review['image_size']
    crop = row.get('crop_xyxy') or [0, 0, width, height]
    return contact_status_in_view(placement, foot_xy, contact_width=contact_width,
                                  crop_xyxy=crop,
                                  content_xyxy=output_content_rectangle(review['image_size'], row.get('crop_xyxy')))


@lru_cache(maxsize=128)
def _sign_contact_width(mask, height):
    sprite = pilot.repaired_back(None, write_artifacts=False) if mask == 'train_0.png' else base.cutout(mask)
    alpha = sprite[:, :, 3] >= 127
    ys, xs = np.nonzero(alpha)
    if not len(xs):
        raise ValueError(f'Sign source has no visible pixels: {mask}')
    bottom_xs = xs[ys >= ys.max() - 2]
    return max(1.0, float(bottom_xs.max() - bottom_xs.min() + 1) * height / sprite.shape[0])


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
    if not isinstance(parameters, dict) or not required.issubset(parameters) or set(parameters) - required - {'shadow_strength', 'retain_lower_half', 'layout'}:
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
    ref_keys = {'frame_id', 'record_id', 'evidence_ids', 'placement_algorithm'}
    if not isinstance(background, dict) or set(background) not in ({'image'}, {'image'} | ref_keys):
        raise ValueError('Sign background must name the image and ingest decision')
    matches = [row for row in frames if row.get('image') == background['image']]
    if len(matches) != 1:
        raise ValueError('Sign background is absent or ambiguous')
    if ref_keys.issubset(matches[0]) and background != {
            'image': matches[0]['image'], **{key: matches[0][key] for key in ref_keys}}:
        raise ValueError('Sign background decision differs from the selected frame')
    if not ref_keys.issubset(matches[0]) and set(background) != {'image'}:
        raise ValueError('Sign background has no matching ingest decision')
    if not isinstance(target, dict) or set(target) != {'state', 'mask'}:
        raise ValueError('Sign target must specify exactly state and mask')
    state, mask = target['state'], target['mask']
    negative = state == 'negative' and mask == 'none'
    if not negative and (state not in STATES or mask not in dict(VIEWS)):
        raise ValueError('Unsupported sign scale state or view mask')
    negative_kind = 'retained_background_boxes' if matches[0].get('existing_sign_boxes') else 'empty_candidate'
    if labels != {'class_id': 0, 'kind': negative_kind if negative else 'visible_fixture_box'}:
        raise ValueError('Unsupported sign label contract')
    if not isinstance(augmentation, dict) or set(augmentation) != {'tone'} or augmentation['tone'] not in TONE_GAINS:
        raise ValueError('Unsupported sign augmentation')
    if negative:
        if (not isinstance(parameters, dict) or parameters.get('kind') != 'empty_background' or
                set(parameters) - {'kind', 'negative_context', 'scene_evidence','crop_xyxy'}):
            raise ValueError('Negative sign state requires empty_background parameters')
        if parameters.get('crop_xyxy') is not None:
            if matches[0].get('existing_sign_boxes') or matches[0].get('sign_absence')!='confirmed':
                raise ValueError('State-specific negative crops require a complete absent frame')
            crop=parameters['crop_xyxy']
            frame=base.read_image(ROOT/matches[0]['image'],cv2.IMREAD_COLOR)
            if (not isinstance(crop,list) or len(crop)!=4 or any(type(v) is not int for v in crop) or
                not 0<=crop[0]<crop[2]<=frame.shape[1] or not 0<=crop[1]<crop[3]<=frame.shape[0] or
                (crop[2]-crop[0])*3!=(crop[3]-crop[1])*4):
                raise ValueError('Negative crop must fit native pixels and retain 4:3 geometry')
        if (matches[0].get('sign_absence') not in ('candidate', 'confirmed', 'existing_labelled') and
                'no identifiable sign' not in matches[0].get('review_status', '').lower()):
            raise ValueError('Background lacks a sign-absence candidate observation')
        return matches[0]
    if state == 'placed':
        _placed_parameters(parameters)
        if (matches[0].get('sign_absence') not in ('candidate', 'confirmed', 'existing_labelled') or
                not isinstance(matches[0].get('sign_absence_evidence'), str) or
                not matches[0]['sign_absence_evidence'].strip()):
            raise ValueError('Placed sign requires documented background sign absence or complete labels')
        if ref_keys.issubset(matches[0]):
            sprite = pilot.repaired_back(None, write_artifacts=False) if mask == 'train_0.png' else base.cutout(mask)
            width = max(1, round(sprite.shape[1] * parameters['height'] / sprite.shape[0]))
            foot = (parameters['left'] + width / 2, parameters['top'] + parameters['height'])
            contact = recorded_contact_status(matches[0], foot,
                                              contact_width=_sign_contact_width(mask, parameters['height']))
            if contact != 'candidate':
                raise ValueError(f'Sign contact is {contact} in saved ground record')
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
    gains = TONE_GAINS[tone]
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
    gains = TONE_GAINS[tone]
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


def _generate(output, plan, run, prepared, sampling_seed=None, background_root=None, working=None):
    output = Path(output).resolve()
    export = output == (ROOT / 'datasets/exports/sign' / plan['target_split']).resolve()
    label_folder = 'labels' if export else 'labels_candidate'
    leftovers = [p for p in output.iterdir() if not (
        p.name in ('images', label_folder) and p.is_dir() and not any(p.iterdir()))] if output.exists() else []
    if output not in (OUTPUT_ROOT.resolve(), (OUTPUT_ROOT / plan['target_split']).resolve(), (ROOT/'datasets/exports/sign'/plan['target_split']).resolve()) or leftovers:
        raise ValueError('Candidate output must be the empty current sign sample directory')
    # Geometry is generated only after the shared guard allocates the plan.
    def render_one(state):
        row, foreground = prepared[state['id']]
        if state['target']['state'] == 'placed':
            validate_planned_stratum(state, [row])
        source_root = ROOT if row.get('workspace_image') is True else (background_root or BACKGROUNDS)
        source_image = source_root / row['image']
        frame = base.read_image(working.path(source_image) if working else source_image, cv2.IMREAD_COLOR)
        crop = state['parameters'].get('crop_xyxy') or row.get('crop_xyxy')
        if crop is not None:
            x1, y1, x2, y2 = crop
            frame = cv2.resize(frame[y1:y2, x1:x2], (640, 480), interpolation=cv2.INTER_LINEAR)
        elif frame.shape[1] * 9 == frame.shape[0] * 16:
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            frame = cv2.copyMakeBorder(frame, 60, 60, 0, 0, cv2.BORDER_CONSTANT)
        elif frame.shape[1] * 3 == frame.shape[0] * 4:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        if frame.shape != (480,640,3):
            rect=output_content_rectangle([frame.shape[1],frame.shape[0]])
            scaled=cv2.resize(frame,(rect[2]-rect[0],rect[3]-rect[1]),interpolation=cv2.INTER_AREA)
            frame=np.zeros((480,640,3),dtype=np.uint8)
            frame[rect[1]:rect[3],rect[0]:rect[2]]=scaled
        if frame.shape != (480, 640, 3):
            raise ValueError('Expected 640x480 BGR background')
        mask = state['target']['mask']
        if state['target']['state'] == 'negative':
            gains = TONE_GAINS[state['augmentation']['tone']]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= gains[0]
            hsv[:, :, 1] *= gains[1]
            image = frame.copy() if gains == (1.0,1.0) else cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            box = None
            geometry = {'value_gain': gains[0], 'saturation_gain': gains[1]}
        else:
            sprite = pilot.repaired_back(None, write_artifacts=False) if mask == 'train_0.png' else base.cutout(mask, working=working)
            image, box, geometry = compose(frame, sprite, state['target']['state'],
                                            state['augmentation']['tone'], None,
                                            row.get('road_anchor_center_x'), state['parameters'])
        retained = row.get('existing_sign_boxes', [])
        if retained and box is not None and any(
                min(box[2], item['output_box_xyxy'][2]) > max(box[0], item['output_box_xyxy'][0]) and
                min(box[3], item['output_box_xyxy'][3]) > max(box[1], item['output_box_xyxy'][1])
                for item in retained):
            raise ValueError('new sign overlaps a labelled background sign')
        return row,foreground,image,box,geometry
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    (output / label_folder).mkdir(exist_ok=True)
    result = {'status': 'running', 'image_size': [640, 480],
              'target_split': plan['target_split'], 'foreground_source': base.SIGN_VIDEO,
              'scope': 'shared parent sources; no independent capture or model quality claim',
              'quota_gap': plan.get('quota_gap'), 'samples': []}
    try:
        for state in plan['strata']:
            if state['planned_count'] == 0:
                continue
            row, foreground, image, box, geometry = render_one(state)
            for _ in range(state['planned_count']):
                name = f"sign_{len(result['samples']):05d}"
                image_path = output / 'images' / (name + '.png')
                label_path = output / label_folder / (name + '.txt')
                label = ''
                if box is not None:
                    x1, y1, x2, y2 = box
                    label = f'0 {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n'
                for existing in row.get('existing_sign_boxes', []):
                    x1, y1, x2, y2 = existing['output_box_xyxy']
                    label += f'0 {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n'
                write_png(image_path, image)
                label_path.write_text(label, encoding='utf8')
                actual = {'geometry': geometry, 'bbox_xyxy': box, 'background': state['sources'][0],
                          'background_review': state['background'],
                          'background_parent': row['parent_image'], 'background_time_seconds': row['pts_seconds'],
                          'working_background': working.actual(state['sources'][0]) if working else state['sources'][0],
                          'working_foreground_mask': working.actual(base.ANNOTATED / state['target']['mask'])
                          if working and foreground is not None and state['target']['mask'] != 'train_0.png' else None,
                          'foreground_mask': None if foreground is None else state['target']['mask'],
                          'foreground_source': None if foreground is None else foreground['source'],
                          'foreground_video': None if foreground is None else base.SIGN_VIDEO,
                          'view': 'none' if foreground is None else dict(VIEWS)[state['target']['mask']],
                          'state': state['target']['state'], 'tone': state['augmentation']['tone'],
                          'layout': state['parameters'].get('layout'),
                          'negative_context': state['parameters'].get('negative_context'),
                          'scene_evidence': state['parameters'].get('scene_evidence'),
                          'sampling_seed': sampling_seed,
                          'source_kind': 'direct_real' if row.get('use_role')=='direct_target' or state['parameters'].get('negative_context')=='direct_real' else 'composite',
                          'state_crop_xyxy':state['parameters'].get('crop_xyxy')}
                if row.get('existing_sign_boxes'):
                    actual['retained_background_signs'] = row['existing_sign_boxes']
                    if foreground is None:
                        actual['state'] = 'background_only'
                if state['target']['state'] == 'placed':
                    actual['background_sign_state'] = row['sign_absence']
                    actual['background_sign_absence_evidence'] = row['sign_absence_evidence']
                    if row.get('existing_sign_boxes'):
                        actual['retained_background_signs'] = row['existing_sign_boxes']
                    else:
                        actual['background_sign_absence'] = row['sign_absence']
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
                if len(result['samples'])%250==0:
                    print(f'{plan["target_split"]}: wrote {len(result["samples"])} images',flush=True)
        result['status'] = 'complete_candidates'
    except BaseException as exc:
        result.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def pool_plan_inputs(source_config, split='train'):
    """Adapt current sign-qualified source frames without redoing image review."""
    database = ROOT / source_config['databases']['frame_environment']
    frames = []
    for review in query_frames(task='sign', role='composite_background', split=split,
                               database=database):
        geometry = review['geometry']
        label = review['label']
        source_image = review.get('source_image')
        if not review.get('parent_video') and (source_image or '').startswith('datasets/annotated/'):
            source_image = next((path for path in review['paths']
                                 if path.startswith('datasets/raw/shared/backgrounds/') and (ROOT / path).is_file()), source_image)
        if source_image and not (ROOT / source_image).is_file():
            source_image = next((path for path in review['paths'] if (ROOT / path).is_file()), source_image)
        if (not source_image or source_image not in review['paths'] or
                geometry.get('coordinate_size') != review['image_size'] or
                label.get('presence') not in ('absent', 'present') or label.get('completeness') != 'complete' or
                label.get('scope', {}).get('kind') not in {'crop', 'frame'}):
            continue
        existing = existing_sign_boxes(review, geometry.get('crop_xyxy'))
        support, protected = pool_ground_rectangle(review)
        if label['presence'] == 'present' and label['scope']['kind'] == 'crop' and not existing:
            continue
        positive = bool(existing)
        parent_video = review.get('parent_video')
        if parent_video and not (ROOT / parent_video).is_file():
            relocations = json.loads((ROOT / 'datasets/raw_source_relocations.json').read_text(encoding='utf8'))
            for move in sorted(relocations['moves'], key=lambda x: -len(x['original_path'])):
                old = move['original_path'].rstrip('/')
                if parent_video == old or parent_video.startswith(old + '/'):
                    candidate = move['current_path'].rstrip('/') + parent_video[len(old):]
                    if (ROOT / candidate).is_file():
                        parent_video = candidate
                        break
        frames.append({'image': source_image, 'parent_image': source_image,
                       'parent_video': parent_video,
                       'pts_seconds': review.get('timestamp_seconds'),
                       'source_group': review['parent_group'], 'workspace_image': True,
                       'crop_xyxy': geometry.get('crop_xyxy'),
                       'sign_absence': 'existing_labelled' if positive else 'confirmed',
                       'sign_absence_evidence': f"frame_environment:{review['record_id']}",
                       **({'existing_sign_boxes': existing} if positive else {}),
                       'source_evidence': {'database': source_config['databases']['frame_environment'],
                                           'record_id': review['record_id']},
                       'ground_evidence': f"frame_environment:{review['record_id']}",
                       'support_rectangle_xyxy': support,
                       'protected_rectangles_xyxy': protected + [item['output_box_xyxy'] for item in existing],
                       'task_use_status': [{'task': 'sign', 'split': split, 'role': 'background',
                                            'status': 'eligible', 'evidence': review['record_id']}],
                       **background_review_reference(review)})
    if not frames:
        raise ValueError('Frame environment has no sign train backgrounds with placement and labels')
    return frames


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
    pool_mode = declaration.get('background_pool') is not None
    if pool_mode and declaration['background_pool'] != source_config.get('background_pool_manifest'):
        raise ValueError('Sign plan names a different shared background pool')
    if selection:
        selected_images = {row['image'] for row in declaration['background_inputs']} if pool_mode else set(selection['backgrounds'])
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
    if output_path not in (OUTPUT_ROOT.resolve(), (OUTPUT_ROOT / declaration['target_split']).resolve(),
                            (ROOT/'datasets/exports/sign'/declaration['target_split']).resolve()):
        raise ValueError('Sign output must be the sample root or a declared sample/export split directory')
    if pilot_only:
        raise ValueError('Legacy pilot mode does not accept production plans')
    custom = 'background_inputs' in declaration
    if custom:
        frames = declaration['background_inputs']
        if not isinstance(frames, list) or not frames:
            raise ValueError('Explicit background inputs must be a nonempty list')
        if len({row.get('image') for row in frames if isinstance(row, dict)}) != len(frames):
            raise ValueError('Sign plan background images must be unique')
        if any(state.get('target', {}).get('state') not in ('placed', 'negative') for state in declaration.get('strata', [])):
            raise ValueError('Explicit background inputs support placed and negative states only')
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
        if bind_background_reviews(frames, source_config, declaration['target_split']) != frames:
            raise ValueError('Sign plan background decisions differ from current ingest records; recompile plan')
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
    mask_paths = [base.ANNOTATED / name for name in sorted(names - {'none', 'train_0.png'})]
    working = WorkingInputs('sign', backgrounds={background_root / row['image'] for row in frames},
                            targets=mask_paths, split=declaration['target_split'])
    media += [base.ANNOTATED / name for name in sorted(names - {'none'})] + list(working.media)
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
    # The plan binds exact decision/evidence IDs. Hashing the entire shared
    # SQLite file would reject unrelated task writes; recheck our rows below.
    procedural.extend(working.evidence)
    if declaration.get('sample_recipe'):
        procedural.append(ROOT / declaration['sample_recipe'])
    if pool_mode:
        procedural.append(ROOT / declaration['background_pool'])
    if path is not None:
        procedural.append(path)
    if plan_path is not None:
        plan_path = Path(plan_path).resolve()
        if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
            raise ValueError('Plan must be an existing file inside the workspace')
        procedural.append(plan_path)
    with guarded('compose_v4_measured', output, media, procedural,
                 {'seed': seed, 'pilot_only': pilot_only,
                  'quota_gap': declaration.get('quota_gap')},
                 [Path(__file__), Path(base.__file__), Path(pilot.__file__),
                  ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py', WORKING_INPUTS_CODE,
                  ROOT / 'datasets/scripts/common/frame_environment.py',
                  ROOT / 'datasets/scripts/common/review/placement_record.py'],
                 production_plan=declaration, sample_output_paths=[Path(output) / 'images'],
                 replace_outputs=output_path.exists() and all(
                     p.name in ('images', 'labels_candidate','labels') and p.is_dir() and not any(p.iterdir())
                     for p in output_path.iterdir())) as run:
        if run.reused:
            return existing_result(output)
        result = _generate(output, run.plan, run, prepared, sampling_seed=seed,
                           background_root=background_root, working=working)
        if plan_path is not None:
            declared_by_id={state['id']:state for state in declaration['strata']}
            for sample in result['samples']:
                original=declared_by_id[sample['state_id']]
                sample['actual']['quota_group']=original.get('quota_group')
                if original.get('direct_decision_id'):
                    sample['actual']['direct_decision_id']=original['direct_decision_id']
            result.update(task='sign',split=declaration['target_split'],plan=Path(plan_path).relative_to(ROOT).as_posix(),
                          generation_plan=Path(plan_path).relative_to(ROOT).as_posix())
            (output_path/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        if custom and bind_background_reviews(frames, source_config, declaration['target_split']) != frames:
            raise ValueError('Selected sign background decisions changed during production')
        return result

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
