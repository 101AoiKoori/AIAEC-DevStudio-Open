"""Compile the explicit person scene batches for train, val and test.

The batch uses the current background dispatch, standee textures and vehicle
occluders. Background source groups are held out when enough exist.

Layouts are explicit per state: each state fixes background, identity, occlusion
state, tone, observation yaw and foot position, so one state is one image and the
batch has no duplicated layout signature.
"""

import argparse
import itertools
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'datasets/scripts/person/config'

DISPATCH = 'datasets/scripts/person/config/background_dispatch.json'
TEXTURES = 'datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json'
OCCLUDERS = 'datasets/annotated/person/v4_vehicle_occluders_20261113/manifest.json'
DIGITAL_WHITE_BACKS = 'datasets/annotated/common/sources/targets/person/white_back/manifest.json'

TONES = [
    {'value_gain': 1.0, 'saturation_gain': 1.0},
    {'value_gain': 1.1, 'saturation_gain': 0.72},
    {'value_gain': 0.88, 'saturation_gain': 0.82},
]
YAWS = [0, -5, 5]
HEIGHTS = [112, 128, 144, 160, 176, 192, 208, 224]
VEHICLE_HEIGHT_RATIO = 0.42
IDENTITY_REGION_RULE = 'geometric head-and-upper-body region x 10-70%, y 3-47% of the artwork'

# Local visible-floor rectangles for this known 640x360 background batch.
# These are conservative visual estimates, not a general floor segmentation
# model. The key of a 720p derivative resolves to its documented parent image.
# Each lower edge sits on visible floor beyond the base of nearby obstructions.
SAMPLE_FLOOR_BOUNDS = {
    'bg_00086_f00359.jpg': (0, 315, 639, 359),
    'bg_00146_f00749.jpg': (0, 235, 639, 359),
    'junction_003_005.jpg': (465, 270, 638, 357),
    'junction_005_001.jpg': (10, 280, 350, 357),
    'junction_005_017.jpg': (350, 300, 530, 357),
    'facades_009_001.jpg': (10, 280, 228, 357),
    'junction_003_011.jpg': (385, 255, 638, 357),
    'junction_003_004.jpg': (390, 285, 638, 357),
    'junction_003_006.jpg': (230, 305, 638, 357),
    'junction_003_010.jpg': (280, 275, 638, 357),
}


def sample_floor_polygon(row, polygon):
    """Intersect a recorded placement polygon with this batch's visible floor."""
    source = row.get('review_evidence', {}).get('source_image') or row['image']
    bounds = SAMPLE_FLOOR_BOUNDS.get(Path(source).name)
    if bounds is None:
        return polygon
    x0 = max(bounds[0], min(point[0] for point in polygon))
    y0 = max(bounds[1], min(point[1] for point in polygon))
    x1 = min(bounds[2], max(point[0] for point in polygon))
    y1 = min(bounds[3], max(point[1] for point in polygon))
    if x1 <= x0 or y1 <= y0:
        return []
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def background_splits(dispatch, *, use_declared=False):
    if not dispatch:
        raise ValueError('dispatch has no usable background image')
    if use_declared:
        splits = {'train': [], 'val': [], 'test': []}
        for index, row in enumerate(dispatch):
            if (row.get('state') in ('confirmed_empty', 'candidate_complete_negative',
                                     'complete_negative', 'complete_positive', 'unknown')
                    and row.get('split') in splits):
                splits[row['split']].append(index)
        return splits
    parents = {}
    for index, row in enumerate(dispatch):
        parent = row.get('source_video') or row.get('source_group') or row['image']
        parents.setdefault(parent, []).append(index)
    groups = list(parents.values())
    if len(groups) >= 3:
        return {'train': sorted(index for group in groups[:-2] for index in group),
                'val': groups[-2], 'test': groups[-1]}
    if len(groups) == 2:
        train, test = groups
        holdout = len(train) // 2
        return {'train': train[:holdout] + train[holdout + 1:] or train,
                'val': [train[holdout]], 'test': test}
    train = groups[0]
    return {'train': train[:-2] or train, 'val': train[-2:-1] or train, 'test': train[-1:]}


def png_size(path):
    """Read PNG dimensions from the IHDR chunk without decoding pixels."""
    with open(path, 'rb') as handle:
        head = handle.read(33)
    if head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        raise ValueError('expected a PNG source: ' + str(path))
    return int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')


def identity_region(width, height):
    x1 = max(1, round(0.10 * width))
    y1 = max(1, round(0.033 * height))
    x2 = max(x1 + 1, min(width, round(0.70 * width)))
    y2 = max(y1 + 1, min(height, round(0.467 * height)))
    return [x1, y1, x2, y2]


def sprite(width, height, target_height):
    target_height = int(target_height)
    target_width = max(2, round(target_height * width / height))
    return [min(target_width, 8192), target_height]


def contact_width(width):
    return max(8, round(width * 0.3))


def ground_top(polygon):
    return int(min(point[1] for point in polygon))


def placements(ground_polygon, ratio, count, offset):
    """Deterministic distinct foot placements inside the ground band."""
    top = max(ground_top(ground_polygon), 288)
    left = min(point[0] for point in ground_polygon)
    right = max(point[0] for point in ground_polygon)
    result, seen = [], set()
    for step in range(4000):
        if len(result) >= count:
            break
        index = step + offset * 17
        height = HEIGHTS[index % len(HEIGHTS)]
        width = max(2, round(height * ratio))
        if height > 359 or width > 620:
            continue
        low = max(top + 3, height + 1)
        high = 357
        if low > high:
            continue
        y = low + ((index * 29) % (high - low + 1))
        x_low = max(width // 2 + 2, left + width // 2)
        x_high = min(638 - width // 2, right - width // 2)
        if x_low >= x_high:
            continue
        x = x_low + ((index * 137) % (x_high - x_low + 1))
        key = (x, y, width, height)
        if key in seen:
            continue
        seen.add(key)
        result.append({'foot_xy': [int(x), int(y)], 'size_wh': [int(width), int(height)]})
    if len(result) < count:
        raise ValueError('ground band cannot hold the requested placements')
    return result


def pair_slot(ground_polygon, width_a, height_a, width_b, height_b, index):
    """One two-person placement with both sprites inside the frame."""
    top = max(ground_top(ground_polygon), 288)
    left = min(point[0] for point in ground_polygon)
    right = max(point[0] for point in ground_polygon)
    low = max(top + 3, max(height_a, height_b) + 1)
    high = 357
    if low > high:
        return None
    y = low + ((index * 19) % (high - low + 1))
    gap = 12
    total = width_a + width_b + gap
    x_low = max(total // 2 + 2, left + total // 2)
    x_high = min(638 - total // 2, right - total // 2)
    if x_low >= x_high:
        return None
    centre = x_low + ((index * 97) % (x_high - x_low + 1))
    x_a = centre - total // 2 + width_a // 2
    x_b = centre + total // 2 - width_b // 2
    return {'a': {'foot_xy': [int(x_a), int(y)], 'size_wh': [int(width_a), int(height_a)]},
            'b': {'foot_xy': [int(x_b), int(y)], 'size_wh': [int(width_b), int(height_b)]}}


def configured_foreground_manifest(role):
    """Select the registered manifest used by newly compiled person plans."""
    from datasets.scripts.common.config import load_task_config
    from datasets.scripts.person.planned_sources import role_paths

    _, source = load_task_config('person')
    paths = role_paths(source, role)
    if not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError(f'No valid person source manifest configured for {role}')
    return paths[0]


def white_back_textures():
    """Load registered white backs paired to their original front artwork.

    A separately registered single observed back is useful for other workflows,
    but cannot be matched to each synthetic front without canonical mappings.
    Preserve the existing digital-manifest selection when still registered.
    """
    from datasets.scripts.common.config import load_task_config
    from datasets.scripts.person.planned_sources import role_paths

    _, source = load_task_config('person')
    manifests = role_paths(source, 'white_back')
    if DIGITAL_WHITE_BACKS in manifests:
        manifests = [DIGITAL_WHITE_BACKS, *[path for path in manifests if path != DIGITAL_WHITE_BACKS]]
    result = {}
    for manifest in manifests:
        try:
            info = read_json(manifest)
        except FileNotFoundError:
            continue
        records = info.get('records', [])
        for index, row in enumerate(records):
            if not row.get('canonical_parent_image') or not row.get('foreground'):
                continue
            width, height = png_size(ROOT / row['foreground'])
            result.setdefault(row['canonical_parent_image'], (index, row['foreground'], width / height, manifest))
    return result


def build_sources(dispatch_manifest=DISPATCH):
    dispatch = read_json(dispatch_manifest)['records']
    texture_manifest = configured_foreground_manifest('foreground_texture')
    occluder_manifest = configured_foreground_manifest('vehicle_occluder')
    textures = read_json(texture_manifest)['records']
    occluders = [dict(row, _source_manifest=occluder_manifest) for row in read_json(occluder_manifest)['records']]
    identities = []
    for index, record in enumerate(textures):
        width, height = png_size(ROOT / record['image'])
        identities.append({
            'record_index': index,
            '_source_manifest': texture_manifest,
            'image': record['image'],
            'class': record['class'],
            'identity': Path(record['image']).stem.split('_', 1)[1],
            'ratio': width / height,
            'region': identity_region(width, height),
        })
    return dispatch, identities, occluders


def tone_at(index):
    return dict(TONES[index % len(TONES)])


def background_block(dispatch, record_index, dispatch_manifest=DISPATCH):
    record = dispatch[record_index]
    polygon = record['ground_polygon_640x360']
    reference = {
        'role': 'reviewed_background',
        'manifest': dispatch_manifest,
        'record_index': record_index,
        'image': record['image'],
        'source_state': record.get('state', 'unknown'),
    }
    if 'record_id' in record:
        reference.update(frame_id=record['frame_id'], record_id=record['record_id'],
                         content_sha256=record['content_sha256'])
    return reference, polygon


def person_actor(ident, placement, yaw, actor_id='person_added', category=None):
    manifest = ident.get('_source_manifest', TEXTURES)
    return {
        'id': actor_id,
        'kind': 'person',
        'source': ident['image'],
        'source_role': 'foreground_texture',
        'source_record': {'manifest': manifest, 'record_index': ident['record_index']},
        'view': 'front',
        'category': category or ident['class'],
        'identity': ident['identity'],
        'identity_evidence': '{}:record_index={}; {}'.format(manifest, ident['record_index'], IDENTITY_REGION_RULE),
        'identity_region_xyxy': list(ident['region']),
        'min_identity_visible': 0.9,
        'foot_xy': list(placement['foot_xy']),
        'contact_width': contact_width(placement['size_wh'][0]),
        'size_wh': list(placement['size_wh']),
        'angles_degrees': {'yaw': yaw, 'pitch': 0, 'roll': 0},
    }


def vehicle_actor(occluder, record_index, ground_polygon, person_placement, slot):
    person_height = person_placement['size_wh'][1]
    vehicle_height = max(2, round(person_height * VEHICLE_HEIGHT_RATIO))
    width, height = png_size(ROOT / occluder['cropped_image'])
    vehicle_width = max(2, round(vehicle_height * width / height))
    top = ground_top(ground_polygon)
    x = person_placement['foot_xy'][0] + 2 + (slot % 3) * 3
    x = min(max(x, vehicle_width // 2 + 2), 638 - vehicle_width // 2)
    y = max(top + 3, person_placement['foot_xy'][1] - 2)
    return {
        'id': 'vehicle_front',
        'kind': 'vehicle',
        'source': occluder['cropped_image'],
        'source_role': 'vehicle_occluder',
        'source_record': {'manifest': occluder.get('_source_manifest', OCCLUDERS), 'record_index': record_index},
        'category': occluder['class'],
        'foot_xy': [int(x), int(y)],
        'contact_width': max(8, min(vehicle_width, round(vehicle_width * 0.3))),
        'size_wh': [int(vehicle_width), int(vehicle_height)],
        'angles_degrees': {'yaw': 0, 'pitch': 0, 'roll': 0},
    }


def make_state(state_id, failure_event, background, polygon, actors, tone, overlaps):
    return {
        'id': state_id,
        'failure_event': failure_event,
        'min_count': 1,
        'max_count': 1,
        'prior': {'alpha': 1, 'beta': 1},
        'exposure_weight': 1,
        'failure_cost': 1,
        'conditions': [],
        'background': background,
        'target': {'actors': actors},
        'labels': {'person': 'existing_and_new_visible_yolo', 'white_back': 'not_labeled'},
        'augmentation': tone,
        'parameters': {
            'resolution': [640, 360],
            'ground_polygon': [list(point) for point in polygon],
            'obstacle_polygons': [],
            'required_overlaps': overlaps,
        },
    }


def distribute(total, buckets):
    """Split an integer total across buckets as evenly as possible."""
    base, extra = divmod(total, buckets)
    return [base + (1 if index < extra else 0) for index in range(buckets)]


def allocate_identities(identities, states_per_combo, combo_index):
    """Round-robin identities so every identity appears in every background and occlusion state."""
    ordered = [identities[(combo_index + step) % len(identities)] for step in range(len(identities))]
    return [ordered[index % len(ordered)] for index in range(states_per_combo)]


def standalone_vehicle(occluder, record_index, ground_polygon, index):
    """Place an electric vehicle without any person: a zero-person negative frame."""
    width, height = png_size(ROOT / occluder['cropped_image'])
    if occluder['class'] == 'upright':
        target_height = 70 + (index % 5) * 10
    else:
        target_height = 44 + (index % 4) * 8
    target_width = max(2, round(target_height * width / height))
    top = ground_top(ground_polygon)
    low = max(top + 4, target_height + 1)
    high = 356
    if low > high:
        raise ValueError('ground band cannot hold a standalone vehicle')
    y = low + ((index * 23) % (high - low + 1))
    x_low = target_width // 2 + 2
    x_high = 638 - target_width // 2
    if x_low >= x_high:
        raise ValueError('standalone vehicle wider than the frame')
    x = x_low + ((index * 131) % (x_high - x_low + 1))
    return {
        'id': 'vehicle_front',
        'kind': 'vehicle',
        'source': occluder['cropped_image'],
        'source_role': 'vehicle_occluder',
        'source_record': {'manifest': occluder.get('_source_manifest', OCCLUDERS), 'record_index': record_index},
        'category': occluder['class'],
        'foot_xy': [int(x), int(y)],
        'contact_width': max(8, min(target_width, round(target_width * 0.3))),
        'size_wh': [int(target_width), int(target_height)],
        'angles_degrees': {'yaw': 0, 'pitch': 0, 'roll': 0},
    }


def compile_split(split, total, backgrounds, identities, occluders, *, multi_share, empty_share,
                  occluded_count=None):
    dispatch, _, _ = build_sources()
    states = []
    counter = itertools.count(1)
    occluded = (750 if split == 'train' else 200) if occluded_count is None else occluded_count
    singles = total - multi_share - empty_share
    quotas = {'none': singles - occluded, 'upright': occluded // 2, 'fallen': occluded // 2}

    # Zero-person negatives. Half are bare backgrounds, half carry an unlabelled
    # electric vehicle so the detector also sees the shape without a person.
    per_background = distribute(empty_share, len(backgrounds))
    for background_slot, background_count in enumerate(per_background):
        record_index = backgrounds[background_slot]
        background, polygon = background_block(dispatch, record_index)
        per_tone = distribute(background_count, len(TONES))
        for tone_index, tone_count in enumerate(per_tone):
            if tone_count == 0:
                continue
            combo_index = record_index * 7 + tone_index
            for position in range(tone_count):
                actors = []
                if position % 2 == 1:
                    record = (combo_index + position) % len(occluders)
                    actors = [standalone_vehicle(occluders[record], record, polygon,
                                                 combo_index * 31 + position)]
                number = next(counter)
                states.append(make_state(
                    '{}_negative_{:05d}_{}_t{}'.format(split, number,
                                                       Path(background['image']).stem, tone_index),
                    'empty', background, polygon, actors, tone_at(tone_index), []))

    for kind in ('none', 'upright', 'fallen'):
        quota = quotas[kind]
        if quota <= 0:
            continue
        per_background = distribute(quota, len(backgrounds))
        for background_slot, background_count in enumerate(per_background):
            record_index = backgrounds[background_slot]
            background, polygon = background_block(dispatch, record_index)
            per_tone = distribute(background_count, len(TONES))
            for tone_index, tone_count in enumerate(per_tone):
                if tone_count == 0:
                    continue
                combo_index = record_index * 3 + tone_index
                chosen = allocate_identities(identities, tone_count, combo_index)
                slots = placements(polygon, identities[0]['ratio'], 16, combo_index)
                for position, ident in enumerate(chosen):
                    yaw = YAWS[(combo_index + position) % len(YAWS)]
                    placement = dict(slots[position % len(slots)])
                    height = placement['size_wh'][1]
                    width = png_size(ROOT / ident['image'])[0]
                    full_height = png_size(ROOT / ident['image'])[1]
                    placement['size_wh'] = sprite(width, full_height, height)
                    x = placement['foot_xy'][0]
                    half = placement['size_wh'][0] // 2
                    placement['foot_xy'][0] = int(min(max(x, half + 2), 638 - half))
                    actors = [person_actor(ident, placement, yaw)]
                    overlaps = []
                    if kind != 'none':
                        pool = [index for index, row in enumerate(occluders) if row['class'] == kind]
                        record = pool[(combo_index + position) % len(pool)]
                        actors.append(vehicle_actor(occluders[record], record, polygon, placement, position))
                        overlaps = [['person_added', 'vehicle_front']]
                    number = next(counter)
                    states.append(make_state(
                        '{}_{:05d}_{}_{}_{}_t{}'.format(split, number, kind, ident['identity'],
                                                        Path(background['image']).stem, tone_index),
                        'person_{}'.format(kind), background, polygon, actors, tone_at(tone_index), overlaps))

    if multi_share:
        pairs = list(itertools.combinations(range(len(identities)), 2))
        per_background = distribute(multi_share, len(backgrounds))
        for background_slot, background_count in enumerate(per_background):
            record_index = backgrounds[background_slot]
            background, polygon = background_block(dispatch, record_index)
            per_tone = distribute(background_count, len(TONES))
            for tone_index, tone_count in enumerate(per_tone):
                if tone_count == 0:
                    continue
                combo_index = record_index * 5 + tone_index
                for position in range(tone_count):
                    first, second = pairs[(combo_index * 7 + position) % len(pairs)]
                    a, b = identities[first], identities[second]
                    height_a = HEIGHTS[(combo_index + position) % len(HEIGHTS)]
                    height_b = max(2, round(height_a * 0.94))
                    width_a_full, height_a_full = png_size(ROOT / a['image'])
                    width_b_full, height_b_full = png_size(ROOT / b['image'])
                    size_a = sprite(width_a_full, height_a_full, height_a)
                    size_b = sprite(width_b_full, height_b_full, height_b)
                    slot = pair_slot(polygon, size_a[0], size_a[1], size_b[0], size_b[1],
                                     combo_index * 11 + position)
                    if slot is None:
                        raise ValueError('ground band cannot hold the pair placement at ' + background['image'])
                    actors = [
                        person_actor(a, slot['a'], YAWS[(position + combo_index) % len(YAWS)], 'person_front'),
                        person_actor(b, slot['b'], YAWS[(position + 1 + combo_index) % len(YAWS)], 'person_back'),
                    ]
                    number = next(counter)
                    states.append(make_state(
                        '{}_{:05d}_multi_{}_{}_{}_t{}'.format(split, number, a['identity'], b['identity'],
                                                              Path(background['image']).stem, tone_index),
                        'person_multi', background, polygon, actors, tone_at(tone_index), []))
    if len(states) != total:
        raise ValueError('{} 状态数 {} 与目标 {} 不一致'.format(split, len(states), total))
    return states


def attach_sources(declaration):
    from datasets.scripts.person.planned_sources import scene_sources

    for state in declaration['strata']:
        one = dict(declaration, budget=1, strata=[state])
        media, procedural = scene_sources(one)
        state['sources'] = sorted(media)
        state['_procedural'] = sorted(procedural)


def build(split, total, backgrounds, multi_share, empty_share, occluded_count=None):
    dispatch, identities, occluders = build_sources()
    states = compile_split(split, total, backgrounds, identities, occluders,
                           multi_share=multi_share, empty_share=empty_share, occluded_count=occluded_count)
    declaration = {
        'schema_version': 1,
        'model_key': 'person',
        'target_split': split,
        'purpose': 'Full person batch compiled from reviewed backgrounds, standee textures and vehicle occluders.',
        'budget': len(states),
        'uncertainty_weight': 0,
        'independence_basis': 'Background parent groups are held out when available; check the listed sources for shared parents across splits. Foreground sources are shared.',
        'constraints': {'foreground': 'planned_actors', 'identity': 'front_visible',
                        'background': 'complete_existing_or_empty'},
        'design': {
            'backgrounds': [dispatch[index]['image'] for index in backgrounds],
            'identities': [ident['image'] for ident in identities],
            'tones': TONES,
            'vehicle_height_ratio': VEHICLE_HEIGHT_RATIO,
            'identity_region_rule': IDENTITY_REGION_RULE,
            'multi_person_states': multi_share,
            'zero_person_negative_states': empty_share,
            'note': 'Layouts are explicit per state; identity regions are a geometric design rule, not measured features.',
        },
        'strata': states,
    }
    attach_sources(declaration)
    for state in declaration['strata']:
        state.pop('_procedural', None)
    return declaration


def build_sample(split='train', budget=100, background_indices=None, *, validate_selected=True,
                 background_scope='recorded_background_scope', min_placed_fraction=0,
                 directed_train=False, layout_rounds=None, seed=17, dispatch_manifest=None):
    """Build feasible explicit scenes, then let the shared allocator select them."""
    from lab.training_common.bayesian_allocation import plan_generation
    from datasets.scripts.common.review.ground_space import GroundSpace
    from datasets.scripts.common.review.placement import assess_placeable_ground
    from datasets.scripts.person.planned_sources import (configured_background_dispatch, recorded_contact,
                                                         recorded_ground, verify_current_backgrounds)
    import cv2
    import numpy as np

    targets = yaml.safe_load((ROOT / 'config/task_targets.yaml').read_text(encoding='utf-8'))
    axes_config = yaml.safe_load((ROOT / 'config/combination_axes.yaml').read_text(encoding='utf-8'))
    counts = targets['tasks']['person']['generation']['person_counts']
    yaw_angles = axes_config['tasks']['person']['generation']['yaw_degrees']
    lighting = axes_config['generation_defaults']['lighting']
    light_names = lighting['priority']
    dispatch_manifest, environment_backed = configured_background_dispatch(dispatch_manifest)
    dispatch, identities, occluders = build_sources(dispatch_manifest)
    noncommunity = [item for item in identities if item['class'] == 'noncommunity']
    if background_indices is None and read_json(dispatch_manifest).get('split_assignment') == 'source_pairing':
        background_indices = background_splits(dispatch, use_declared=True)[split]
    texture_records = read_json(configured_foreground_manifest('foreground_texture'))['records']
    states = []
    rng = random.Random(seed)
    issue_record = yaml.safe_load((CONFIG / 'sample_issues.yaml').read_text(encoding='utf-8')) or {}
    excluded = {item['image'] for item in issue_record.get('exclude_backgrounds', [])}
    indices = range(len(dispatch)) if background_indices is None else background_indices
    indices = [index for index in indices if dispatch[index]['image'] not in excluded]
    if not indices:
        raise ValueError(f'no background candidates available for {split}')
    # Draw fresh layouts for a large batch, allowing for narrow floor polygons
    # and rejected overlaps. Lighting variation alone does not add a layout.
    if layout_rounds is None:
        layout_rounds = max(1, math.ceil(budget / (15 * len(indices))))
    for background_index in indices:
        row = dispatch[background_index]
        background, polygon = background_block(dispatch, background_index, dispatch_manifest)
        saved_ground = recorded_ground(row)
        horizon = None
        if saved_ground:
            placement_record, ground_assessment = saved_ground
            source_width, source_height = row['source_image_size']
            if placement_record.get('reference_quad_px') is None:
                continue
            polygon = [[max(0, min(639, round((x + .5) * 640 / source_width - .5))),
                        max(0, min(359, round((y + .5) * 360 / source_height - .5)))]
                       for x, y in placement_record['reference_quad_px']]
            if placement_record.get('horizon_y') is not None:
                horizon = (placement_record['horizon_y'] + .5) * 360 / source_height - .5
        else:
            polygon = sample_floor_polygon(row, polygon)
        if not polygon:
            continue
        ground_space = GroundSpace(polygon, image_shape=(360, 640), far_scale=0.70, horizon_y=horizon)
        if not saved_ground:
            # Historical dispatch records used a batch-local pixel assessment.
            image = cv2.imdecode(np.fromfile(str(ROOT / row['image']), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('cannot decode background ' + row['image'])
            source_height, source_width = image.shape[:2]
            image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
            source = row.get('review_evidence', {}).get('source_image') or row['image']
            floor_mask = None
            if Path(source).name in SAMPLE_FLOOR_BOUNDS or row.get('placement_region_basis'):
                floor_mask = np.zeros((360, 640), dtype=np.uint8)
                cv2.fillPoly(floor_mask, [np.asarray(polygon, dtype=np.int32)], 1)
            known_boxes = []
            for box in [*row.get('boxes', []), *row.get('white_back_reference_boxes', [])]:
                coordinates = box.get('xyxy', box.get('box')) if isinstance(box, dict) else box
                if coordinates is not None and len(coordinates) >= 4:
                    x1, y1, x2, y2 = coordinates[:4]
                    known_boxes.append({'xyxy': [x1 * 640 / source_width, y1 * 360 / source_height,
                                                  x2 * 640 / source_width, y2 * 360 / source_height]})
            ground_assessment = assess_placeable_ground(image, ground_space.polygon,
                                                         boxes=known_boxes, floor_mask=floor_mask)
        if ground_assessment.status != 'candidate':
            continue
        left = max(1, min(point[0] for point in polygon))
        right = min(638, max(point[0] for point in polygon))
        bottom = min(max(point[1] for point in polygon) - 3, 354)
        # The polygon is the currently available ground estimate; depth is a
        # bounded image-space approximation, not a calibrated 3-D surface.
        top = max(ground_top(polygon) + 3, bottom - 130)
        if bottom <= top + 8:
            continue
        mid_x = (left + right) / 2
        depth_rows = range(top, bottom - 2)
        if saved_ground:
            far_x, far_y = ((polygon[0][axis] + polygon[1][axis]) / 2 for axis in (0, 1))
            near_x, near_y = ((polygon[2][axis] + polygon[3][axis]) / 2 for axis in (0, 1))
            # The centre of a slanted ground quad moves with image row.
            scales = {y: ground_space.perspective_scale((far_x + (near_x - far_x) * (y - far_y) / (near_y - far_y), y))
                      for y in depth_rows if far_y <= y <= near_y and near_y > far_y}
            depth_rows = [y for y, scale in scales.items() if scale is not None]
        if len(depth_rows) < max(counts):
            continue
        if not directed_train:
            source_width, source_height = row.get('source_image_size', [640, 360])
        is_four_three = source_width * 3 == source_height * 4
        min_placed_height = max(2, math.ceil(min_placed_fraction * (480 if is_four_three else 360)))
        shifted_polygon = [[x, round(y * 4 / 3) if is_four_three else y + 60] for x, y in polygon]
        for count in counts:
            for angle in yaw_angles:
                for light_name, layout_index in itertools.product(light_names, range(layout_rounds)):
                    for attempt in range(16):
                        chosen = rng.sample(identities, count)
                        aim_for_noncommunity = directed_train and noncommunity and rng.random() < 0.5
                        if aim_for_noncommunity and all(
                            item['class'] == 'community' for item in chosen
                        ):
                            chosen[rng.randrange(count)] = rng.choice(noncommunity)
                        # One axis value anchors allocation while each other person
                        # independently receives a feasible standee orientation.
                        angles = [angle] + [rng.choice(yaw_angles) for _ in range(count - 1)]
                        if len(set(angles)) == 1:
                            angles[-1] = rng.choice([value for value in yaw_angles if value != angle])
                        ys = rng.sample(depth_rows, count)
                        base_height = rng.choice((190, 210, 230))
                        relative = ([scales[y] for y in ys] if saved_ground else
                                    [ground_space.perspective_scale((mid_x, y)) for y in ys])
                        if any(scale is None for scale in relative):
                            continue
                        heights = [max(min_placed_height, round(base_height * scale * rng.uniform(0.92, 1.08)))
                                   for scale in relative]
                        widths = [max(8, round(h * item['ratio'] * abs(math.cos(math.radians(yaw)))))
                                  for item, h, yaw in zip(chosen, heights, angles)]
                        # Ground support constrains foot contacts, not the full
                        # width of a standing figure above the floor. Allow people
                        # at different depths to overlap horizontally.
                        contacts = [contact_width(width) for width in widths]
                        free = right - left - sum(contacts) - (count + 1) * 5
                        if free < 0:
                            continue
                        # Random gaps and a shuffled actor order avoid always placing
                        # the same identity at the same horizontal position.
                        cuts = [0] + sorted(rng.randrange(free + 1) for _ in range(count)) + [free]
                        gaps = [cuts[index + 1] - cuts[index] for index in range(count + 1)]
                        order = list(range(count))
                        rng.shuffle(order)
                        x = left + 5 + gaps[0]
                        actors = []
                        for slot, number in enumerate(order):
                            width, height = widths[number], heights[number]
                            contact = contacts[number]
                            x += contact / 2
                            # Full sprites must fit the image even when the floor
                            # region is closer to the left or right edge.
                            actor_x = min(max(x, width / 2 + 1), 638 - width / 2)
                            local = ground_space.to_local((actor_x, ys[number]))
                            if local is None:
                                actors = []
                                break
                            footpoint = ground_space.footpoint(local, contact_width=contact)
                            if footpoint is None:
                                actors = []
                                break
                            output_y = round(footpoint[1] * 4 / 3) if is_four_three else round(footpoint[1] + 60)
                            placement = {'foot_xy': [round(footpoint[0]), output_y], 'size_wh': [width, height]}
                            actor = person_actor(chosen[number], placement, angles[number], f'person_{number + 1}')
                            # Preserve a meaningful amount of the existing upper-body
                            # proxy and separately the top quarter of the foreground.
                            # Both are geometric approximations, not face recognition.
                            actor['min_identity_visible'] = 0.70
                            actor['min_upper_visible'] = 0.80
                            # Unoccluded people are allowed; when occlusion occurs,
                            # bound the visible-mask loss rather than bounding-box IoU.
                            actor['occlusion_fraction_range'] = [0.02, 0.55]
                            actors.append(actor)
                            x += contact / 2 + 5 + gaps[slot + 1]
                        if not actors:
                            continue
                        vehicle_variant = rng.choice(('none', 'upright_left', 'upright_right', 'fallen'))
                        vehicle_kind = 'upright' if vehicle_variant.startswith('upright_') else vehicle_variant
                        overlaps = []
                        if vehicle_kind != 'none':
                            matching = [index for index, item in enumerate(occluders) if item['class'] == vehicle_kind]
                            vehicle_index = rng.choice(matching)
                            # Put the vehicle slightly closer than its selected person;
                            # paint ordering below follows ground contact depth.
                            occlusion_targets = ([actor for actor in actors if actor['category'] == 'noncommunity']
                                                 if aim_for_noncommunity else actors)
                            person = min(occlusion_targets, key=lambda actor: actor['foot_xy'][1])
                            placement = {'foot_xy': person['foot_xy'], 'size_wh': person['size_wh']}
                            vehicle = vehicle_actor(occluders[vehicle_index], vehicle_index, shifted_polygon, placement, 0)
                            # An upright bicycle occupies roughly the lower two
                            # thirds of the person; its side offset varies, instead
                            # of always stacking its centre over the person's feet.
                            if vehicle_kind == 'upright':
                                vehicle_height = round(person['size_wh'][1] * rng.uniform(0.58, 0.70))
                                vehicle_src_width, vehicle_src_height = png_size(ROOT / occluders[vehicle_index]['cropped_image'])
                                vehicle_width = max(2, round(vehicle_height * vehicle_src_width / vehicle_src_height))
                                vehicle['size_wh'] = [vehicle_width, vehicle_height]
                                vehicle['contact_width'] = max(8, min(vehicle_width, round(vehicle_width * 0.3)))
                                side = -1 if vehicle_variant == 'upright_left' else 1
                                offset = max(3, round((vehicle_width + person['size_wh'][0]) * 0.16))
                                vehicle['foot_xy'][0] = min(max(person['foot_xy'][0] + side * offset,
                                                                 vehicle_width // 2 + 2), 638 - vehicle_width // 2)
                            vehicle['foot_xy'][1] = person['foot_xy'][1] + rng.randint(3, 8)
                            actors.append(vehicle)
                            overlaps.append([person['id'], vehicle['id']])
                        # Coordinates in GroundSpace refer to the original 640x360
                        # frame, before 4:3 fitting / letterboxing to 640x480.
                        def original_foot(actor):
                            x, y = actor['foot_xy']
                            return (x, y * 3 / 4 if is_four_three else y - 60)

                        # A candidate is usable only if each complete ground contact
                        # is supported and clear of known or suspected obstacles.
                        if saved_ground:
                            contact_checks = (recorded_contact(ground_assessment, actor, (640, 480)) for actor in actors)
                        else:
                            contact_checks = (ground_assessment.contact_status(original_foot(actor),
                                              contact_width=actor['contact_width']) for actor in actors)
                        if any(status != 'candidate' for status in contact_checks):
                            continue
                        depths = {actor['id']: ground_space.relative_depth(original_foot(actor)) for actor in actors}
                        if any(depth is None for depth in depths.values()):
                            continue
                        actors.sort(key=lambda actor: depths[actor['id']])
                        ground_records = []
                        for actor in actors:
                            point = original_foot(actor)
                            local = ground_space.to_local(point)
                            ground_records.append({
                                'actor_id': actor['id'],
                                'local_xy': [round(value, 6) for value in local],
                                'relative_depth': round(depths[actor['id']], 6),
                                'perspective_scale': round(ground_space.perspective_scale(point), 6),
                                'local_near_axis_image_heading_degrees': round(ground_space.image_heading(local, 90), 3),
                            })
                        gain_min, gain_max = lighting['image_value_gain'][light_name]
                        tone = {'value_gain': round(rng.uniform(gain_min, gain_max), 3),
                                'saturation_gain': rng.choice((0.65, 0.85, 1.0, 1.15))}
                        state_id = f'sample_{background_index:03d}_{count}_{angle:+d}_{light_name}'
                        if layout_index:
                            state_id += f'_{layout_index}'
                        state = make_state(
                            state_id,
                            'person_multi', background, shifted_polygon, actors, tone, overlaps)
                        state['min_count'] = 0
                        state['parameters']['ground_placement'] = {
                            'basis': 'review ground quad with configured far/near scale; local near axis is not sprite yaw',
                            'source_frame_resolution': [640, 360], 'floor_support': ground_assessment.basis,
                            'contact_status': 'candidate', 'actors': ground_records,
                        }
                        if saved_ground:
                            state['parameters']['ground_placement']['recorded_algorithm'] = placement_record['algorithm_version']
                        state['parameters']['resolution'] = [640, 480]
                        state['axes'] = {'person_count': count, 'yaw': angle, 'lighting': light_name,
                                         'background': row.get('source_group') or row['image'],
                                         'background_image': row['image'],
                                         'vehicle_occlusion': vehicle_variant,
                                         'source_resolution': f'{source_width}x{source_height}',
                                         'class_composition': ('both' if len({item['class'] for item in chosen}) > 1
                                                               else chosen[0]['class'])}
                        state['exposure_weight'] = len(light_names) - light_names.index(light_name)
                        sources = {row['image']}
                        for name in ('source_video', 'parent_image'):
                            if row.get(name):
                                sources.add(row[name])
                        for actor in actors:
                            sources.add(actor['source'])
                            record_index = actor['source_record']['record_index']
                            record = occluders[record_index] if actor['kind'] == 'vehicle' else texture_records[record_index]
                            for name in ('artwork', 'path', 'source_frame', 'source_video'):
                                if record.get(name):
                                    sources.add(record[name])
                        state['sources'] = sorted(sources)
                        states.append(state)
                        break
    if not states:
        raise ValueError('no feasible person sample scenes')
    random.Random(seed).shuffle(states)
    states = states[:10000]
    declaration = {
        'schema_version': 1, 'model_key': 'person', 'target_split': split,
        'purpose': 'Person sample scenes selected by conditional balancing of feasible backgrounds, counts and standee angles.',
        'budget': min(budget, len(states)), 'uncertainty_weight': 0,
        'independence_basis': 'Sample backgrounds retain their source groups; foregrounds are flat standee artworks.',
        'constraints': {'foreground': 'planned_actors', 'identity': 'front_visible',
                        'background': background_scope},
        'conditional_balance': {'axes': ['person_count', 'yaw', 'lighting', 'background', 'background_image', 'vehicle_occlusion',
                                         'source_resolution', 'class_composition'],
                                'stratify_by': 'person_count', 'secondary_stratify_by': 'background_image',
                                'prior': 1, 'seed': seed},
        'design': {'counts': list(counts), 'standee_yaw_degrees': list(yaw_angles),
                   'lighting': light_names, 'resolution': [640, 480],
                   'background_fit': 'fit 4:3 to 640x480, or pad 640x360 content with 60 pixels per vertical edge',
                   'view_basis': 'front-source flat standee rotated in plane projection, not a real side-view portrait',
                   'placement_basis': 'GroundSpace on recorded ground polygon clipped to local visible-floor estimates; configured far scale 0.70, not calibrated 3-D',
                   'occlusion': {'fraction_range_when_occluded': [0.02, 0.55],
                                 'min_identity_region_visible': 0.70,
                                 'min_upper_foreground_visible': 0.80,
                                 'upper_foreground_basis': 'top quarter of original foreground; geometric proxy, not semantic face detection'}},
        'strata': states,
    }
    # Validate only selected scenes. A candidate that cannot be composed is
    # removed and the same allocator chooses a replacement from the pool.
    failures = []
    validated_ids = set()
    while validate_selected:
        selected = [state for state in plan_generation(declaration, ROOT)['strata'] if state['planned_count']]
        pending = [state for state in selected if state['id'] not in validated_ids]
        if not pending:
            break
        errors = validate(dict(declaration, strata=pending), limit=None)
        failed = {item['state_id'] for item in errors}
        validated_ids.update(state['id'] for state in pending if state['id'] not in failed)
        if not errors:
            break
        failures.extend(errors)
        declaration['strata'] = [state for state in declaration['strata'] if state['id'] not in failed]
        declaration['budget'] = min(declaration['budget'], len(declaration['strata']))
        if not declaration['strata']:
            raise ValueError('no candidate scene could be composed')
    if validate_selected and environment_backed:
        verify_current_backgrounds(selected, split)
    return declaration, failures


def _has_white_contrast(base, class_name, white_textures):
    actors = base['target']['actors']
    front = next((actor for actor in actors if actor['category'] == class_name), None)
    return bool(front and any(actor['id'] != front['id'] and actor['source'] in white_textures for actor in actors))


def _contrast_states(base, class_name, white_textures):
    """Vary the people while preserving the base scene, lighting and placement."""
    original = base['target']['actors']
    front = next(actor for actor in original if actor['category'] == class_name)
    other = next(actor for actor in original if actor['id'] != front['id'] and actor['source'] in white_textures)
    record_index, white_source, white_ratio, manifest = white_textures[other['source']]
    height = other['size_wh'][1]
    width = max(8, round(height * white_ratio))
    back = {
        'id': other['id'], 'kind': 'person', 'view': 'back', 'category': None, 'identity': None,
        'source': white_source, 'source_role': 'white_back',
        'source_record': {'manifest': manifest, 'record_index': record_index},
        'foot_xy': other['foot_xy'], 'size_wh': [width, height],
        'contact_width': contact_width(width),
        'angles_degrees': {'yaw': 0, 'pitch': 0, 'roll': 0},
    }
    mixed = [front if actor['id'] == front['id'] else back for actor in original]
    for scenario, actors in (('empty', []), ('person_single', [front]),
                             ('white_back_hard_negative', [back]), ('person_with_white_back', mixed)):
        ids = {actor['id'] for actor in actors}
        ground = base['parameters']['ground_placement']
        params = {**base['parameters'], 'ground_placement': {
            **ground, 'actors': [item for item in ground['actors'] if item['actor_id'] in ids],
        }}
        yield {**base, 'id': f'{base["id"]}_{scenario}', 'failure_event': scenario,
               'min_count': 1, 'target': {'actors': actors}, 'parameters': params,
               'axes': {**base['axes'], 'coverage_scene': scenario,
                        'person_count': int(front['id'] in ids),
                        'yaw': front['angles_degrees']['yaw'] if front['id'] in ids else None,
                        'class_composition': class_name if front['id'] in ids else 'empty'}}


def build_supervised_sample(background_indices, *, contrasts_per_class=4, multi_count=20, dispatch_manifest=None):
    """Make matched scenes in a small supervised sample alongside ordinary groups."""
    from lab.training_common.bayesian_allocation import plan_generation
    from datasets.scripts.person.planned_sources import (complete_background, configured_background_dispatch,
                                                         verify_current_backgrounds)

    dispatch_manifest, environment_backed = configured_background_dispatch(dispatch_manifest)
    rows = read_json(dispatch_manifest)['records']
    if any(not 0 <= index < len(rows) or rows[index].get('boxes') or
           not complete_background(rows[index], rows[index]['state'].removeprefix('candidate_'), 'train')
           for index in background_indices):
        raise ValueError('supervised contrasts require complete, person-empty train backgrounds')
    dataset = yaml.safe_load((CONFIG / 'dataset.yaml').read_text(encoding='utf-8'))
    placement_fraction = dataset['target_spec']['minimum_placed_height_fraction']
    batch, failures = build_sample(budget=multi_count, background_indices=background_indices,
                                   background_scope='task_complete', min_placed_fraction=placement_fraction,
                                   dispatch_manifest=dispatch_manifest)
    white_textures = white_back_textures()
    layout_pools = {}
    for state in batch['strata']:
        if state['axes']['person_count'] == 2 and state['axes']['vehicle_occlusion'] == 'none':
            layout_pools.setdefault(state['background']['image'], []).append(state)

    contrasts = []
    taken = set()
    for class_name in ('community', 'noncommunity'):
        selected = 0
        while selected < contrasts_per_class:
            before = selected
            for pool in layout_pools.values():
                for base in pool:
                    if base['id'] in taken or not _has_white_contrast(base, class_name, white_textures):
                        continue
                    taken.add(base['id'])
                    variants = list(_contrast_states(base, class_name, white_textures))
                    candidate = dict(batch, budget=len(variants), strata=variants)
                    attach_sources(candidate)
                    if validate(candidate, limit=1):
                        continue
                    contrasts.extend(variants)
                    selected += 1
                    break
                if selected == contrasts_per_class:
                    break
            if selected == before:
                raise ValueError(f'could only locate {selected} matched contrast layouts for {class_name}')

    multi = [dict(state, min_count=1) for state in plan_generation(batch, ROOT)['strata'] if state['planned_count']]
    for state in [*contrasts, *multi]:
        state.pop('_procedural', None)
        state.pop('planned_count', None)
    batch.update(strata=[*contrasts, *multi], budget=len(contrasts) + len(multi),
                 purpose='Matched pure empty, single front, white back, mixed front/back, and multi-person scenes.')
    batch.pop('conditional_balance')
    if environment_backed:
        verify_current_backgrounds(batch['strata'], 'train')
    return batch, failures


def _compose_planned_state(declaration, state, *, historical_replay=False):
    from datasets.scripts.person import blender_existing
    from datasets.scripts.person.compose_planned_ground_person import prepare
    from datasets.scripts.person.person_scene_plan import compose_scene

    item = prepare(dict(declaration, budget=1, strata=[state]), historical_replay=historical_replay)
    if state['background'].get('role') == 'blender_existing':
        return blender_existing.produce_scene(item, state)
    return compose_scene(item['background_image'], item['scene'], item['foregrounds'], item['identity_masks'])


def _person_coverage(result):
    """Identify occluders from composited frontmost pixels, in paint order."""
    objects = result['objects']
    full_masks = result['full_masks']
    visible_masks = result['visible_masks']
    coverage = {'unoccluded_classes': [], 'vehicle': {}, 'mutual_occlusion': [], 'boxes_by_class': {}}
    for index, subject in enumerate(objects):
        if subject.get('view') != 'front':
            continue
        category = subject['category']
        coverage['boxes_by_class'][category] = coverage['boxes_by_class'].get(category, 0) + 1
        if subject['visible_fraction'] == 1:
            coverage['unoccluded_classes'].append(category)
        minimum_overlap = 0.02 * full_masks[index].sum()
        overlaps = [
            (blocker, int((full_masks[index] & visible_masks[blocker]).sum()))
            for blocker in range(index+1, len(objects))
        ]
        for blocker, pixels in overlaps:
            if pixels < minimum_overlap:
                continue
            occluder = objects[blocker]
            record = {'subject': subject['id'], 'class': category, 'occluder': occluder['id'], 'pixels': pixels}
            if occluder['kind'] == 'vehicle':
                coverage['vehicle'].setdefault(occluder['category'], []).append(record)
            elif occluder.get('view') == 'front':
                coverage['mutual_occlusion'].append(record)
    return coverage


def _train_background_pool(args):
    index, rounds, fraction, dispatch_manifest = args
    import cv2

    cv2.setNumThreads(1)
    try:
        plan, _ = build_sample(background_indices=[index], validate_selected=False, directed_train=True,
                               background_scope='task_complete', min_placed_fraction=fraction,
                               layout_rounds=rounds, seed=173+index, dispatch_manifest=dispatch_manifest)
    except ValueError as error:
        if str(error) == 'no feasible person sample scenes':
            return None
        raise
    for state in plan['strata']:
        state['id'] = 'full_' + state['id']
    return plan


def _full_train_pool(indices, synth_budget, fraction, dispatch_manifest):
    rounds = max(2, math.ceil(2.5 * synth_budget / (105 * len(indices))))
    workers = min(6, os.cpu_count() or 1, len(indices))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        shards = list(filter(None, executor.map(_train_background_pool, (
            (index, rounds, fraction, dispatch_manifest) for index in indices
        ))))
    if not shards:
        raise ValueError('no complete background produced a feasible train layout')
    pool = [state for shard in shards for state in shard['strata']]
    random.Random(17).shuffle(pool)
    return dict(shards[0], budget=synth_budget, strata=pool[:10000])


def _previous_train_states(available, indices, total, synth_budget):
    """Retain the layouts whose complete outputs exist in the approved prior batch."""
    path = ROOT / 'datasets/exports/person/train/manifest.json'
    if total <= 300 or not path.is_file():
        return []
    approved = read_json(path)
    exported = {item['state_id']: item for item in approved['records'] if item.get('state_id')}
    if not exported or len(exported) > min(300, synth_budget):
        return []
    prior = read_json(CONFIG / 'production_plan.json')
    retained = []
    for state in prior['strata']:
        item = exported.get(state['id'], {})
        background = state['background']
        index = background.get('record_index')
        if (index in indices and available[index]['image'] == background['image']
                and item.get('label_status') == 'complete_task_yolo'
                and item.get('parameters') == state['parameters']
                and item.get('source_assets') == state['sources']
                and state['parameters'].get('joint_coverage')):
            retained.append(state)
    return retained


def _inspect_train_candidate(job):
    declaration, state = job
    try:
        attach_sources(dict(declaration, strata=[state]))
        finding = _person_coverage(_compose_planned_state(declaration, state))
    except (ValueError, OSError, KeyError) as error:
        return state['id'], None, None, str(error)
    return state['id'], state['sources'], finding, None


def _single_cv_thread():
    import cv2

    cv2.setNumThreads(1)


def build_supervised_train(total=300, *, dispatch_manifest=None):
    """Compile bounded train quotas from split-assigned sources and renderable layouts."""
    if type(total) is not int or total < 1:
        raise ValueError('train budget must be a positive integer')
    if total <= 300:
        return _build_supervised_train(total, dispatch_manifest=dispatch_manifest)
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1), initializer=_single_cv_thread) as executor:
        return _build_supervised_train(total, executor, dispatch_manifest=dispatch_manifest)


def _build_supervised_train(total, executor=None, *, dispatch_manifest=None):
    from datasets.scripts.common.package_direct_real import prepare
    from datasets.scripts.person.compose_planned_ground_person import _batch_layout
    from datasets.scripts.person.planned_sources import (complete_background, configured_background_dispatch,
                                                         verify_current_backgrounds)
    from lab.training_common.bayesian_allocation import plan_generation

    dataset = yaml.safe_load((CONFIG / 'dataset.yaml').read_text(encoding='utf-8'))
    source_plan = yaml.safe_load((ROOT / 'config/full_train_allocation.yaml').read_text(encoding='utf-8'))['tasks']['person']
    if type(total) is not int or total < 1:
        raise ValueError('train budget must be a positive integer')
    direct, _ = prepare('person', split='train')
    direct_ids = [record[1]['id'] for record in direct]
    direct_frames = {record[1]['frame_environment']['frame_id'] for record in direct}
    synth_budget = total - len(direct_ids)
    blender_goal = round(total * source_plan['source_mix']['blender'])
    dispatch_manifest, environment_backed = configured_background_dispatch(dispatch_manifest)
    available = read_json(dispatch_manifest)['records']
    indices = [
        index for index, row in enumerate(available)
        if not row.get('boxes') and complete_background(row, row['state'].removeprefix('candidate_'), 'train')
    ]
    if not indices:
        raise ValueError('no complete, train-assigned person backgrounds')
    fraction = dataset['target_spec']['minimum_placed_height_fraction']
    if total > 300:
        batch = _full_train_pool(indices, synth_budget, fraction, dispatch_manifest)
    else:
        batch, _ = build_sample(budget=max(450, synth_budget), background_indices=indices, validate_selected=False,
                                background_scope='task_complete', directed_train=True, min_placed_fraction=fraction,
                                dispatch_manifest=dispatch_manifest)
    pool = batch['strata']
    by_id = {state['id']: state for state in pool}
    chosen, consumed, checked, invalid = [], set(), {}, {}
    source_counts, cell_counts, capacity_gaps = Counter(), Counter(), Counter()
    white_textures = white_back_textures()

    def background(state):
        return available[state['background']['record_index']]

    def is_blender(state):
        return background(state).get('generation_method') == 'blender_render'

    if not environment_backed and dispatch_manifest == DISPATCH:
        chosen.extend(_previous_train_states(available, indices, total, synth_budget))
    for state in chosen:
        consumed.add(state['id'])
        cell_counts[state['parameters']['joint_coverage']['quota_cell']] += 1
        source_counts['blender_background_composite' if is_blender(state) else 'real_background_composite'] += 1
    retained_count = len(chosen)

    def scope_review(row):
        if environment_backed:
            return row['label'].get('evidence') or {}
        return row.get('person_scope_review', {})

    def vehicle_free(state):
        return (state['axes']['vehicle_occlusion'] == 'none'
                and scope_review(background(state)).get('vehicle_visible') is False)

    def matched_states(state, category):
        for scene in list(_contrast_states(state, category, white_textures))[1:]:
            scene['id'] += f'_{category}'
            yield scene

    scope = {key: batch[key] for key in ('model_key', 'target_split', 'constraints')}

    def inspect(scene_sets):
        pending = {state['id']: state for scenes in scene_sets for state in scenes
                   if state['id'] not in checked and state['id'] not in invalid}
        jobs = ((scope, state) for state in pending.values())
        results = (executor.map(_inspect_train_candidate, jobs, chunksize=4) if executor else
                   map(_inspect_train_candidate, jobs))
        for key, sources, fact, error in results:
            if error is not None:
                invalid[key] = error
            else:
                pending[key]['sources'] = sources
                checked[key] = fact
        return [[checked.get(scene['id']) for scene in scenes] for scenes in scene_sets]

    def take(cell, count, candidates, *, passes=lambda _state, _facts: True, variants=None, retain_minimum=True):
        # Reserve every image of a matched set in the originating background domain.
        unit = 3 if variants else 1
        if retain_minimum:
            count = max(0, count - cell_counts[cell]//unit)
        requested = count
        # Coverage is a design target, not a reason to discard other viable states.
        # A matched contrast is indivisible: do not allocate a partial set.
        count = min(count, max(0, synth_budget - len(chosen)) // unit)
        initial_count = cell_counts[cell]
        candidates = {state['id']: state for state in candidates if state['id'] not in consumed}
        while count:
            real, blender = [], []
            for key, state in candidates.items():
                if key not in invalid:
                    (blender if is_blender(state) else real).append(state)
            attainable = len(real) + len(blender)
            if not attainable:
                break
            count = min(count, attainable)
            target_share = max(0, blender_goal-source_counts['blender_background_composite']) / (synth_budget-len(chosen))
            target = min(count, len(blender), round(count * target_share))
            target = max(target, count-len(real))
            for domain, number in ((real, count-target), (blender, target)):
                if not number:
                    continue
                optional = [dict(state, min_count=0) for state in domain]
                declaration = dict(batch, strata=optional, budget=number)
                if cell == 'pure_empty':
                    declaration['conditional_balance'] = {
                        'axes': ['background', 'background_image'], 'stratify_by': 'background', 'seed': 17,
                    }
                allocation = plan_generation(declaration, ROOT)['strata']
                selected = [candidates.pop(row['id']) for row in allocation if row['planned_count']]
                scene_sets = [list(variants(base)) if variants else [base] for base in selected]
                findings = inspect(scene_sets)
                for base, scenes, facts in zip(selected, scene_sets, findings):
                    if any(fact is None or not passes(scene, fact) for scene, fact in zip(scenes, facts)):
                        continue
                    for scene, finding in zip(scenes, facts):
                        scene['min_count'] = 1
                        scene['parameters']['joint_coverage'] = {'quota_cell': cell, **finding}
                        scene.pop('_procedural', None)
                    chosen.extend(scenes)
                    consumed.add(base['id'])
                    source_counts['blender_background_composite' if is_blender(base)
                                  else 'real_background_composite'] += len(scenes)
                    cell_counts[cell] += len(scenes)
                    count -= 1
        achieved = (cell_counts[cell] - initial_count) // unit
        if requested > achieved:
            capacity_gaps[cell] += (requested - achieved) * unit

    bases = [state for state in pool if state['axes']['person_count'] == 2
             and state['axes']['vehicle_occlusion'] == 'none']
    pure = []
    used_empty_frames = {s['background']['image'] for s in chosen if s['failure_event'] == 'empty'}
    for index in indices:
        row = available[index]
        if row.get('frame_id') in direct_frames:
            continue
        review = scope_review(row)
        if review.get('scene_state') != 'no_person_or_standee' or review.get('vehicle_visible') is not False:
            continue
        bg, polygon = background_block(available, index, dispatch_manifest)
        width, height = row['source_image_size']
        polygon = [[x, round(y * 4/3) if width * 3 == height * 4 else y + 60] for x, y in polygon]
        empty = make_state(f'strict_empty_{index}', 'empty', bg, polygon, [],
                           {'value_gain': 1, 'saturation_gain': 1}, [])
        empty['parameters']['resolution'] = [640, 480]
        empty['axes'] = {'person_count': 0, 'background': row['source_group'], 'background_image': row['image']}
        if row['image'] not in used_empty_frames:
            pure.append(empty)
    attach_sources(dict(batch, strata=pure))
    pure_desired = max(1, total//35)
    pure_capacity = len(pure) + len(used_empty_frames)
    take('pure_empty', min(pure_desired, pure_capacity), pure)
    for category in ('community', 'noncommunity'):
        class_bases = [state for state in bases if _has_white_contrast(state, category, white_textures)]
        take(f'matched_{category}', max(1, total//25), class_bases,
             variants=lambda state: matched_states(state, category))
        take(f'{category}_no_vehicle', max(1, total//25), [state for state in pool if vehicle_free(state)],
             passes=lambda _state, facts: category in facts['unoccluded_classes'])
        for posture in ('upright', 'fallen'):
            candidates = [state for state in pool if state['axes']['vehicle_occlusion'].startswith(posture)
                          and any(a['kind'] == 'person' and a['category'] == category
                                  for a in state['target']['actors'])]
            take(f'{category}_{posture}', max(1, total//25), candidates,
                 passes=lambda _state, facts: any(a['class'] == category for a in facts['vehicle'].get(posture, [])))
    take('person_mutual_occlusion_vehicle_free', max(1, total//25), [state for state in pool if vehicle_free(state)],
         passes=lambda _state, facts: bool(facts['mutual_occlusion']))
    for count in (2, 3):
        take(f'{count}_people_fallen', max(1, total//50),
             [state for state in pool if state['axes']['person_count'] == count
              and state['axes']['vehicle_occlusion'] == 'fallen'],
             passes=lambda _state, facts: bool(facts['vehicle'].get('fallen')))
    for count in (5, 6):
        take(f'{count}_people_vehicle_free', max(1, total//50),
             [state for state in pool if state['axes']['person_count'] == count and vehicle_free(state)])
    for count in (2, 3, 4, 5, 6):
        missing = max(0, total//25 - sum(s['axes']['person_count'] == count for s in chosen))
        take(f'{count}_people', missing, [state for state in pool if state['axes']['person_count'] == count],
             retain_minimum=False)
    take('additional_multi', synth_budget-len(chosen), by_id.values(), retain_minimum=False)
    source_counts['direct_real'] = len(direct_ids)
    planned_classes = Counter()
    for state in chosen:
        planned_classes.update(state['parameters']['joint_coverage']['boxes_by_class'])
    if not chosen:
        raise ValueError('no feasible synthetic person states; direct-only export requires a separate entry')
    batch.update(strata=chosen, budget=len(chosen), direct_capture_ids=direct_ids,
                 purpose='Bounded person train scenes with explicit observed joint coverage and complete labels.')
    batch['design'].update({
        'requested_images': total, 'dispatch_manifest': dispatch_manifest, 'eligible_backgrounds': len(indices),
        'planned_source_mix': dict(source_counts),
        'baseline_source_mix': source_plan['source_mix'], 'blender_cap': source_plan['blender_cap'],
        'blender_cap_enforced': False,
        'planned_images': len(chosen) + len(direct_ids),
        'unfilled_images': max(0, total - len(chosen) - len(direct_ids)),
        'planned_direct_real_gap': max(0, round(total * source_plan['source_mix']['direct_real']) - len(direct_ids)),
        'retained_verified_synthetic_layouts': retained_count,
        'pure_empty_unique_base_capacity': pure_capacity,
        'pure_empty_unique_base_gap': max(0, pure_desired - pure_capacity),
        'scaled_joint_quota_capacity_gaps': dict(capacity_gaps),
        'joint_scene_counts': dict(cell_counts), 'verified_synth_boxes_by_class': dict(planned_classes),
        'background_source_groups': dict(Counter(background(state)['source_group'] for state in chosen)),
        'discarded_non_composable_candidates': len(invalid),
    })
    batch.pop('conditional_balance')
    _batch_layout(batch)
    if environment_backed:
        verify_current_backgrounds(chosen, 'train')
    return batch


SPLITS = {
    'train': (3000, 600, 0),
    'val': (300, 60, 0),
    'test': (300, 60, 0),
}

FILES = {
    'train': 'full_train_plan_3000.json',
    'val': 'val_plan_300.json',
    'test': 'test_plan_300.json',
}


def validate(declaration, limit=20, *, historical_replay=False):
    """Compose every declared state in memory without writing any file."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import cv2

    if historical_replay:
        from datasets.scripts.common.frame_environment import verified_image

        backgrounds = {(bg['image'], bg['content_sha256']): bg for state in declaration['strata']
                       if (bg := state['background']).get('record_id')}
        for background in backgrounds.values():
            verified_image(background)

    def compose_one(state):
        try:
            _compose_planned_state(declaration, state, historical_replay=historical_replay)
        except Exception as error:  # noqa: BLE001 - every failure is reported by state
            return {'state_id': state.get('id'), 'error': '{}: {}'.format(type(error).__name__, error)}
        return None

    # Full selected batches are independent. Keep limited-error diagnostics
    # sequential, and bound native OpenCV threads during concurrent composition
    # so four workers do not each start a full machine-sized thread pool.
    states = declaration['strata']
    if limit is None and len(states) >= 8:
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                return [error for error in executor.map(compose_one, states) if error is not None]
        finally:
            cv2.setNumThreads(previous_threads)
    failures = []
    for state in states:
        error = compose_one(state)
        if error is not None:
            failures.append(error)
            if limit is not None and len(failures) >= limit:
                break
    return failures


def main():
    from datasets.scripts.person.planned_sources import TASK_COMPLETE, configured_background_dispatch, file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=sorted(SPLITS), default=None)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--validate', action='store_true', help='validate explicit historical layouts')
    parser.add_argument('--sample', action='store_true',
                        help='compile a small candidate batch using the current background dispatch')
    parser.add_argument('--dispatch-manifest', help='explicitly select a registered background dispatch')
    parser.add_argument('--plan-output', type=Path, help='write a new JSON plan in person/config')
    parser.add_argument('--replay-plan', help='validate an existing person plan against its recorded inputs')
    parser.add_argument('--supervised-sample', nargs='+', type=int, metavar='BACKGROUND_INDEX',
                        help='compile matched contrasts using these complete, person-empty train backgrounds')
    parser.add_argument('--supervised-train', action='store_true',
                        help='compile the bounded complete-label train plan with joint coverage')
    parser.add_argument('--budget', type=int, metavar='IMAGES',
                        help='total image budget including direct real for a single split')
    parser.add_argument('--include-train-direct', action='store_true',
                        help='include direct person records already assigned to train in the supervised sample')
    parser.add_argument('--empty-share', type=int, default=None,
                        help='zero-person states in historical compilation using the original dispatch')
    args = parser.parse_args()
    if args.replay_plan:
        if any((args.split, args.sample, args.supervised_sample, args.supervised_train, args.dispatch_manifest,
                args.write, args.plan_output, args.budget is not None, args.empty_share is not None,
                args.include_train_direct)):
            parser.error('--replay-plan cannot be combined with compilation or writing options')
        declaration = read_json(file(args.replay_plan))
        if declaration.get('model_key') != 'person' or declaration.get('target_split') not in SPLITS:
            parser.error('--replay-plan must select an existing person split plan')
        failures = validate(declaration, historical_replay=True)
        print(json.dumps({'replay_plan': args.replay_plan, 'split': declaration['target_split'],
                          'states': len(declaration['strata']), 'failures': failures}, ensure_ascii=False))
        return 1 if failures else 0

    if sum((args.sample, args.supervised_sample is not None, args.supervised_train)) > 1:
        parser.error('choose one compilation mode')
    if args.include_train_direct and args.supervised_sample is None:
        parser.error('--include-train-direct requires --supervised-sample')
    if args.budget is not None and (not (args.split or args.supervised_train)
                                    or args.sample or args.supervised_sample is not None):
        parser.error('--budget requires --supervised-train or one production --split')
    if (args.supervised_train or args.supervised_sample is not None) and args.split not in (None, 'train'):
        parser.error('supervised train and supervised sample use the train split')
    if args.empty_share is not None and (args.dispatch_manifest != DISPATCH or args.supervised_train or
                                         args.sample or args.supervised_sample is not None):
        parser.error('--empty-share requires explicit --dispatch-manifest for the original background dispatch')
    if args.write != bool(args.plan_output):
        parser.error('write a new plan with --write and --plan-output together')
    if args.write and not (args.split or args.supervised_train or args.sample or args.supervised_sample is not None):
        parser.error('choose one split or sample to write a plan')
    destination = (ROOT / args.plan_output).resolve() if args.plan_output else None
    if destination and (destination.parent != CONFIG or destination.suffix != '.json' or destination.exists()):
        parser.error('--plan-output must name a new JSON file in person/config')

    dispatch_manifest, environment_backed = configured_background_dispatch(args.dispatch_manifest)

    def report(declaration, summary):
        if destination:
            with destination.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(declaration, stream, ensure_ascii=False, indent=1)
                stream.write('\n')
        print(json.dumps(summary, ensure_ascii=False))

    if args.sample or args.supervised_sample is not None:
        if args.supervised_sample is None:
            declaration, failures = build_sample(
                split=args.split or 'train',
                dispatch_manifest=dispatch_manifest,
                background_scope=TASK_COMPLETE if environment_backed else 'recorded_background_scope')
        else:
            declaration, failures = build_supervised_sample(args.supervised_sample, dispatch_manifest=dispatch_manifest)
        if args.include_train_direct:
            from datasets.scripts.common.package_direct_real import prepare

            direct, _ = prepare('person', split='train')
            declaration['direct_capture_ids'] = [record[1]['id'] for record in direct]
        report(declaration, {'dispatch_manifest': dispatch_manifest, 'sample_states': declaration['budget'],
                             'candidates': len(declaration['strata']),
                             'backgrounds': len({state['background']['image'] for state in declaration['strata']}),
                             'direct_real': len(declaration.get('direct_capture_ids', [])), 'skipped': failures})
        return 0

    splits = ['train'] if args.supervised_train else [args.split] if args.split else list(SPLITS)
    for split in splits:
        total = args.budget if args.budget is not None else 300 if args.supervised_train else SPLITS[split][0]
        if args.empty_share is None:
            if split == 'train':
                declaration = build_supervised_train(total, dispatch_manifest=dispatch_manifest)
            else:
                from datasets.scripts.person.build_evaluation_plan import build as build_evaluation

                declaration = build_evaluation(split, total, dispatch_manifest=dispatch_manifest)
            report(declaration, declaration['design'])
        else:
            dispatch = read_json(DISPATCH)
            backgrounds = background_splits(
                dispatch['records'], use_declared=dispatch.get('split_assignment') == 'source_pairing')[split]
            backgrounds = [index for index in backgrounds if not dispatch['records'][index].get('source_scene')]
            declaration = build(split, total, backgrounds, SPLITS[split][1], args.empty_share)
            failures = validate(declaration) if args.validate else []
            if failures:
                print(json.dumps({'split': split, 'failures': failures}, ensure_ascii=False))
                return 1
            report(declaration, {'split': split, 'states': len(declaration['strata']),
                                 'dispatch_manifest': dispatch_manifest})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
