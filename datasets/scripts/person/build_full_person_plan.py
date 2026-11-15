"""Compile the explicit person scene batches for train, val and test.

The batch is designed from the task goal first and then filled from the reviewed
sources this task actually has: six accepted empty backgrounds, eighteen standee
textures, four electric-vehicle occluders and three planned tone variants.

Split design (recorded in the plan's ``design`` block):

* train uses four of the five legacy empty-site backgrounds (80% parent cap).
* val holds out the fifth legacy background.
* test holds out the only background from the other source group.

Layouts are explicit per state: each state fixes background, identity, occlusion
state, tone, observation yaw and foot position, so one state is one image and the
batch has no duplicated layout signature.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'datasets/scripts/person/config'

DISPATCH = 'datasets/scripts/person/config/background_dispatch.json'
TEXTURES = 'datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json'
OCCLUDERS = 'datasets/annotated/person/v4_vehicle_occluders_20261113/manifest.json'

# background record indexes inside background_dispatch.json
TRAIN_BACKGROUNDS = [0, 1, 4, 5]
VAL_BACKGROUNDS = [2]
TEST_BACKGROUNDS = [3]

TONES = [
    {'value_gain': 1.0, 'saturation_gain': 1.0},
    {'value_gain': 1.1, 'saturation_gain': 0.72},
    {'value_gain': 0.88, 'saturation_gain': 0.82},
]
YAWS = [0, -5, 5]
HEIGHTS = [112, 128, 144, 160, 176, 192, 208, 224]
VEHICLE_HEIGHT_RATIO = 0.42
IDENTITY_REGION_RULE = 'geometric head-and-upper-body region x 10-70%, y 3-47% of the artwork'


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


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
    top = ground_top(ground_polygon)
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
        x_low = width // 2 + 2
        x_high = 638 - width // 2
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
    top = ground_top(ground_polygon)
    low = max(top + 3, max(height_a, height_b) + 1)
    high = 357
    if low > high:
        return None
    y = low + ((index * 19) % (high - low + 1))
    gap = 12
    total = width_a + width_b + gap
    x_low = total // 2 + 2
    x_high = 638 - total // 2
    if x_low >= x_high:
        return None
    centre = x_low + ((index * 97) % (x_high - x_low + 1))
    x_a = centre - total // 2 + width_a // 2
    x_b = centre + total // 2 - width_b // 2
    return {'a': {'foot_xy': [int(x_a), int(y)], 'size_wh': [int(width_a), int(height_a)]},
            'b': {'foot_xy': [int(x_b), int(y)], 'size_wh': [int(width_b), int(height_b)]}}


def build_sources():
    dispatch = read_json(DISPATCH)['records']
    textures = read_json(TEXTURES)['records']
    occluders = read_json(OCCLUDERS)['records']
    identities = []
    for index, record in enumerate(textures):
        width, height = png_size(ROOT / record['image'])
        identities.append({
            'record_index': index,
            'image': record['image'],
            'class': record['class'],
            'identity': Path(record['image']).stem.split('_', 1)[1],
            'ratio': width / height,
            'region': identity_region(width, height),
        })
    return dispatch, identities, occluders


def tone_at(index):
    return dict(TONES[index % len(TONES)])


def background_block(dispatch, record_index):
    record = dispatch[record_index]
    polygon = record['ground_polygon_640x360']
    return {
        'role': 'reviewed_background',
        'manifest': DISPATCH,
        'record_index': record_index,
        'image': record['image'],
    }, polygon


def person_actor(ident, placement, yaw, actor_id='person_added', category=None):
    return {
        'id': actor_id,
        'kind': 'person',
        'source': ident['image'],
        'source_role': 'foreground_texture',
        'source_record': {'manifest': TEXTURES, 'record_index': ident['record_index']},
        'view': 'front',
        'category': category or ident['class'],
        'identity': ident['identity'],
        'identity_evidence': '{}:record_index={}; {}'.format(TEXTURES, ident['record_index'], IDENTITY_REGION_RULE),
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
        'source_record': {'manifest': OCCLUDERS, 'record_index': record_index},
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
    shares = distribute(states_per_combo, len(ordered))
    result = []
    for ident, share in zip(ordered, shares):
        result.extend([ident] * share)
    return result


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
        'source_record': {'manifest': OCCLUDERS, 'record_index': record_index},
        'category': occluder['class'],
        'foot_xy': [int(x), int(y)],
        'contact_width': max(8, min(target_width, round(target_width * 0.3))),
        'size_wh': [int(target_width), int(target_height)],
        'angles_degrees': {'yaw': 0, 'pitch': 0, 'roll': 0},
    }


def compile_split(split, total, backgrounds, identities, occluders, *, multi_share, empty_share):
    dispatch, _, _ = build_sources()
    states = []
    counter = itertools.count(1)
    occluded = 750 if split == 'train' else 200
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


def build(split, total, backgrounds, multi_share, empty_share):
    dispatch, identities, occluders = build_sources()
    states = compile_split(split, total, backgrounds, identities, occluders,
                           multi_share=multi_share, empty_share=empty_share)
    declaration = {
        'schema_version': 1,
        'model_key': 'person',
        'target_split': split,
        'purpose': 'Full person batch compiled from reviewed backgrounds, standee textures and vehicle occluders.',
        'budget': len(states),
        'uncertainty_weight': 0,
        'independence_basis': 'Four legacy backgrounds (train), one held-out legacy background (val), one other-group background (test); shared textures and two vehicle videos.',
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


SPLITS = {
    'train': (3000, TRAIN_BACKGROUNDS, 600, 0),
    'val': (300, VAL_BACKGROUNDS, 60, 0),
    'test': (300, TEST_BACKGROUNDS, 60, 0),
}

FILES = {
    'train': 'full_train_plan_3000.json',
    'val': 'val_plan_300.json',
    'test': 'test_plan_300.json',
}


def validate(declaration):
    """Compose every declared state in memory without writing any file."""
    from datasets.scripts.person.compose_planned_ground_person import prepare
    from datasets.scripts.person.person_scene_plan import compose_scene

    failures = []
    for number, state in enumerate(declaration['strata'], 1):
        try:
            item = prepare(dict(declaration, budget=1, strata=[state]))
            compose_scene(item['background_image'], item['scene'], item['foregrounds'], item['identity_masks'])
        except Exception as error:  # noqa: BLE001 - every failure is reported by state
            failures.append({'state_id': state.get('id'), 'error': '{}: {}'.format(type(error).__name__, error)})
            if len(failures) >= 20:
                break
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=sorted(SPLITS), default=None)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--empty-share', type=int, default=None,
                        help='zero-person negative states per split; 0 reproduces the produced batches')
    args = parser.parse_args()
    splits = [args.split] if args.split else list(SPLITS)
    for split in splits:
        total, backgrounds, multi, empty = SPLITS[split]
        if args.empty_share is not None:
            empty = args.empty_share
        path = CONFIG / FILES[split]
        if args.validate and not args.write and path.is_file():
            declaration = json.loads(path.read_text(encoding='utf-8'))
        else:
            declaration = build(split, total, backgrounds, multi, empty)
            if args.write:
                path.write_text(json.dumps(declaration, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
                print('wrote', path.relative_to(ROOT).as_posix())
        states = declaration['strata']
        summary = {
            'split': split,
            'states': len(states),
            'multi': sum(1 for state in states if state['failure_event'] == 'person_multi'),
            'empty': sum(1 for state in states if state['failure_event'] == 'empty'),
            'upright': sum(1 for state in states if state['failure_event'] == 'person_upright'),
            'fallen': sum(1 for state in states if state['failure_event'] == 'person_fallen'),
            'none': sum(1 for state in states if state['failure_event'] == 'person_none'),
            'backgrounds': len({state['background']['image'] for state in states}),
            'identities': len({actor['identity'] for state in states
                               for actor in state['target']['actors'] if actor['kind'] == 'person'}),
        }
        print(json.dumps(summary, ensure_ascii=False))
        if args.validate:
            failures = validate(declaration)
            print(json.dumps({'split': split, 'failed': len(failures), 'failures': failures},
                             ensure_ascii=False, indent=1))
            if failures:
                raise SystemExit(1)


if __name__ == '__main__':
    main()
