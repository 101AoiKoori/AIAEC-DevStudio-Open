"""Create source-grouped person/vehicle candidates on real scene ground crops.

The 0–5.97 second ground crops were visually checked for existing foreground
targets and wall reflections. They all come from one recorded video and are
training candidates, never an independent validation/test set. The 2D yaw
and pitch are bounded planar projections, not novel measured 3D viewpoints.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use datasets/scripts/person/compose_planned_ground_person.py.")

import argparse
import csv
import json
import math
import random
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from datasets.scripts.common import images as assets
from datasets.scripts.common.composite import alpha_place, visible_masks, truncated_gauss, random_actor, draw_shadow
from datasets.scripts.common.images import read_image as read_color, write_image, exposure
from datasets.scripts.common.annotations import class_yolo_row as yolo_row
from datasets.scripts.common.perspective import planar_warp, project_corners
from camera_tone import apply_tone
from lab.training_common.production_guard import generation_guard

VIDEO = 'Archive/Assets/working_sources/legacy/site/场景/综合视频/空场与场景物料_001.mp4'
BACKGROUND_ROOT = ROOT / 'datasets/assets/person/dataset'
PEOPLE_ROOT = ROOT / 'Archive/Assets/working_sources/digital/person/team_assets'
VEHICLE_SELECTION = ROOT / 'datasets/annotated/e_bike/multiview_masks_refined_20261110_v2/selection.json'
GROUND_CROP = (480, 600, 1440, 1080)
CONTEXT_CROP = (0, 360, 1440, 1080)
SIGN_CROP = (0, 120, 1920, 1080)
WIDTH, HEIGHT = 960, 480
PERSON_CLASSES = {'community': 0, 'noncommunity': 1}
VEHICLE_CLASSES = {'upright': 0, 'fallen': 1}


def path_in_workspace(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('source path outside workspace')
    return path






def scenarios(index):
    """20-image cycle yields 25/30/30/15 percent scene types exactly."""
    cases = (('single', 1, 0, 'none'), ('multi', 2, 0, 'none'),
             ('occlusion', 1, 1, 'vehicle_front'), ('no_person', 0, 1, 'none'),
             ('single', 1, 0, 'none'), ('multi', 2, 0, 'none'),
             ('occlusion', 1, 1, 'person_front'), ('occlusion', 2, 2, 'vehicle_front'),
             ('no_person', 0, 1, 'none'), ('multi', 3, 0, 'none'),
             ('single', 1, 0, 'none'), ('single', 1, 0, 'none'),
             ('single', 1, 0, 'none'), ('multi', 2, 0, 'none'),
             ('multi', 3, 0, 'none'), ('multi', 2, 0, 'none'),
             ('occlusion', 1, 1, 'vehicle_front'), ('occlusion', 2, 2, 'person_front'),
             ('occlusion', 3, 2, 'vehicle_front'), ('no_person', 0, 0, 'none'))
    if index < 0:
        raise ValueError('negative index')
    kind, people, bikes, relation = cases[index % len(cases)]
    return {'kind': kind, 'people': people, 'vehicles': bikes, 'relation': relation}


def scenario_occurrence(index):
    """Within each scene type, cycle labels instead of correlating class to type."""
    current = scenarios(index)['kind']
    per_cycle = {'single': 5, 'multi': 6, 'occlusion': 6, 'no_person': 3}[current]
    return (index // 20) * per_cycle + sum(
        scenarios(previous)['kind'] == current for previous in range(index % 20))




def lighting_level(occurrence):
    if occurrence < 0:
        raise ValueError('invalid occurrence')
    return ('dim', 'bright', 'normal')[occurrence % 3]


def background_kind_for(index, plan):
    if index % 3 == 0:
        return 'ground_only'
    if index % 3 == 2 and plan['people'] <= 1 and plan['vehicles'] <= 1:
        return 'sign_context'
    return 'track_context'


def illumination_stratum(index):
    """Separate each feasible scene/background/class/relation combination."""
    plan = scenarios(index)
    occurrence = scenario_occurrence(index)
    person = occurrence % 2 if plan['people'] else None
    vehicle = (occurrence // 2) % 2 if plan['vehicles'] else None
    return (plan['kind'], background_kind_for(index, plan), plan['relation'], person, vehicle)


@lru_cache(maxsize=1)
def illumination_period():
    """240 samples cover the periods of scene, background, and class plans."""
    positions = {}
    for index in range(240):
        positions.setdefault(illumination_stratum(index), []).append(index)
    return positions


def stratified_lighting_level(index):
    """Cycle light inside each stratum without scene/class confounding."""
    key = illumination_stratum(index)
    slots = illumination_period()[key]
    local_index = slots.index(index % 240)
    occurrence = (index // 240) * len(slots) + local_index
    offset = sum(ord(char) for char in repr(key)) % 3
    return lighting_level(occurrence + offset)


def planned_scene_indices(background_kind, target, lighting):
    """Return the existing scene-cycle indices matching an explicit state.

    Only combinations the current compositor can actually generate are
    accepted. The returned indices span the complete 720-frame cycle of the
    existing scenario, class, background and illumination schedules.
    """
    required = {'kind', 'people', 'vehicles', 'relation',
                'first_person_class', 'first_vehicle_class'}
    if not isinstance(target, dict) or set(target) != required:
        raise ValueError('target must declare scenario, counts, relation and first classes')
    if background_kind not in {'ground_only', 'track_context', 'sign_context'}:
        raise ValueError('unsupported background kind')
    if lighting not in {'dim', 'normal', 'bright'}:
        raise ValueError('unsupported lighting state')
    if type(target['people']) is not int or type(target['vehicles']) is not int:
        raise ValueError('target counts must be integers')
    matches = []
    for index in range(720):
        scenario = scenarios(index)
        occurrence = scenario_occurrence(index)
        person_class = ('community' if occurrence % 2 == 0 else 'noncommunity') if scenario['people'] else None
        vehicle_class = ('upright' if (occurrence // 2) % 2 == 0 else 'fallen') if scenario['vehicles'] else None
        actual = {**scenario, 'first_person_class': person_class,
                  'first_vehicle_class': vehicle_class}
        if (actual == target and background_kind_for(index, scenario) == background_kind
                and stratified_lighting_level(index) == lighting):
            matches.append(index)
    if not matches:
        raise ValueError('requested person scene state is not supported by the compositor')
    return tuple(matches)


def validate_person_production_plan(declaration):
    """Reject states this person-only scene generator cannot implement."""
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'person':
        raise ValueError('a person production declaration is required')
    if declaration.get('target_split') != 'train':
        raise ValueError('this generator supports only training candidates')
    if declaration.get('constraints') != {
            'person_scope': 'near_frontal', 'person_box': 'visible',
            'white_back': 'not_synthesized', 'distinct_person_sources': True}:
        raise ValueError('unsupported person scene constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('at least one person scene state is required')
    index_map = {}
    for state in states:
        if not isinstance(state, dict) or not isinstance(state.get('id'), str) or not state['id']:
            raise ValueError('each state requires an ID')
        allowed = {'id', 'failure_event', 'min_count', 'max_count', 'prior',
                   'exposure_weight', 'failure_cost', 'conditions', 'background',
                   'target', 'labels', 'augmentation', 'parameters', 'sources'}
        if set(state) != allowed:
            raise ValueError('state contains missing or unsupported fields')
        if state['id'] in index_map:
            raise ValueError('duplicate state ID')
        if state.get('conditions') != []:
            raise ValueError('additional conditions are not implemented')
        background = state.get('background')
        augmentation = state.get('augmentation')
        if not isinstance(background, dict) or set(background) != {'kind'}:
            raise ValueError('background must contain only kind')
        if not isinstance(augmentation, dict) or set(augmentation) != {'lighting'}:
            raise ValueError('augmentation must contain only lighting')
        if state.get('labels') != {'person': 'visible_yolo', 'vehicle': 'not_emitted'}:
            raise ValueError('unsupported label contract')
        if state.get('parameters') != {'resolution': [960, 480],
                                       'geometry': 'bounded_planar',
                                       'sampling': 'bounded_gaussian'}:
            raise ValueError('unsupported generation parameters')
        target = state.get('target')
        indices = planned_scene_indices(background['kind'], target, augmentation['lighting'])
        sources = ['scene_video_49s_shared']
        if target['people']:
            sources.append('person.team_assets')
        if target['vehicles']:
            sources.append('e_bike.multiview_masks_refined_20261110_v2')
        if state.get('sources') != sources:
            raise ValueError('source groups do not match the requested scene')
        index_map[state['id']] = indices
    return index_map


def has_requested_overlap(masks, kinds, relation):
    """Check actual pixel overlap with the requested back-to-front ordering."""
    if len(masks) != len(kinds) or relation not in ('vehicle_front', 'person_front'):
        raise ValueError('bad overlap inputs')
    behind, front = ('person', 'vehicle') if relation == 'vehicle_front' else ('vehicle', 'person')
    return any(kinds[i] == behind and kinds[j] == front and np.any(masks[i] & masks[j])
               for i in range(len(masks)) for j in range(i+1, len(masks)))






def eligible_backgrounds():
    with (BACKGROUND_ROOT / 'reports/backgrounds_manifest.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    chosen = []
    for row in rows:
        if 0 <= int(row['frame']) <= 179:
            path = (BACKGROUND_ROOT / 'backgrounds' / row['file']).resolve()
            if not path.is_relative_to((BACKGROUND_ROOT / 'backgrounds').resolve()) or not path.is_file():
                raise ValueError('missing background or unsafe filename')
            chosen.append({'path': path, 'frame': int(row['frame']), 'seconds': float(row['t'])})
    if len(chosen) != 50:
        raise ValueError('source changed: expected 50 visually reviewed crop frames')
    sign = BACKGROUND_ROOT / 'backgrounds/bg_00137_f00720.jpg'
    if not sign.is_file():
        raise ValueError('missing visually reviewed sign-context background')
    return chosen


def background_choice(index, plan, backgrounds, rng):
    """Separate complete person/vehicle supervision from person-only context."""
    kind = background_kind_for(index, plan)
    if kind == 'ground_only':
        kind, crop, supervised = 'ground_only', GROUND_CROP, True
        background = rng.choice(backgrounds)
    elif kind == 'sign_context':
        kind, crop, supervised = 'sign_context', SIGN_CROP, False
        background = {'path': BACKGROUND_ROOT / 'backgrounds/bg_00137_f00720.jpg',
                      'frame': 720, 'seconds': 24.0}
    else:
        kind, crop, supervised = 'track_context', CONTEXT_CROP, False
        background = rng.choice(backgrounds)
    return background, kind, crop, supervised


def sample_person_anchors(rng, count, background_kind):
    if background_kind == 'ground_only':
        left, right = 130, 830
    elif background_kind == 'track_context':
        left, right = 410, 820
    elif background_kind == 'sign_context':
        if count > 1:
            raise ValueError('sign context accepts at most one person')
        left, right = 430, 490
    else:
        raise ValueError('unknown background kind')
    positions = []
    for _ in range(count):
        for _ in range(100):
            x = rng.uniform(left, right)
            if all(abs(x-previous) >= 90 for previous in positions):
                positions.append(x)
                break
        else:
            raise ValueError('cannot place distinct person anchors')
    return positions


def validate_actor_placement(actor, background_kind):
    sh, sw = actor['sprite'].shape[:2]
    left = actor['foot_x'] - sw/2
    if background_kind == 'track_context' and left < 340:
        raise ValueError('actor would overlap existing vehicles or barrier')
    if background_kind == 'sign_context':
        if left < 380:
            raise ValueError('actor would overlap existing sign')
        boundary_y = 260 + max(0, actor['foot_x'] - 265) * .6875
        if actor['foot_y'] < boundary_y + 8:
            raise ValueError('actor footpoint is not on visible ground')
    if actor['foot_y'] > HEIGHT - 5 or actor['foot_y'] - sh < 0:
        raise ValueError('actor vertical placement outside frame')


def load_sources(use_whiteborder=False):
    people = {}
    textures = None
    if use_whiteborder:
        texture_path = ROOT / 'datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json'
        textures = json.loads(texture_path.read_text(encoding='utf8'))['records']
    for cls, folder, expected in (('community', '社区人员', 16), ('noncommunity', '非社区人员', 2)):
        paths = sorted((PEOPLE_ROOT / folder).glob('*.png'))
        if len(paths) != expected:
            raise ValueError(f'expected {expected} {cls} source images')
        if use_whiteborder:
            selected = [r for r in textures if r['class'] == cls]
            if len(selected) != expected or {path_in_workspace(r['artwork']) for r in selected} != set(paths):
                raise ValueError('white-border artwork inventory does not match sources')
            people[cls] = [(path_in_workspace(r['artwork']), assets.load_rgba(path_in_workspace(r['image']))) for r in selected]
        else:
            people[cls] = [(p, assets.load_rgba(p)) for p in paths]
    selection = json.loads(VEHICLE_SELECTION.read_text(encoding='utf8'))
    if selection.get('split') is not None:
        raise ValueError('vehicle sources must remain unsplit')
    vehicles = {'upright': [], 'fallen': []}
    for item in selection['vehicles']:
        if item['class'] not in vehicles:
            raise ValueError('unknown vehicle class')
        for key in ('path', 'source_frame', 'source_video'):
            if not path_in_workspace(item[key]).is_file():
                raise ValueError(f'missing {key}')
        vehicles[item['class']].append((item, assets.load_rgba(path_in_workspace(item['path']))))
    if any(len(items) < 2 for items in vehicles.values()):
        raise ValueError('need at least two source masks for each vehicle pose')
    return people, vehicles






def car_foot_y(rng):
    """Fitted ground region for the reviewed 640x480 development backgrounds."""
    return HEIGHT * truncated_gauss(rng, .85, .04, .73, .94)


def generate_one(index, rng, backgrounds, people, vehicles, white_back=None, car_background=None):
    plan = scenarios(index)
    occurrence = scenario_occurrence(index)
    if car_background is None:
        background, background_kind, crop, vehicle_supervised = background_choice(index, plan, backgrounds, rng)
        original = read_color(background['path'])
        if original.shape[:2] != (1080, 1920):
            raise ValueError('unexpected source dimensions')
        x0, y0, x1, y1 = crop
        base = original[y0:y1, x0:x1].copy()
        if base.shape[:2] != (HEIGHT, WIDTH):
            base = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        background, background_kind, crop, vehicle_supervised = car_background, 'ground_only', (0,0,640,480), True
        original = read_color(background['path'])
        if original.shape[:2] != (480, 640):
            raise ValueError('car background must be native 640x480')
        # The temporary extension permits reusing the 960px placement engine;
        # the V4 pilot crops exactly [160:800], retaining only the original image.
        base = cv2.copyMakeBorder(original, 0, 0, 160, 160, cv2.BORDER_REPLICATE)
    level = stratified_lighting_level(index)
    if level == 'dim':
        exposure_gain = truncated_gauss(rng, .86, .025, .82, .89)
    elif level == 'bright':
        exposure_gain = truncated_gauss(rng, 1.125, .018, 1.10, 1.15)
    else:
        exposure_gain = truncated_gauss(rng, 1, .045, .90, 1.08)
    gamma = truncated_gauss(rng, 1, .025, .95, 1.06)
    if car_background is not None:
        level, exposure_gain, gamma = 'native', 1.0, 1.0
    actors = []
    anchors = []
    person_x = sample_person_anchors(rng, plan['people'], background_kind)
    for i in range(plan['people']):
        cls = 'community' if (occurrence + i) % 2 == 0 else 'noncommunity'
        source, image = rng.choice(people[cls])
        foot_x = person_x[i]
        if car_background is not None:
            foot_y = car_foot_y(rng)
        elif background_kind == 'sign_context':
            foot_y = HEIGHT * truncated_gauss(rng, .95, .012, .93, .975)
        else:
            foot_y = HEIGHT * truncated_gauss(rng, .75, .07, .60, .90)
        height = 110 + (foot_y/HEIGHT-.60)*290 + truncated_gauss(rng, 0, 5, -12, 12)
        actor = random_actor(rng, 'person', cls, image, source.relative_to(ROOT).as_posix(), foot_x, foot_y, height)
        actors.append(actor)
        anchors.append((foot_x, foot_y, height))
    if white_back is not None:
        source, image = white_back
        foot_x = rng.uniform(260, 700)
        if anchors:
            foot_x = min(780, max(180, anchors[0][0] + rng.choice((-140, 140))))
        foot_y = car_foot_y(rng) if car_background is not None else HEIGHT * truncated_gauss(rng, .76, .04, .68, .86)
        height = 110 + (foot_y/HEIGHT-.60)*290 + truncated_gauss(rng, 0, 5, -12, 12)
        actors.append(random_actor(rng, 'person_back', 'white_back', image, source, foot_x, foot_y, height))
    for i in range(plan['vehicles']):
        cls = 'upright' if (occurrence // 2 + i) % 2 == 0 else 'fallen'
        source, image = rng.choice(vehicles[cls])
        if not anchors:
            foot_x = sample_person_anchors(rng, 1, background_kind)[0]
            if car_background is not None:
                foot_y = car_foot_y(rng)
            elif background_kind == 'sign_context':
                foot_y = HEIGHT * truncated_gauss(rng, .95, .012, .93, .975)
            else:
                foot_y = HEIGHT * truncated_gauss(rng, .75, .07, .60, .90)
            height = 110 + (foot_y/HEIGHT-.60)*290 + truncated_gauss(rng, 0, 5, -12, 12)
            actors.append(random_actor(rng, 'vehicle', cls, image, source, foot_x, foot_y, height))
            continue
        px, py, height = anchors[i % len(anchors)]
        sign = -1 if i % 2 else 1
        if plan['relation'] == 'separate':
            foot_x, foot_y = px + sign * rng.uniform(150, 190), py + rng.uniform(-5, 5)
        elif plan['relation'] == 'vehicle_front':
            foot_x, foot_y = px + sign * rng.uniform(15, 35), py + rng.uniform(5, 12)
        else:
            foot_x, foot_y = px + sign * rng.uniform(15, 35), py - rng.uniform(5, 12)
        actors.append(random_actor(rng, 'vehicle', cls, image, source, foot_x, foot_y, height))
    actors.sort(key=lambda a: a['foot_y'])
    for actor in actors:
        validate_actor_placement(actor, background_kind)
    canvas = base.copy()
    for actor in actors:
        draw_shadow(canvas, actor)
    masks = [alpha_place(canvas, actor['sprite'], actor['foot_x'], actor['foot_y']) for actor in actors]
    if plan['kind'] == 'occlusion' and not has_requested_overlap(
            masks, [actor['kind'] for actor in actors], plan['relation']):
        raise ValueError('requested foreground order has no actual pixel overlap')
    visible = visible_masks(masks)
    canvas = exposure(canvas, exposure_gain, gamma)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if not 35 <= float(gray.mean()) <= 220:
        raise ValueError('scene exposure outside usable range')
    metadata = []
    for actor, full, shown in zip(actors, masks, visible):
        fraction = shown.sum() / full.sum()
        if fraction < .55 or shown.sum() < 120:
            raise ValueError('occluded target is too ambiguous; resample scene')
        visible_luma = float(gray[shown].mean())
        if not 30 <= visible_luma <= 240:
            raise ValueError('visible target exposure outside usable range')
        box = assets.box(shown)
        meta = {k: actor[k] for k in ('kind', 'category', 'source', 'foot_x', 'foot_y',
                                      'angles_degrees', 'relative_exposure', 'person_reference_height_px',
                                      'sprite_width_px', 'sprite_height_px')}
        meta.update({'visible_bbox_xyxy': box, 'visible_fraction': float(fraction),
                     'visible_luma_mean': visible_luma,
                     'bbox_convention': 'xyxy pixels, right/bottom exclusive'})
        metadata.append(meta)
    background_meta = {'image': background['path'].relative_to(ROOT).as_posix(),
            'source_video': VIDEO, 'source_group': 'scene_video_49s_shared',
            'frame': background['frame'], 'seconds': background['seconds'], 'crop_xyxy': list(crop),
            'kind': background_kind, 'resize_to': [WIDTH, HEIGHT],
            'person_supervision': 'AI_visual_review_no_obvious_background_person',
            'vehicle_supervision': 'allowed_ground_crop' if vehicle_supervised else 'disabled_unlabelled_existing_or_reflected_vehicles',
            'background_review': 'all 50 crop thumbnails checked' if background_kind != 'sign_context' else 'single sign scene checked at full resolution'}
    if car_background is not None:
        background_meta.update(source_video=background['source_video'], source_group=background['source_group'],
                               kind='car_frame_640_reviewed', resize_to=[640,480], temporary_padding=[160,160],
                               person_supervision='AI_visual_review_no_in_scope_targets',
                               vehicle_supervision='AI_visual_review_no_in_scope_targets',
                               background_review=background['review_evidence'],
                               frame_time_status='same-id video alignment not verified', split='regression')
    return canvas, masks, visible, metadata, background_meta, \
        {'level': level, 'stratum': list(illumination_stratum(index)),
         'scene_gain': exposure_gain, 'gamma': gamma,
         'method': 'bounded global exposure and per-sprite gain'}


def crop_v4(image, full, visible, objects, left):
    """Crop 960x480 to 640x480 and recompute all visible boxes from masks."""
    if image.shape[:2] != (480, 960) or type(left) is not int or not 0 <= left <= 320:
        raise ValueError('invalid 640x480 crop')
    if not (len(full) == len(visible) == len(objects)):
        raise ValueError('object and mask counts differ')
    new_full, new_visible, result = [], [], []
    for fm, vm, obj in zip(full, visible, objects):
        if fm.shape != (480, 960) or vm.shape != fm.shape or np.any(vm & ~fm):
            raise ValueError('invalid source masks')
        f, v = fm[:, left:left+640].copy(), vm[:, left:left+640].copy()
        item = dict(obj)
        if 'foot_x' in item:
            item['foot_x_canvas_960'] = item['foot_x']
            item['foot_x'] -= left
        item['crop_retained_fraction'] = float(f.sum() / fm.sum()) if fm.any() else 0.0
        item['visible_fraction'] = float(v.sum() / f.sum()) if f.any() else 0.0
        item['visible_bbox_xyxy'] = assets.box(v) if v.any() else None
        item['crop_status'] = 'visible_candidate' if v.any() else ('fully_occluded' if f.any() else 'out_of_crop')
        new_full.append(f); new_visible.append(v); result.append(item)
    return image[:, left:left+640].copy(), new_full, new_visible, result


def choose_v4_left(masks, rng):
    """Keep all placed foreground within the small first-pass review frame."""
    occupied = [np.nonzero(mask)[1] for mask in masks if np.any(mask)]
    if not occupied:
        return 160
    min_x = min(int(x.min()) for x in occupied)
    max_x = max(int(x.max()) + 1 for x in occupied)
    low, high = max(0, max_x - 640), min(320, min_x)
    if low > high:
        raise ValueError('foreground cannot fit in 640px crop')
    return rng.randint(low, high)


def require_distinct_v4_people(objects):
    """Do not invent two physical standees from one frontal source image."""
    sources = [obj['source'] for obj in objects if obj['kind'] == 'person']
    if len(sources) != len(set(sources)):
        raise ValueError('same frontal standee source repeated in one frame')


def production_inputs(*, white_back=None, white_record=None, car_backgrounds=None, batch=False, ground_only=False):
    """Declare every possible input before a randomized production run."""
    procedural = {VEHICLE_SELECTION}
    media = {p for folder in ('社区人员', '非社区人员') for p in (PEOPLE_ROOT / folder).glob('*.png')}
    for row in json.loads(VEHICLE_SELECTION.read_text(encoding='utf8'))['vehicles']:
        media.update(path_in_workspace(row[key]) for key in ('path', 'source_frame', 'source_video'))
    if car_backgrounds:
        car_record = path_in_workspace(car_backgrounds)
        procedural.add(car_record)
        info = json.loads(car_record.read_text(encoding='utf8'))
        media.add(path_in_workspace(info['source_video']))
        for item in info['items']:
            media.update(path_in_workspace(item[key]) for key in ('image', 'review_evidence'))
    else:
        procedural.add(BACKGROUND_ROOT / 'reports/backgrounds_manifest.csv')
        media.update(row['path'] for row in eligible_backgrounds())
        if not ground_only:
            media.add(BACKGROUND_ROOT / 'backgrounds/bg_00137_f00720.jpg')
        media.add(path_in_workspace(VIDEO))
    if white_record:
        record = path_in_workspace(white_record)
        procedural.add(record)
        info = json.loads(record.read_text(encoding='utf8'))
        media.add(path_in_workspace(white_back))
        media.update(path_in_workspace(info[key]) for key in ('parent_image', 'parent_video', 'review_evidence'))
    if batch:
        manifest = ROOT / 'datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json'
        procedural.add(manifest)
        media.update(path_in_workspace(row['image']) for row in json.loads(manifest.read_text(encoding='utf8'))['records'])
    return sorted(media | procedural), sorted(procedural)


def generate_v4_pilot(output, count, seed, white_back_path, white_record_path, car_background_record=None, batch=False):
    """Reject the historical entry until its white-back states are planned."""
    raise ValueError('legacy V4 person generation is not connected to a production plan')


def _generate_v4_pilot(output, count, seed, white_back_path, white_record_path, car_background_record=None, batch=False, source_refs=None):
    """Create at most seven auditable candidates; no labels are declared accepted."""
    indices = (0, 3, 6, 9, 12, 18, 39)
    if not 1 <= count <= (128 if batch else len(indices)):
        raise ValueError('V4 count exceeds the selected mode limit')
    output = path_in_workspace(output)
    if not output.is_relative_to(ROOT / 'datasets/exports/person') or output.exists():
        raise ValueError('output must be a new directory under datasets/exports/person')
    source_path, source_record = map(path_in_workspace, (white_back_path, white_record_path))
    source = json.loads(source_record.read_text(encoding='utf8'))
    if path_in_workspace(source['foreground']) != source_path:
        raise ValueError('white-back foreground and source record differ')
    for field in ('parent_image', 'parent_video', 'review_evidence'):
        if not path_in_workspace(source[field]).is_file():
            raise ValueError('white-back source evidence missing: ' + field)
    white = (source, assets.load_rgba(source_path))
    car_backgrounds = []
    if car_background_record is not None:
        car_record_path = path_in_workspace(car_background_record)
        car_record = json.loads(car_record_path.read_text(encoding='utf8'))
        if car_record.get('split') != 'regression' or not path_in_workspace(car_record['source_video']).is_file():
            raise ValueError('car background record must preserve regression video source')
        for item in car_record['items']:
            if (item.get('person_status') != 'no_visible_in_scope' or
                    item.get('e_bike_status') != 'no_visible_in_scope' or
                    not path_in_workspace(item['review_evidence']).is_file()):
                raise ValueError('car background has incomplete task review')
            path = path_in_workspace(item['image'])
            if not path.is_file():
                raise ValueError('car background missing')
            car_backgrounds.append({'path':path, 'frame':item['frame'], 'seconds':None,
                                    'source_video':car_record['source_video'],
                                    'source_group':car_record['source_group'],
                                    'review_evidence':car_record_path.relative_to(ROOT).as_posix()})
        if not car_backgrounds:
            raise ValueError('car background record is empty')
    backgrounds = [] if car_backgrounds else eligible_backgrounds()
    people, vehicles = load_sources(use_whiteborder=batch)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('images', 'candidate_labels/person', 'candidate_labels/e_bike', 'masks/full', 'masks/visible', 'overlays'):
        (output / folder).mkdir(parents=True)
    summary = {'status': 'running', 'requested': count, 'completed': 0, 'seed': seed,
               'split': 'regression' if car_backgrounds else 'unassigned', 'review': 'candidate_not_accepted',
               'limits': 'One background video; planar sprites, AI-reviewed masks and white-back cutout require further visual review.'}
    summary['production_receipt_operation'] = 'person.real_ground_v4:' + output.relative_to(ROOT).as_posix()
    summary['white_back_source_record'] = source_record.relative_to(ROOT).as_posix()
    if batch:
        summary['foreground_texture_manifest'] = 'datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json'
    def save_summary():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save_summary()
    try:
        with (output/'manifest.jsonl').open('w', encoding='utf8') as records:
            for sample_index in range(count):
                plan_index = indices[sample_index % len(indices)] + (sample_index // len(indices))*60 if batch else indices[sample_index]
                rng = random.Random(seed + sample_index * 1009)
                max_attempts = 30 if batch else 100
                for attempt in range(1, max_attempts+1):
                    try:
                        car = car_backgrounds[sample_index % len(car_backgrounds)] if car_backgrounds else None
                        use_white = (plan_index % 20 in (3,12,19)) if batch else (plan_index in (3,12))
                        sample = generate_one(plan_index, rng, backgrounds, people, vehicles,
                                              white if use_white else None, car)
                        image, full, visible, objects, bg, lighting = sample
                        require_distinct_v4_people(objects)
                        if not car and (bg['kind'] != 'ground_only' or bg['vehicle_supervision'] != 'allowed_ground_crop'):
                            raise ValueError('V4 pilot requires reviewed ground-only background')
                        left = 160 if car else choose_v4_left(full, rng)
                        image, full, visible, objects = crop_v4(image, full, visible, objects, left)
                        if any(obj['crop_retained_fraction'] < .999 for obj in objects):
                            raise ValueError('V4 pilot foreground clipped during crop')
                        break
                    except ValueError:
                        if attempt == max_attempts:
                            if not batch:
                                raise RuntimeError(f'V4 scene {sample_index} could not meet constraints')
                            summary.setdefault('rejected_indices', []).append(sample_index)
                else:
                    continue
                stem = f'pilot_{sample_index+1:04d}' if batch else f'pilot_{sample_index+1:02d}'
                profile, value, saturation = (('original',1.,1.),('mild',.9,.85),('moderate',.7,.6))[sample_index % 3] if batch else ('original',1.,1.)
                if batch and profile != 'original':
                    image = apply_tone(image, value_gain=value, saturation_gain=saturation)
                write_image(output/'images'/(stem+'.png'), image)
                overlay = image.copy()
                labels = {'person': [], 'e_bike': []}
                for n, (obj, fm, vm) in enumerate(zip(objects, full, visible)):
                    fm_path, vm_path = f'masks/full/{stem}_{n:02d}.png', f'masks/visible/{stem}_{n:02d}.png'
                    write_image(output/fm_path, fm.astype(np.uint8)*255)
                    write_image(output/vm_path, vm.astype(np.uint8)*255)
                    obj.update({'full_mask': fm_path, 'visible_mask': vm_path,
                                'semantic_review': 'pending', 'physical_asset': obj['source']})
                    box = obj['visible_bbox_xyxy']
                    if box is None:
                        continue
                    if obj['kind'] == 'person':
                        labels['person'].append(yolo_row(PERSON_CLASSES[obj['category']], box, 640, 480))
                        color = (0, 230, 0)
                    elif obj['kind'] == 'vehicle':
                        labels['e_bike'].append(yolo_row(VEHICLE_CLASSES[obj['category']], box, 640, 480))
                        color = (255, 130, 0)
                    else:
                        contours, _ = cv2.findContours(vm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, (0, 200, 255), 1)
                        cv2.putText(overlay, 'white_back:not_label', (box[0], max(12, box[1]-4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 200, 255), 1, cv2.LINE_AA)
                        continue
                    cv2.rectangle(overlay, tuple(box[:2]), (box[2]-1, box[3]-1), color, 2)
                    cv2.putText(overlay, obj['kind'], (box[0], max(12, box[1]-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
                for kind, rows in labels.items():
                    (output/'candidate_labels'/kind/(stem+'.txt')).write_text('\n'.join(rows)+('\n' if rows else ''), encoding='utf8')
                write_image(output/'overlays'/(stem+'.png'), overlay)
                record = {'image': f'images/{stem}.png', 'candidate_labels': {k:f'candidate_labels/{k}/{stem}.txt' for k in labels},
                          'supervision': {'person':'semantic_review_pending', 'e_bike':'semantic_review_pending'},
                          'overlay':f'overlays/{stem}.png', 'split':'regression' if car else 'unassigned', 'seed':seed + sample_index*1009,
                          'attempt':attempt, 'scenario':scenarios(plan_index), 'background':bg, 'lighting':lighting,
                          'crop_xyxy_960x480':[left,0,left+640,480], 'output_size':[640,480],
                          'image_processing':{'resize':'native 640x480 retained' if car else 'source crop to 960x480, then 640x480 crop',
                                              'compression':'lossless PNG', 'blur':'none added'},
                          'objects':objects,'status':'AI_synthetic_candidate_requires_visual_review',
                          'white_back_policy':'hard_negative_no_person_box',
                          'tone_profile':profile,'tone_parameters':{'value_gain':value,'saturation_gain':saturation}}
                records.write(json.dumps(record, ensure_ascii=False)+'\n'); records.flush()
                summary['completed'] += 1; save_summary()
        summary['status'] = 'partial_candidates' if summary.get('rejected_indices') else 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'; summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_summary()
    return summary


def generate_planned_person(output, declaration_path, seed=7901):
    raise RuntimeError("Retired production API; use compose_planned_ground_person.produce")


def _produce_planned_person(output, seed, run, index_map):
    if output.exists():
        raise ValueError('output must be a new batch')
    backgrounds = eligible_backgrounds()
    people, vehicles = load_sources()
    output.mkdir(parents=True)
    for folder in ('images', 'labels', 'masks/full', 'masks/visible'):
        (output/folder).mkdir(parents=True)
    summary = {'status': 'running', 'seed': seed, 'completed': 0,
               'source_refs': run.source_refs, 'plan': run.plan,
               'scope': 'person candidates; generated pixels and model performance are unverified'}

    def save_summary():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

    save_summary()
    try:
        with (output/'manifest.jsonl').open('w', encoding='utf8') as records:
            serial = 0
            for state in run.plan['strata']:
                indices = index_map[state['id']]
                for ordinal in range(state['planned_count']):
                    cycle_index = indices[ordinal % len(indices)]
                    sample_seed = seed + serial * 1009
                    rng = random.Random(sample_seed)
                    for attempt in range(1, 101):
                        try:
                            image, full, visible, objects, bg, light = generate_one(
                                cycle_index, rng, backgrounds, people, vehicles)
                            require_distinct_v4_people(objects)
                            break
                        except ValueError as exc:
                            if attempt == 100:
                                raise RuntimeError(f"state {state['id']} could not be generated") from exc
                    stem = f'person_{serial+1:06d}'
                    image_path = output/'images'/(stem+'.png')
                    write_image(image_path, image)
                    rows = []
                    for number, (meta, fm, vm) in enumerate(zip(objects, full, visible)):
                        for name, mask in (('full', fm), ('visible', vm)):
                            relative = f'masks/{name}/{stem}_{number:02d}.png'
                            write_image(output/relative, mask.astype(np.uint8)*255)
                            meta[name+'_mask'] = relative
                        if meta['kind'] == 'person':
                            rows.append(yolo_row(PERSON_CLASSES[meta['category']],
                                                 meta['visible_bbox_xyxy'], WIDTH, HEIGHT))
                    label_path = output/'labels'/(stem+'.txt')
                    label_path.write_text('\n'.join(rows)+('\n' if rows else ''), encoding='utf8')
                    actual = {'seed': sample_seed, 'attempt': attempt, 'cycle_index': cycle_index,
                              'background': bg, 'lighting': light, 'objects': objects,
                              'label': label_path.relative_to(ROOT).as_posix()}
                    run.record_sample(state['id'], output=image_path,
                                      sources=state['sources'], parameters=state['parameters'], actual=actual)
                    records.write(json.dumps({'image': image_path.relative_to(ROOT).as_posix(),
                                              'state_id': state['id'], **actual}, ensure_ascii=False)+'\n')
                    records.flush()
                    serial += 1
                    summary['completed'] = serial
                    save_summary()
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_summary()
    return summary


def main():
    raise RuntimeError("Retired production API; use compose_planned_ground_person.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int)
    ap.add_argument('--seed', type=int, default=7901)
    ap.add_argument('--v4-pilot', action='store_true')
    ap.add_argument('--v4-batch', action='store_true')
    ap.add_argument('--white-back', type=Path)
    ap.add_argument('--white-back-record', type=Path)
    ap.add_argument('--car-backgrounds', type=Path)
    ap.add_argument('--production-plan', type=Path, required=True)
    args = ap.parse_args()
    if args.v4_pilot or args.v4_batch or args.white_back or args.white_back_record or args.car_backgrounds:
        ap.error('legacy V4 and white-back modes are not yet connected to the production plan')
    if args.count is not None:
        ap.error('set the sample count in production-plan budget')
    try:
        print(json.dumps(generate_planned_person(args.output, args.production_plan, args.seed), ensure_ascii=False))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        ap.exit(2, f'planned production failed: {exc}\n')


def _produce_standard(args, source_refs):
    output = path_in_workspace(args.output)
    if not output.is_relative_to(ROOT / 'datasets/exports/person') or output.exists():
        raise ValueError('output must be a new directory under datasets/exports/person')
    if not 1 <= args.count <= 100000:
        raise ValueError('count must be 1..100000')
    backgrounds = eligible_backgrounds()
    people, vehicles = load_sources()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('person/images', 'person/labels', 'vehicle/images', 'vehicle/labels', 'masks/full', 'masks/visible'):
        (output / folder).mkdir(parents=True)
    summary = {'status': 'running', 'requested': args.count, 'completed': 0, 'seed': args.seed,
               'source_group': 'scene_video_49s_shared', 'split': 'train_candidate',
               'source_video': VIDEO,
               'background_modes': {'ground_only': {'crop_xyxy': list(GROUND_CROP), 'vehicle_supervision': True},
                                    'track_context': {'crop_xyxy': list(CONTEXT_CROP), 'vehicle_supervision': False},
                                    'sign_context': {'crop_xyxy': list(SIGN_CROP), 'vehicle_supervision': False}},
               'class_names': {'person': PERSON_CLASSES, 'vehicle': VEHICLE_CLASSES},
               'limits': 'one real-video source; planar projection, not measured 3D; source masks need visual review',
               'illumination': 'bounded global exposure with small per-sprite variation',
               'distribution': {'method': 'bounded Gaussian via rejection, no clipping',
                                'yaw_degrees': [0, 5, -12, 12], 'pitch_degrees': [0, 2.5, -6, 6],
                                'roll_degrees': [0, 1, -3, 3], 'sprite_gain': [1, .035, .90, 1.08],
                                'scene_gain_dim': [.86, .025, .82, .89],
                                'scene_gain_bright': [1.125, .018, 1.10, 1.15],
                                'scene_gain_normal': [1, .045, .90, 1.08],
                                'gamma': [1, .025, .95, 1.06],
                                'format': '[mean, standard deviation, minimum, maximum]',
                                'allocation': 'scene exact 25/30/30/15; light balanced within each scene/background/class/relation stratum'}}
    summary['production_receipt_operation'] = 'person.real_ground:' + output.relative_to(ROOT).as_posix()
    def save_summary():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save_summary()
    try:
        with (output/'manifest.jsonl').open('w', encoding='utf8') as records:
            for index in range(args.count):
                rng = random.Random(args.seed + index*1009)
                for attempt in range(1, 101):
                    try:
                        image, full, visible, objects, bg, light = generate_one(index, rng, backgrounds, people, vehicles)
                        break
                    except ValueError:
                        if attempt == 100:
                            raise RuntimeError(f'scene {index} could not meet visibility constraints')
                stem = f'scene_{index+1:06d}'
                image_path = output/'person/images'/(stem+'.png')
                write_image(image_path, image)
                vehicle_supervised = bg['vehicle_supervision'] == 'allowed_ground_crop'
                if vehicle_supervised:
                    shutil.copyfile(image_path, output/'vehicle/images'/(stem+'.png'))
                rows = {'person': [], 'vehicle': []}
                for n, (meta, fm, vm) in enumerate(zip(objects, full, visible)):
                    fm_path, vm_path = f'masks/full/{stem}_{n:02d}.png', f'masks/visible/{stem}_{n:02d}.png'
                    write_image(output/fm_path, fm.astype(np.uint8)*255)
                    write_image(output/vm_path, vm.astype(np.uint8)*255)
                    meta.update({'full_mask': fm_path, 'visible_mask': vm_path})
                    classes = PERSON_CLASSES if meta['kind'] == 'person' else VEHICLE_CLASSES
                    rows[meta['kind']].append(yolo_row(classes[meta['category']], meta['visible_bbox_xyxy'], WIDTH, HEIGHT))
                for kind in ('person', 'vehicle'):
                    if kind == 'vehicle' and not vehicle_supervised:
                        continue
                    text = '\n'.join(rows[kind]) + ('\n' if rows[kind] else '')
                    (output/kind/'labels'/(stem+'.txt')).write_text(text, encoding='utf8')
                record = {'image': stem+'.png', 'split': 'train_candidate', 'attempt': attempt,
                          'scenario': scenarios(index), 'background': bg, 'lighting': light, 'objects': objects,
                          'person_image': f'person/images/{stem}.png', 'person_label': f'person/labels/{stem}.txt',
                          'vehicle_image': f'vehicle/images/{stem}.png' if vehicle_supervised else None,
                          'vehicle_label': f'vehicle/labels/{stem}.txt' if vehicle_supervised else None,
                          'status': 'AI_synthetic_candidate_requires_visual_review'}
                records.write(json.dumps(record, ensure_ascii=False)+'\n'); records.flush()
                summary['completed'] = index+1; save_summary()
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'; summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_summary()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
