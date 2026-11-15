"""Generate source-grouped e-bike detection candidates using recorded ground.

Classes are upright=0 and fallen=1. The video-derived foreground masks are
review candidates; output is not automatically verified ground truth.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/e_bike/. Production remains paused.")

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Retain the documented bike.scene access used by existing bike helpers.
# It resolves to this task module, never the person generator.
scene = sys.modules[__name__]
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common import images as assets
from datasets.scripts.common.images import read_image as read_color, write_image, exposure
from datasets.scripts.common.annotations import class_yolo_row as yolo_row
from datasets.scripts.common.perspective import planar_warp
from datasets.scripts.common.composite import alpha_place, visible_masks, truncated_gauss, random_actor, draw_shadow

SELECTION = source_path('e_bike', 'ground_vehicle_selection')
FIELD_SELECTION = ROOT / 'datasets/annotated/e_bike/field_eval_masks_20261110/selection_review_candidate.json'
FIELD_BACKGROUND = ROOT / 'datasets/annotated/e_bike/video_review_20261110/images/source_003/frame_000015.jpg'
FIELD_VIDEO = 'datasets/raw/e_bike/archive_assets_legacy/视频/单辆直立电动车_低视角环绕_005.mp4'
CAMERA_SELECTION = ROOT / 'datasets/annotated/e_bike/vehicle_camera_eval_masks_20261110/selection_review_candidate.json'
CAMERA_BACKGROUND = ROOT / 'datasets/raw/traffic_light/archive_assets_legacy/照片/小车摄像头实拍_近似平视_信号灯正面与道路_001.png'



_SOURCE_CONFIG = load_task_config('e_bike')[1]
VIDEO = _SOURCE_CONFIG['vehicle_scene_parent_video']
BACKGROUND_ROOT = (ROOT / _SOURCE_CONFIG['vehicle_scene_background_root']).resolve()
PEOPLE_ROOT = (ROOT / _SOURCE_CONFIG['person_occluder_artwork_root']).resolve()
VEHICLE_SELECTION = (ROOT / _SOURCE_CONFIG['vehicle_scene_selection']).resolve()
if any(not p.is_relative_to(ROOT) for p in (BACKGROUND_ROOT, PEOPLE_ROOT, VEHICLE_SELECTION)):
    raise ValueError('e-bike configured sources must remain inside the workspace')
GROUND_CROP = (480, 600, 1440, 1080)
WIDTH, HEIGHT = 960, 480
VEHICLE_CLASSES = {'upright': 0, 'fallen': 1}

def path_in_workspace(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('source path outside workspace')
    return path


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


def validate_actor_placement(actor, background_kind):
    if background_kind != 'ground_only':
        raise ValueError('e-bike compositor supports ground_only placement')
    height = actor['sprite'].shape[0]
    if actor['foot_y'] > HEIGHT - 5 or actor['foot_y'] - height < 0:
        raise ValueError('actor vertical placement outside frame')


def load_people():
    people = {}
    for category, folder, expected in (('community', '社区人员', 16), ('noncommunity', '非社区人员', 2)):
        paths = sorted((PEOPLE_ROOT / folder).glob('*.png'))
        if len(paths) != expected:
            raise ValueError(f'expected {expected} {category} occluder source images')
        people[category] = [(path, assets.load_rgba(path)) for path in paths]
    return people


def scenario(index):
    slot = index % 20
    counts = (1, 2, 3, 4, 1, 2, 5, 7)
    if slot < 8:
        return 'upright_only', ['upright'] * counts[slot]
    if slot < 16:
        count = counts[slot - 8]
        return 'fallen_present', ['fallen'] + ['upright' if j % 2 else 'fallen' for j in range(1, count)]
    return 'empty', []


def sources(selection=SELECTION, include_people=True):
    people = load_people() if include_people else {}
    data = json.loads(selection.read_text(encoding='utf-8'))
    vehicles = {'upright': [], 'fallen': []}
    for entry in data['vehicles'] if 'vehicles' in data else data['records']:
        for key in ('path', 'source_frame', 'source_video', 'source_image'):
            if key not in entry:
                continue
            if not scene.path_in_workspace(entry[key]).is_file():
                raise ValueError(f'missing source: {entry[key]}')
        vehicles[entry['class']].append((entry, scene.assets.load_rgba(scene.path_in_workspace(entry['path']))))
    return people, vehicles


def validate_declaration(declaration):
    if not isinstance(declaration, dict):
        raise ValueError('production declaration must be an object')
    top_fields = {'schema_version', 'model_key', 'target_split', 'purpose', 'budget',
                  'uncertainty_weight', 'independence_basis', 'constraints', 'strata'}
    if set(declaration) != top_fields:
        raise ValueError('unsupported or missing declaration fields')
    if (declaration.get('model_key') != 'e_bike' or declaration.get('target_split') != 'train' or
            declaration.get('constraints') != {'minimum_visible_pixels': 120, 'minimum_visible_fraction': .55}):
        raise ValueError('unsupported model, split, or constraints')
    if type(declaration.get('budget')) is not int or not 1 <= declaration['budget'] <= 100000:
        raise ValueError('budget must be between 1 and 100000')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('a nonempty state plan is required')
    fields = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight',
              'failure_cost', 'conditions', 'background', 'target', 'labels', 'augmentation',
              'parameters', 'sources'}
    signatures = set()
    for state in states:
        if not isinstance(state, dict) or set(state) != fields or state['conditions'] != []:
            raise ValueError('unsupported state fields or unimplemented conditions')
        if type(state['max_count']) is not int or state['max_count'] > 1:
            raise ValueError('each exact state can generate at most one image')
        signature = json.dumps({name: state[name] for name in ('background', 'target', 'labels', 'augmentation', 'parameters')},
                               ensure_ascii=False, sort_keys=True, allow_nan=False)
        if signature in signatures:
            raise ValueError('duplicate exact generation states are not allowed')
        signatures.add(signature)


def planned_scene(state, backgrounds, vehicles, background_source, background_group):
    """Render one fully specified state; no state choice is made at random.

    This entry supports ground-only vehicle scenes.  Other requested state
    fields are rejected rather than silently treated as implemented.
    """
    background, target, labels = state['background'], state['target'], state['labels']
    augmentation, parameters = state['augmentation'], state['parameters']
    if set(background) != {'image', 'crop_xyxy'} or set(target) != {'vehicles'}:
        raise ValueError('unsupported background or target fields')
    if labels != {'format': 'yolo_visible_box', 'classes': {'upright': 0, 'fallen': 1}}:
        raise ValueError('unsupported label contract')
    if set(augmentation) != {'gain', 'gamma'} or set(parameters) != {'actors'}:
        raise ValueError('unsupported augmentation or parameter fields')
    requested = target['vehicles']
    actors_spec = parameters['actors']
    if not isinstance(requested, list) or not isinstance(actors_spec, list) or len(requested) != len(actors_spec) or len(requested) > 7:
        raise ValueError('vehicles and actors must be matching lists of at most seven')
    matches = [row for row in backgrounds if row['path'].relative_to(ROOT).as_posix() == background['image']]
    if len(matches) != 1:
        raise ValueError('declared background is not an eligible source')
    selected = matches[0]
    crop = background['crop_xyxy']
    if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError('crop must contain four integer coordinates')
    gain, gamma = augmentation['gain'], augmentation['gamma']
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (gain, gamma)) or not .78 <= gain <= 1.20 or not .92 <= gamma <= 1.10:
        raise ValueError('exposure is outside the supported range')
    original = scene.read_color(selected['path'])
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= original.shape[1] and 0 <= y0 < y1 <= original.shape[0]):
        raise ValueError('invalid background crop')
    base = original[y0:y1, x0:x1].copy()
    if base.shape[:2] != (scene.HEIGHT, scene.WIDTH):
        base = cv2.resize(base, (scene.WIDTH, scene.HEIGHT), interpolation=cv2.INTER_LINEAR)
    actors, used_sources = [], {background['image'], background_source}
    keys = {'foot_x', 'foot_y', 'size_px', 'yaw', 'pitch', 'roll', 'relative_exposure'}
    for target_item, spec in zip(requested, actors_spec):
        if not isinstance(target_item, dict) or set(target_item) != {'class', 'source'} or target_item['class'] not in ('upright', 'fallen'):
            raise ValueError('unsupported vehicle target')
        if not isinstance(spec, dict) or set(spec) != keys:
            raise ValueError('actor must specify position, size, angles and exposure')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in spec.values()):
            raise ValueError('actor parameters must be finite numbers')
        if type(spec['size_px']) is not int or not 1 <= spec['size_px'] <= 960 or not -12 <= spec['yaw'] <= 12 or not -6 <= spec['pitch'] <= 6 or not -3 <= spec['roll'] <= 3 or not .90 <= spec['relative_exposure'] <= 1.08:
            raise ValueError('actor parameters outside supported range')
        available = [(entry, sprite) for entry, sprite in vehicles[target_item['class']] if entry['path'] == target_item['source']]
        if len(available) != 1:
            raise ValueError('vehicle source missing, duplicated, or inconsistent with class')
        entry, sprite = available[0]
        used_sources.update(entry[key] for key in ('path', 'source_frame', 'source_video', 'source_image') if key in entry)
        sprite = planar_warp(sprite, spec['yaw'], spec['pitch'], spec['roll'])
        sprite = scene.assets.resized(sprite, **({'height': spec['size_px']} if target_item['class'] == 'upright' else {'width': spec['size_px']}))
        sprite = sprite.copy()
        sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.float32) * spec['relative_exposure'], 0, 255).astype(np.uint8)
        actor = {'kind': 'vehicle', 'category': target_item['class'], 'source': entry['path'],
                 'sprite': sprite, 'foot_x': spec['foot_x'], 'foot_y': spec['foot_y'],
                 'angles_degrees': {key: spec[key] for key in ('yaw', 'pitch', 'roll')}}
        scene.validate_actor_placement(actor, 'ground_only')
        actors.append(actor)
    if state['sources'] != sorted(used_sources):
        raise ValueError('sources must be the sorted, complete set of actual background and vehicle sources')
    actors.sort(key=lambda actor: actor['foot_y'])
    for actor in actors:
        scene.draw_shadow(base, actor)
    full = [alpha_place(base, actor['sprite'], actor['foot_x'], actor['foot_y']) for actor in actors]
    visible = visible_masks(full)
    objects = []
    for actor, mask, shown in zip(actors, full, visible):
        if shown.sum() < 120 or shown.sum() / mask.sum() < .55:
            raise ValueError('vehicle visibility below the declared supported constraint')
        objects.append({'class': actor['category'], 'bbox_xyxy': scene.assets.box(shown),
                        'visible_fraction': float(shown.sum() / mask.sum()),
                        'mask_source': actor['source'], 'angles_degrees': actor['angles_degrees']})
    base = scene.exposure(base, gain, gamma)
    info = {'frame': selected['frame'], 'seconds': selected['seconds'], 'image': background['image'],
            'source_video': background_source, 'source_group': background_group, 'crop_xyxy': crop}
    return base, objects, visible, info


def one(index, rng, backgrounds, people, vehicles, background_source=scene.VIDEO,
        background_group='scene_video_49s_shared'):
    kind, classes = scenario(index)
    selected = rng.choice(backgrounds)
    original = scene.read_color(selected['path'])
    x0, y0, x1, y1 = selected.get('crop_xyxy', scene.GROUND_CROP)
    if not (0 <= x0 < x1 <= original.shape[1] and 0 <= y0 < y1 <= original.shape[0]):
        raise ValueError('invalid background crop')
    base = original[y0:y1, x0:x1].copy()
    if base.shape[:2] != (scene.HEIGHT, scene.WIDTH):
        base = cv2.resize(base, (scene.WIDTH, scene.HEIGHT), interpolation=cv2.INTER_LINEAR)
    # Draw a common row of perspective parking slots and place the bikes in
    # those slots. This synthetic layout is not a measured field layout.
    marking = (background_group != 'vehicle_camera_pics_zip_session_uncertain' and
               (index // 20 + index % 3) % 2 == 0)
    num_slots = 7 if len(classes) > 4 else 4
    spacing = 130 if num_slots == 7 else 230
    left_edge = 20
    if marking:
        for edge in (left_edge + spacing*i for i in range(num_slots+1)):
            cv2.line(base, (edge, 210), (max(0, edge - 45), 479), (185, 187, 185), 5, cv2.LINE_AA)
    row_y = scene.truncated_gauss(rng, 375, 12, 350, 400)
    slots = sorted(rng.sample(range(num_slots), len(classes)))
    actors = []
    size_band = ('small', 'medium', 'large')[(index // 20 + index % 3) % 3]
    if len(classes) >= 3 and size_band == 'large':
        size_band = 'medium'
    if len(classes) > 4:
        size_band = 'small'
    height_range = {'small': (85, 115), 'medium': (145, 180), 'large': (220, 260)}[size_band]
    for slot, cls in zip(slots, classes):
        source, sprite = rng.choice(vehicles[cls])
        foot = row_y + rng.uniform(-10, 10)
        x = left_edge + spacing*(slot + .5) - 45*(foot-210)/269 + rng.uniform(-8, 8)
        height = rng.uniform(*height_range)
        actors.append(scene.random_actor(rng, 'vehicle', cls, sprite, source, x, foot, height))
    # Some positive examples include a foreground or background cardboard person.
    if people and classes and index % 4 == 1:
        cls = 'community' if (index // 4) % 2 else 'noncommunity'
        source, sprite = rng.choice(people[cls])
        nearest = actors[0]
        foot = nearest['foot_y'] + rng.choice((-12, 12))
        x = nearest['foot_x'] + rng.choice((-35, 35))
        actors.append(scene.random_actor(rng, 'person', cls, sprite,
                      source.relative_to(ROOT).as_posix(), x, foot,
                      nearest['person_reference_height_px']))
    actors.sort(key=lambda actor: actor['foot_y'])
    for actor in actors:
        scene.validate_actor_placement(actor, 'ground_only')
        scene.draw_shadow(base, actor)
    masks = [alpha_place(base, actor['sprite'], actor['foot_x'], actor['foot_y']) for actor in actors]
    shown = visible_masks(masks)
    person_sources = [actor['source'] for actor in actors if actor['kind'] == 'person']
    person_overlap = any(np.any(masks[i] & masks[j])
                         for i in range(len(actors)) for j in range(len(actors))
                         if actors[i]['kind'] == 'person' and actors[j]['kind'] == 'vehicle')
    level = ('dim', 'normal', 'bright')[(index // 20 + index % 20) % 3]
    params = {'dim': (.86, .025, .82, .89), 'normal': (1., .045, .90, 1.08),
              'bright': (1.125, .018, 1.10, 1.15)}
    gain = scene.truncated_gauss(rng, *params[level])
    gamma = scene.truncated_gauss(rng, 1., .025, .95, 1.06)
    base = scene.exposure(base, gain, gamma)
    objects, masks_out = [], []
    for actor, full, visible in zip(actors, masks, shown):
        if actor['kind'] != 'vehicle':
            continue
        if visible.sum() < 120 or visible.sum()/full.sum() < .55:
            raise ValueError('vehicle visibility too low')
        bbox = scene.assets.box(visible)
        objects.append({'class': actor['category'], 'bbox_xyxy': bbox,
                        'visible_fraction': float(visible.sum()/full.sum()),
                        'mask_source': actor['source'], 'angles_degrees': actor['angles_degrees']})
        masks_out.append(visible)
    return base, objects, masks_out, {'frame': selected['frame'], 'seconds': selected['seconds'],
            'image': selected['path'].relative_to(ROOT).as_posix(), 'source_video': background_source,
            'source_group': background_group, 'crop_xyxy': [x0, y0, x1, y1],
            'parking_lines_synthetic': marking, 'parking_slots': slots,
            'vehicle_size_band': size_band, 'person_sources': person_sources,
            'person_vehicle_overlap': person_overlap}, {'level': level, 'gain': gain, 'gamma': gamma}


def main():
    raise RuntimeError("Retired production API; use compose_v4_grounded_train.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan', required=True, help='Workspace-relative production declaration JSON')
    args = parser.parse_args()
    out = scene.path_in_workspace(args.output)
    allowed = (ROOT / 'datasets/exports/e_bike', ROOT / 'datasets/exports/authoritative/e_bike')
    if not any(out.is_relative_to(parent) and out != parent for parent in allowed):
        parser.error('output must be a new batch under an e-bike export directory')
    plan_path = scene.path_in_workspace(args.plan)
    declaration = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    validate_declaration(declaration)
    backgrounds = scene.eligible_backgrounds()
    _, vehicles = sources(SELECTION, include_people=False)
    bg_source, bg_group = scene.VIDEO, 'scene_video_49s_shared'
    procedural = [plan_path, SELECTION, scene.BACKGROUND_ROOT / 'reports/backgrounds_manifest.csv',
                  ROOT/'datasets/scripts/e_bike/config/source.yaml', ROOT/'datasets/scripts/e_bike/config/dataset.yaml']
    media = [scene.path_in_workspace(item) for state in declaration.get('strata', []) for item in state.get('sources', [])]
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    operation = 'e_bike.synthesize_vehicle_scene:' + out.relative_to(ROOT).as_posix()
    codes = [Path(__file__), Path(scene.__file__), Path(scene.assets.__file__),
             ROOT / 'datasets/scripts/common/perspective.py', ROOT / 'datasets/scripts/common/composite.py', ROOT / 'datasets/scripts/common/annotations.py', ROOT / 'lab/training_common/production_guard.py', ROOT/'datasets/scripts/common/config.py']
    with generation_guard(operation, list(dict.fromkeys(procedural + media)), [out],
                          {'plan': args.plan}, code_paths=list(dict.fromkeys(codes)),
                          procedural_inputs=procedural, root=ROOT, production_plan=declaration,
                          sample_output_paths=[out / 'images'], auxiliary_output_paths=[out / 'masks']) as run:
        if run.reused:
            return
        if out.exists():
            parser.error('output must be a new directory')
        # Validate every declared state before any output is created.
        states = run.plan['strata']
        for state in states:
            planned_scene(state, backgrounds, vehicles, bg_source, bg_group)
        out.mkdir(parents=True)
        for name in ('images', 'labels', 'masks'):
            (out/name).mkdir()
        summary = {'status': 'running', 'requested': declaration['budget'], 'completed': 0,
                   'source_group': bg_group, 'split': 'train', 'classes': {'upright': 0, 'fallen': 1},
                   'limits': 'source masks are candidates; acquisition-session separation is unverified',
                   'plan': args.plan, 'states': {}}
        def save():
            (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        save()
        try:
            with (out/'manifest.jsonl').open('w', encoding='utf8') as manifest:
                index = 0
                for state in states:
                  image, objects, masks, bg = planned_scene(state, backgrounds, vehicles, bg_source, bg_group)
                  for _ in range(state['planned_count']):
                    stem = f'bike_{index+1:06d}'
                    scene.write_image(out/'images'/f'{stem}.jpg', image)
                    labels = []
                    for j, (obj, mask) in enumerate(zip(objects, masks)):
                        path = f'masks/{stem}_{j:02d}.png'
                        scene.write_image(out/path, mask.astype(np.uint8)*255)
                        obj['visible_mask'] = path
                        labels.append(scene.yolo_row(scene.VEHICLE_CLASSES[obj['class']], obj['bbox_xyxy'], scene.WIDTH, scene.HEIGHT))
                    (out/'labels'/f'{stem}.txt').write_text('\n'.join(labels)+('\n' if labels else ''), encoding='utf8')
                    actual = {'background': bg, 'objects': objects, 'augmentation': state['augmentation'],
                              'parameters': state['parameters'], 'media': state['sources']}
                    run.record_sample(state['id'], output=out/'images'/f'{stem}.jpg',
                                      sources=state['sources'], parameters=state['parameters'], actual=actual)
                    summary['states'][state['id']] = summary['states'].get(state['id'], 0)+1
                    manifest.write(json.dumps({'image': f'images/{stem}.jpg', 'label': f'labels/{stem}.txt',
                        'state_id': state['id'], 'objects': objects, 'background': bg,
                        'augmentation': state['augmentation'], 'parameters': state['parameters'],
                        'sources': state['sources'], 'status': 'synthetic_candidate'}, ensure_ascii=False)+'\n')
                    summary['completed'] = index+1
                    if index % 50 == 0:
                        manifest.flush(); save()
                    index += 1
            summary['status'] = 'complete_candidates'
        except BaseException as exc:
            summary['status'] = 'failed_or_interrupted'; summary['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            save()
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
