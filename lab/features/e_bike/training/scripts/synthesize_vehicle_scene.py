"""Generate source-grouped e-bike detection candidates using recorded ground.

Classes are upright=0 and fallen=1. The video-derived foreground masks are
review candidates; output is not automatically verified ground truth.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/person/training/scripts'))
import synthesize_real_ground as scene

SELECTION = ROOT / 'datasets/annotated/e_bike/sam2_orbit_candidates_20261110/selection_review_candidate.json'
FIELD_SELECTION = ROOT / 'datasets/annotated/e_bike/field_eval_masks_20261110/selection_review_candidate.json'
FIELD_BACKGROUND = ROOT / 'datasets/annotated/e_bike/video_review_20261110/images/source_003/frame_000015.jpg'
FIELD_VIDEO = 'Archive/Assets/电动车与人物/视频/单辆直立电动车_低视角环绕_005.mp4'
CAMERA_SELECTION = ROOT / 'datasets/annotated/e_bike/vehicle_camera_eval_masks_20261110/selection_review_candidate.json'
CAMERA_BACKGROUND = ROOT / 'Archive/Assets/红绿灯/照片/小车摄像头实拍_近似平视_信号灯正面与道路_001.ppm'


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
    people = scene.load_sources()[0] if include_people else {}
    data = json.loads(selection.read_text(encoding='utf-8'))
    vehicles = {'upright': [], 'fallen': []}
    for entry in data['vehicles'] if 'vehicles' in data else data['records']:
        for key in ('path', 'source_frame', 'source_video'):
            if not scene.path_in_workspace(entry[key]).is_file():
                raise ValueError(f'missing source: {entry[key]}')
        vehicles[entry['class']].append((entry, scene.assets.load_rgba(scene.path_in_workspace(entry['path']))))
    return people, vehicles


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
    masks = [scene.alpha_place(base, actor['sprite'], actor['foot_x'], actor['foot_y']) for actor in actors]
    shown = scene.visible_masks(masks)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--seed', type=int, default=8127)
    parser.add_argument('--mode', choices=('train_candidate', 'field_validation_candidate',
                                            'camera_test_candidate_pending_source_audit'),
                        default='train_candidate')
    args = parser.parse_args()
    out = scene.path_in_workspace(args.output)
    if not out.is_relative_to(ROOT / 'datasets/exports/e_bike') or out.exists():
        parser.error('output must be a new directory under datasets/exports/e_bike')
    if not 1 <= args.count <= 100000:
        parser.error('count must be between 1 and 100000')
    if args.mode == 'train_candidate':
        backgrounds, (people, vehicles) = scene.eligible_backgrounds(), sources()
        bg_source, bg_group = scene.VIDEO, 'scene_video_49s_shared'
    elif args.mode == 'field_validation_candidate':
        if not FIELD_BACKGROUND.is_file():
            parser.error('field validation background missing')
        backgrounds = [{'path': FIELD_BACKGROUND, 'frame': 15, 'seconds': None}]
        people, vehicles = sources(FIELD_SELECTION, include_people=False)
        bg_source, bg_group = FIELD_VIDEO, 'field_20261108_session_candidate'
    else:
        if not CAMERA_BACKGROUND.is_file():
            parser.error('camera background missing')
        backgrounds = [{'path': CAMERA_BACKGROUND, 'frame': None, 'seconds': None,
                        'crop_xyxy': [0, 160, 640, 480]}]
        people, vehicles = sources(CAMERA_SELECTION, include_people=False)
        bg_source = CAMERA_BACKGROUND.relative_to(ROOT).as_posix()
        bg_group = 'vehicle_camera_pics_zip_session_uncertain'
    out.mkdir(parents=True)
    for name in ('images', 'labels', 'masks'):
        (out/name).mkdir()
    summary = {'status': 'running', 'requested': args.count, 'completed': 0,
               'seed': args.seed, 'source_group': bg_group,
               'split': args.mode, 'classes': {'upright': 0, 'fallen': 1},
               'allocation': '40% upright-only, 40% fallen-present, 20% no vehicle',
               'limits': 'same model; masks require review; acquisition-session separation is unverified',
               'scenarios': {}, 'lighting': {}}
    def save():
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save()
    try:
        with (out/'manifest.jsonl').open('w', encoding='utf8') as manifest:
            for index in range(args.count):
                rng = random.Random(args.seed + 1009*index)
                for attempt in range(1, 101):
                    try:
                        image, objects, masks, bg, lighting = one(index, rng, backgrounds, people, vehicles,
                                                                  bg_source, bg_group)
                        break
                    except ValueError:
                        if attempt == 100:
                            raise RuntimeError(f'could not generate scene {index}')
                stem = f'bike_{index+1:06d}'
                scene.write_image(out/'images'/f'{stem}.jpg', image)
                labels = []
                for j, (obj, mask) in enumerate(zip(objects, masks)):
                    path = f'masks/{stem}_{j:02d}.png'
                    scene.write_image(out/path, mask.astype(np.uint8)*255)
                    obj['visible_mask'] = path
                    labels.append(scene.yolo_row(scene.VEHICLE_CLASSES[obj['class']], obj['bbox_xyxy'], scene.WIDTH, scene.HEIGHT))
                (out/'labels'/f'{stem}.txt').write_text('\n'.join(labels)+('\n' if labels else ''), encoding='utf8')
                category = scenario(index)[0]
                summary['scenarios'][category] = summary['scenarios'].get(category, 0)+1
                summary['lighting'][lighting['level']] = summary['lighting'].get(lighting['level'], 0)+1
                manifest.write(json.dumps({'image': f'images/{stem}.jpg', 'label': f'labels/{stem}.txt',
                    'scenario': category, 'objects': objects, 'background': bg, 'lighting': lighting,
                    'attempt': attempt, 'status': 'synthetic_candidate_needs_visual_review'}, ensure_ascii=False)+'\n')
                summary['completed'] = index+1
                if index % 50 == 0:
                    manifest.flush(); save()
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'; summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
