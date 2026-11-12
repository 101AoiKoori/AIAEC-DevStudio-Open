"""Create source-separated synthetic person evaluation candidates, not training.

Val uses the person scan video and 005/006 vehicle videos; test uses the
pics.zip camera images. Recording sessions and shared physical figures remain
unverified; generated masks and labels need visual/human review.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import synthesize_real_ground as shared

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'datasets/annotated/person/eval_sources_20261110/manifest.json'
VEHICLES = {
    'val': ROOT / 'datasets/annotated/e_bike/field_eval_masks_20261110/selection_review_candidate.json',
    'test': ROOT / 'datasets/annotated/e_bike/vehicle_camera_eval_masks_20261110/selection_review_candidate.json',
}
WIDTH, HEIGHT = 960, 480


def workspace_file(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('source not found in workspace: ' + str(value))
    return path


def load_inputs(split):
    manifest = json.loads(SOURCE.read_text(encoding='utf8'))
    if manifest.get('status') != 'complete_candidates':
        raise ValueError('person source preparation incomplete')
    people = {}
    for row in manifest['records']:
        if row['split'] == split:
            category = 'community' if row['class_id'] == 0 else 'noncommunity'
            people[category] = (row, shared.assets.load_rgba(workspace_file(row['sprite'])))
    if set(people) != {'community', 'noncommunity'}:
        raise ValueError('missing person class')
    bginfo = manifest['backgrounds'][split]
    image = shared.read_color(workspace_file(bginfo['path']))
    background = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    vehicle_manifest = json.loads(VEHICLES[split].read_text(encoding='utf8'))
    vehicles = {'upright': [], 'fallen': []}
    for row in vehicle_manifest['records']:
        if row['class'] not in vehicles:
            raise ValueError('unknown vehicle category')
        for field in ('path', 'source_frame', 'source_video'):
            workspace_file(row[field])
        vehicles[row['class']].append((row, shared.assets.load_rgba(workspace_file(row['path']))))
    if any(not values for values in vehicles.values()):
        raise ValueError('missing vehicle pose')
    return people, vehicles, background, bginfo, vehicle_manifest['source_group']


def make_sample(index, rng, people, vehicles, background):
    plan = shared.scenarios(index)
    occurrence = shared.scenario_occurrence(index)
    level = shared.stratified_lighting_level(index)
    gain = {'dim': .86, 'normal': 1.0, 'bright': 1.12}[level]
    gamma = shared.truncated_gauss(rng, 1, .025, .95, 1.06)
    anchors, actors = [], []
    for i in range(plan['people']):
        category = 'community' if (occurrence + i) % 2 == 0 else 'noncommunity'
        record, image = people[category]
        for _ in range(100):
            x = rng.uniform(170, 790)
            if all(abs(x - p[0]) >= 110 for p in anchors):
                break
        else:
            raise ValueError('cannot place separate actors')
        y = rng.uniform(335, 438)
        height = 105 + (y - 335) * .55 + rng.uniform(-8, 8)
        actor = shared.random_actor(rng, 'person', category, image, record['sprite'], x, y, height)
        actor['origin'] = record
        actors.append(actor)
        anchors.append((x, y, height))
    for i in range(plan['vehicles']):
        category = 'upright' if (occurrence // 2 + i) % 2 == 0 else 'fallen'
        record, image = rng.choice(vehicles[category])
        if anchors:
            x, y, height = anchors[i % len(anchors)]
            x += (-1 if i % 2 else 1) * rng.uniform(15, 35)
            y += rng.uniform(5, 12) if plan['relation'] == 'vehicle_front' else -rng.uniform(5, 12)
        else:
            x, y, height = rng.uniform(180, 780), rng.uniform(335, 438), rng.uniform(105, 160)
        actors.append(shared.random_actor(rng, 'vehicle', category, image, record, x, y, height))
    actors.sort(key=lambda actor: actor['foot_y'])
    canvas = background.copy()
    for actor in actors:
        shared.validate_actor_placement(actor, 'ground_only')
        shared.draw_shadow(canvas, actor)
    masks = [shared.alpha_place(canvas, actor['sprite'], actor['foot_x'], actor['foot_y']) for actor in actors]
    if plan['kind'] == 'occlusion' and not shared.has_requested_overlap(
            masks, [actor['kind'] for actor in actors], plan['relation']):
        raise ValueError('requested occlusion absent')
    visible = shared.visible_masks(masks)
    canvas = shared.exposure(canvas, gain, gamma)
    records = []
    for actor, full, shown in zip(actors, masks, visible):
        fraction = float(shown.sum() / full.sum())
        if fraction < .55 or shown.sum() < 120:
            raise ValueError('target too occluded')
        box = shared.assets.box(shown)
        records.append({'kind': actor['kind'], 'category': actor['category'], 'source': actor['source'],
                        'origin': actor.get('origin'), 'visible_bbox_xyxy': box,
                        'visible_fraction': fraction, 'angles_degrees': actor['angles_degrees']})
    return canvas, masks, visible, records, plan, {'level': level, 'gain': gain, 'gamma': gamma}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261120)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'datasets/exports/person').resolve()) or output.exists():
        parser.error('output must be a new directory in datasets/exports/person')
    if args.count < 1 or args.count > 10000:
        parser.error('count must be between 1 and 10000')
    inputs = {split: load_inputs(split) for split in ('val', 'test')}
    output.mkdir(parents=True)
    summary = {'status': 'running', 'requested_per_split': args.count, 'completed': {'val': 0, 'test': 0},
        'seed': args.seed, 'purpose': 'synthetic person evaluation candidates; not training or verified truth',
        'limits': 'recording session relation to training unknown; same physical figure designs and vehicle model; masks need human review',
        'source_groups': {'val': ['person_scan_012', inputs['val'][4]], 'test': ['pics_zip_camera_session_uncertain']},
        'scenarios': {}, 'brightness': {}}
    def save():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save()
    try:
        for split, seed_offset in (('val', 0), ('test', 1000000)):
            for name in ('images', 'labels', 'masks/full', 'masks/visible'):
                (output/split/name).mkdir(parents=True)
            people, vehicles, background, bginfo, vehicle_group = inputs[split]
            scenes, brightness = Counter(), Counter()
            with (output/split/'manifest.jsonl').open('w', encoding='utf8') as manifest:
                for i in range(args.count):
                    rng = random.Random(args.seed + seed_offset + i * 1009)
                    for attempt in range(1, 201):
                        try:
                            image, full, visible, objects, plan, light = make_sample(i, rng, people, vehicles, background)
                            break
                        except ValueError:
                            if attempt == 200:
                                raise RuntimeError(f'cannot place {split} sample {i}')
                    stem = f'{split}_{i:06d}'
                    shared.write_image(output/split/'images'/(stem+'.png'), image)
                    labels = []
                    for j, (obj, fm, vm) in enumerate(zip(objects, full, visible)):
                        fullpath, visiblepath = f'masks/full/{stem}_{j:02d}.png', f'masks/visible/{stem}_{j:02d}.png'
                        shared.write_image(output/split/fullpath, fm.astype(np.uint8)*255)
                        shared.write_image(output/split/visiblepath, vm.astype(np.uint8)*255)
                        obj.update({'full_mask': fullpath, 'visible_mask': visiblepath})
                        if obj['kind'] == 'person':
                            labels.append(shared.yolo_row(shared.PERSON_CLASSES[obj['category']], obj['visible_bbox_xyxy'], WIDTH, HEIGHT))
                    (output/split/'labels'/(stem+'.txt')).write_text('\n'.join(labels)+ ('\n' if labels else ''), encoding='utf8')
                    record = {'image': f'images/{stem}.png', 'label': f'labels/{stem}.txt',
                              'scenario': plan, 'background': bginfo, 'vehicle_source_group': vehicle_group,
                              'lighting': light, 'objects': objects, 'split_candidate': split,
                              'status': 'synthetic_AI_candidate_not_human_truth', 'attempt': attempt}
                    manifest.write(json.dumps(record, ensure_ascii=False)+'\n')
                    scenes[plan['kind']] += 1
                    brightness[light['level']] += 1
                    summary['completed'][split] = i + 1
            summary['scenarios'][split] = dict(scenes)
            summary['brightness'][split] = dict(brightness)
            save()
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
