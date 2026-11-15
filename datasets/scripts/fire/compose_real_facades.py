"""Compose the configured real facade crops using shared Bayesian allocation.

The existing three crop/patch observations are candidate annotations. Original
photos are read directly; removed runs inspection images are not dependencies.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.images import read_image
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.paper import paste
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.common.paths import sample_dir
from lab.training_common.production_guard import generation_guard

TASK = 'fire'


def selected_sources(config, data):
    """Use one atomic task/split/role relation per configured reviewed crop."""
    patches = config['patches']
    selected = {}
    wanted = tuple(config[k] for k in ('task', 'split', 'role', 'status'))
    for record in data['records']:
        for relation in record.get('task_use_status', []):
            key = relation.get('source_id')
            if key in patches and tuple(relation.get(k) for k in ('task', 'split', 'role', 'status')) == wanted:
                if key in selected:
                    raise ValueError('ambiguous facade source relation: ' + key)
                selected[key] = {'id': key, 'image': record['path'],
                                 'parent': record.get('source_group', record['path']),
                                 'quad': patches[key]}
    if set(selected) != set(patches) or len({r['image'] for r in selected.values()}) != len(patches):
        raise ValueError('configured facade crops require distinct matching source relations')
    return list(selected.values())


def configured_sources():
    _, source = load_task_config(TASK)
    config = source['real_facades']
    relations = (ROOT / config['relations']).resolve()
    if not relations.is_relative_to(ROOT) or not relations.is_file():
        raise ValueError('real facade relations are missing or outside workspace')
    rows = selected_sources(config, json.loads(relations.read_text(encoding='utf8')))
    parents = []
    for row in rows:
        for field in ('image', 'parent'):
            path = (ROOT / row[field]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError('facade source is missing: ' + row[field])
            parents.append(path)
    return config, relations, rows, list(dict.fromkeys(parents))


def build_plan(config, rows, count, seed):
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError('count must be 1..10000')
    if config['supported_counts'] != [0, 1]:
        raise ValueError('the recorded single wall patch supports zero or one sticker')
    states = []
    artwork = load_task_config(TASK)[1]['target_artwork']
    for row in rows:
        for number in config['supported_counts']:
            states.append({'id': row['id'] + '_count_' + str(number),
                'failure_event': 'fire_sticker_missed_or_false_positive',
                'min_count': 0, 'max_count': count,
                'prior': {'alpha': 2 if number else 1, 'beta': 4},
                'exposure_weight': 1, 'failure_cost': 1.2 if number else 1,
                'conditions': [row['id'], 'count:' + str(number)],
                'background': {'image': row['image'], 'parent': row['parent'],
                               'crop_xyxy': config['crop_xyxy'], 'review_state': config['review_state']},
                'target': {'kind': 'fire_sticker', 'count': number, 'quad': row['quad'], 'artwork': artwork},
                'labels': {'class_id': 0, 'format': 'yolo', 'status': 'crop_candidate'},
                'augmentation': config['augmentation'],
                'sources': list(dict.fromkeys([row['image'], row['parent'], artwork])), 'parameters': {'seed': seed, 'source_id': row['id']}})
    return {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': config['split'],
            'purpose': 'real facade crop composition; design priors', 'budget': count,
            'uncertainty_weight': .5, 'independence_basis': 'three parent photo paths; capture independence unknown',
            'constraints': {'labels': 'crop_candidate', 'counts': [0, 1]}, 'strata': states}


def configured_plan(config, rows, plan_path):
    """The saved current plan must describe the current configured crop sources."""
    request = json.loads(plan_path.read_text(encoding='utf8'))
    seeds = {s['parameters']['seed'] for s in request['strata']}
    if len(seeds) != 1:
        raise ValueError('current fire plan requires one declared seed')
    seed = seeds.pop()
    expected = build_plan(config, rows, request['budget'], seed)
    if request != expected:
        raise ValueError('saved fire plan differs from source.yaml crops, regions or parameters')
    return request, seed


def compose_frame(image, artwork, state, config, seed):
    """Pure array transform; no source discovery or filesystem output."""
    if list(image.shape[1::-1]) != config['source_image_size']:
        raise ValueError('facade parent image dimensions differ from recorded crop')
    x0, y0, x1, y1 = config['crop_xyxy']
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError('crop bounds are outside parent')
    result = image[y0:y1, x0:x1].copy()
    rng = random.Random(seed)
    boxes = []
    if state['target']['count']:
        paper = np.full((140, 100, 3), 255, np.uint8)
        paper[5:134] = cv2.resize(artwork, (114, 129), interpolation=cv2.INTER_AREA)[:, 7:107]
        quad = np.asarray(state['target']['quad'], dtype=np.float32)
        quad *= np.array([result.shape[1], result.shape[0]]) / np.array(config['region_reference_size'])
        scale = rng.uniform(*config['augmentation']['patch_scale'])
        center = quad.mean(axis=0)
        quad = center + (quad - center) * scale
        boxes.append(paste(result, paper, quad))
    contrast = rng.uniform(*config['augmentation']['contrast'])
    brightness = rng.uniform(*config['augmentation']['brightness'])
    saturation = rng.uniform(*config['augmentation']['saturation'])
    result = np.clip(result.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), boxes, {
        'contrast': contrast, 'brightness': brightness, 'saturation': saturation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=load_task_config(TASK)[1]['production_plan'])
    parser.add_argument('--output', default='datasets/samples/fire')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output != sample_dir(TASK):
        raise ValueError('crop candidate labels currently support only datasets/samples/fire')
    config, relations, rows, media = configured_sources()
    artwork = source_path(TASK, 'target_artwork')
    plan_path = (ROOT / args.plan).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('fire plan must be an existing workspace JSON')
    declaration, seed = configured_plan(config, rows, plan_path)
    planned = plan_production(declaration, ROOT)
    configs = [ROOT / f'datasets/scripts/{TASK}/config/{name}.yaml' for name in ('dataset', 'source')]
    with generation_guard('fire.real_facades:' + output.relative_to(ROOT).as_posix(),
            [*media, artwork], [output], vars(args),
            code_paths=[Path(__file__), ROOT/'datasets/scripts/common/paper.py',
                        ROOT/'datasets/scripts/common/perspective.py', ROOT/'datasets/scripts/common/bayesian.py',
                        ROOT/'datasets/scripts/common/images.py', ROOT/'datasets/scripts/common/annotations.py'],
            procedural_inputs=[relations, plan_path, *configs], root=ROOT, production_plan=declaration,
            sample_output_paths=[output/'images']) as run:
        if run.reused:
            return
        if output.exists() and any(output.iterdir()):
            raise ValueError('current sample directory is not empty')
        (output/'images').mkdir(parents=True, exist_ok=True)
        (output/'labels_candidate').mkdir(exist_ok=True)
        states = [state for state in run.plan['strata'] for _ in range(state['planned_count'])]
        random.Random(seed).shuffle(states)
        icon = read_image(artwork)
        records = []
        for index, state in enumerate(states):
            result, boxes, tone = compose_frame(read_image(ROOT/state['background']['image']), icon, state, config, seed+index)
            stem = f'fire_{index:06d}'
            image_path, label_path = output/'images'/f'{stem}.png', output/'labels_candidate'/f'{stem}.txt'
            ok, encoded = cv2.imencode('.png', result)
            if not ok:
                raise ValueError('image encoding failed')
            encoded.tofile(str(image_path))
            label_path.write_text(''.join(yolo_row(b, result.shape[1], result.shape[0])+'\n' for b in boxes), encoding='utf8')
            actual = {'image': image_path.relative_to(ROOT).as_posix(), 'label': label_path.relative_to(ROOT).as_posix(),
                'parent': state['background']['parent'], 'source_image': state['background']['image'],
                'crop_xyxy': config['crop_xyxy'], 'count': state['target']['count'], 'tone': tone,
                'generation_method': 'real_photo_crop_plus_digital_artwork', 'label_status': 'crop_candidate'}
            run.record_sample(state['id'], output=image_path, sources=state['sources'], parameters=state['parameters'], actual=actual)
            records.append(actual)
        (output/'manifest.json').write_text(json.dumps({'plan': planned, 'records': records}, ensure_ascii=False, indent=2), encoding='utf8')
        run.set_result({'count': len(records), 'status': 'crop_candidates'})


if __name__ == '__main__':
    main()
