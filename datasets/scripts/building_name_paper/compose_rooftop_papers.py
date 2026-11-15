"""Compose planned rooftop paper candidates from explicitly described backgrounds.

All region and label declarations are candidates, not proof of image contents.
Coordinates are in the unresized source image, with quads ordered TL, TR, BR, BL.
The entry point requires the workspace production guard and a train-only plan.
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.images import read_image  # noqa: E402
from datasets.scripts.common.paper import name_paper as _name_paper, paste  # noqa: E402
from datasets.scripts.common.annotations import yolo_row  # noqa: E402
from datasets.scripts.common.names import CHARS, SUFFIX, make_name  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from datasets.scripts.common.bayesian import plan_production  # noqa: E402
from datasets.scripts.common.config import source_path, load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir

CONSTRAINTS = {'route_context': 'building_facade', 'traffic_signal_injection': 'none',
               'text_mirror': False, 'label_usage': 'preserve_source_coverage_and_generated_truth'}
LABELS = {'task': 'paper_detection', 'class_id': 0, 'format': 'yolo_v8',
          'status': 'source_recorded_or_generated_detection', 'generated_ocr': 'generated_string'}

FONT = source_path('building_name_paper', 'font')


def name_paper(text):
    return _name_paper(text, FONT)


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in '\r\n\t'):
        raise ValueError(label + ' must be a nonempty single-line string')
    return value


def _path(value, root=None):
    root = ROOT if root is None else Path(root).resolve()
    _text(value, 'path')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root or path.relative_to(root).as_posix() != value:
        raise ValueError('expected a canonical workspace-relative path')
    return path


def _size(value):
    if (not isinstance(value, list) or len(value) != 2 or
            any(type(x) is not int or x <= 0 for x in value)):
        raise ValueError('image_size must be [width, height]')
    return value


def _box(value, size):
    if (not isinstance(value, list) or len(value) != 4 or
            any(type(x) is not int for x in value)):
        raise ValueError('box must be integer xyxy')
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError('box outside image')
    return value


def _quad(value, size):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('quad requires four points')
    for point in value:
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in point) or
                not (0 <= point[0] < size[0] and 0 <= point[1] < size[1])):
            raise ValueError('quad point outside image')
    cross = []
    for i in range(4):
        a, b, c = value[i], value[(i + 1) % 4], value[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if min(cross) <= 0:
        raise ValueError('quad must be convex in TL, TR, BR, BL order')
    # Reject rotated starting points and obvious text mirroring.
    if value[0][1] + value[1][1] >= value[2][1] + value[3][1] or value[0][0] + value[3][0] >= value[1][0] + value[2][0]:
        raise ValueError('quad orientation would rotate or mirror text')
    return value


def _envelope(quad):
    return [min(p[0] for p in quad), min(p[1] for p in quad),
            max(p[0] for p in quad), max(p[1] for p in quad)]


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def existing_annotation(item, size):
    """Keep a detectable paper even when its text is unknown."""
    if set(item) != {'building_id', 'box_xyxy', 'ocr_status', 'text', 'evidence'}:
        raise ValueError('unsupported existing paper fields')
    _text(item['building_id'], 'building_id')
    _box(item['box_xyxy'], size)
    _text(item['evidence'], 'annotation evidence')
    if item['ocr_status'] == 'known':
        _text(item['text'], 'confirmed text')
    elif item['ocr_status'] == 'unknown':
        if item['text'] is not None:
            raise ValueError('unknown OCR text must be null')
    else:
        raise ValueError('OCR status must be known or unknown')
    return {'building_id': item['building_id'], 'box_xyxy': item['box_xyxy'],
            'detection_status': 'source_candidate', 'ocr_status': item['ocr_status'],
            'text': item['text'], 'evidence': item['evidence'], 'generated_text': None}


def validate_request(request, root=None):
    """Validate executable states before the allocator or any output operation."""
    root = ROOT if root is None else Path(root).resolve()
    _, source_config = load_task_config('building_name_paper')
    if source_config.get('background_selection') != 'production_plan_candidates':
        raise ValueError('unsupported configured background selection')
    parent_roots = source_config.get('background_parent_roots')
    if not isinstance(parent_roots, list) or not parent_roots:
        raise ValueError('source.yaml must declare background_parent_roots')
    parent_roots = [_path(value, root) for value in parent_roots]
    if (not isinstance(request, dict) or request.get('model_key') != 'building_name_paper' or
            request.get('target_split') != 'train' or request.get('constraints') != CONSTRAINTS):
        raise ValueError('expected the train-only rooftop candidate contract')
    states = request.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('states are required')
    for state in states:
        if not isinstance(state.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', state['id']):
            raise ValueError('unsafe state id')
        if (state.get('labels') != LABELS or state.get('augmentation') not in ({'mode': 'none'}, {'mode': 'scene_bounded_v1'}) or
                state.get('sources') != ['real_scene_background', 'name_font_asset']):
            raise ValueError('unsupported label, augmentation or source contract')
        if state.get('conditions') != ['rooftop_paper', 'no_added_occlusion']:
            raise ValueError('unsupported conditions')
        if state.get('failure_event') != 'paper_or_background_label_uncertainty':
            raise ValueError('unsupported failure event')
        if set(state.get('parameters', {})) != {'seed'} or type(state['parameters']['seed']) is not int or not 0 <= state['parameters']['seed'] < 2**63:
            raise ValueError('bounded integer seed required')
        target = state.get('target', {})
        if set(target) != {'kind', 'papers'} or target['kind'] != 'building_name_paper':
            raise ValueError('unsupported target')
        papers = target['papers']
        if not isinstance(papers, list):
            raise ValueError('papers must be a list; an empty list preserves complete source labels')
        slots = []
        for paper in papers:
            if set(paper) != {'slot_id', 'text'}:
                raise ValueError('unsupported generated paper fields')
            slots.append(_text(paper['slot_id'], 'slot_id'))
            _text(paper['text'], 'requested text')
        if len(slots) != len(set(slots)):
            raise ValueError('duplicate target slot')
        bg = state.get('background', {})
        if set(bg) != {'sampling', 'candidates'} or bg['sampling'] != 'uniform_group_then_image':
            raise ValueError('unsupported background sampling')
        candidates = bg['candidates']
        if not isinstance(candidates, list) or not candidates:
            raise ValueError('background candidates are required')
        seen = set()
        for candidate in candidates:
            if set(candidate) != {'image', 'source_group', 'parent_source', 'generation_method',
                                 'image_size', 'coverage', 'coverage_evidence', 'existing_papers', 'regions'}:
                raise ValueError('unsupported background fields')
            _path(candidate['image'], root)
            parent = _path(candidate['parent_source'], root)
            if not any(parent.is_relative_to(allowed) for allowed in parent_roots):
                raise ValueError('background parent is outside source.yaml background_parent_roots')
            _text(candidate['source_group'], 'source_group')
            _text(candidate['generation_method'], 'generation_method')
            size = _size(candidate['image_size'])
            if candidate['image'] in seen:
                raise ValueError('duplicate background in state')
            seen.add(candidate['image'])
            if candidate['coverage'] not in ('complete_candidate', 'unknown'):
                raise ValueError('coverage must be complete_candidate or unknown')
            if not papers and candidate['coverage'] != 'complete_candidate':
                raise ValueError('direct source states require complete recorded detection labels')
            _text(candidate['coverage_evidence'], 'coverage evidence')
            existing = candidate['existing_papers']
            if not isinstance(existing, list):
                raise ValueError('existing_papers must be a list')
            boxes = [existing_annotation(item, size)['box_xyxy'] for item in existing]
            regions = candidate['regions']
            if not isinstance(regions, list):
                raise ValueError('regions must be a list')
            by_slot = {}
            for region in regions:
                if set(region) not in ({'slot_id', 'building_id', 'quad_xy', 'evidence'},
                                       {'slot_id', 'building_id', 'quad_xy', 'evidence', 'replacement_of'}):
                    raise ValueError('unsupported region fields')
                slot = _text(region['slot_id'], 'region slot_id')
                if slot in by_slot:
                    raise ValueError('duplicate region slot')
                _text(region['building_id'], 'building_id')
                _text(region['evidence'], 'region evidence')
                by_slot[slot] = _quad(region['quad_xy'], size)
                if 'replacement_of' in region:
                    matches = [p for p in existing if p['building_id'] == region['replacement_of']]
                    if (len(matches) != 1 or candidate['coverage'] != 'complete_candidate' or
                            _envelope(region['quad_xy']) != matches[0]['box_xyxy']):
                        raise ValueError('replacement must cover exactly one recorded paper box on a complete source')
            if any(slot not in by_slot for slot in slots):
                raise ValueError('candidate lacks target region')
            for slot in slots:
                envelope = _envelope(by_slot[slot])
                region = next(r for r in regions if r['slot_id'] == slot)
                replaced = next((p['box_xyxy'] for p in existing if p['building_id'] == region.get('replacement_of')), None)
                remaining = list(boxes)
                if replaced is not None:
                    remaining.remove(replaced)
                if any(_intersects(envelope, box) for box in remaining):
                    raise ValueError('paper region overlaps an existing or planned paper')
                if replaced is None:
                    boxes.append(envelope)
    return request


def choose_background(candidates, rng):
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate['source_group'], []).append(candidate)
    group = rng.choice(sorted(groups))
    return rng.choice(sorted(groups[group], key=lambda value: value['image']))


def compose_one(image, candidate, papers):
    """Return BGR image and separate detection/OCR records; no files are written."""
    height, width = image.shape[:2]
    if [width, height] != candidate['image_size']:
        raise ValueError('source image dimensions differ from region coordinates')
    regions = {item['slot_id']: item for item in candidate['regions']}
    replaced = {regions[p['slot_id']].get('replacement_of') for p in papers}
    records = [existing_annotation(item, [width, height]) for item in candidate['existing_papers']
               if item['building_id'] not in replaced]
    for item in papers:
        region = regions[item['slot_id']]
        box = paste(image, name_paper(item['text']), region['quad_xy'])
        records.append({'building_id': region['building_id'], 'box_xyxy': box,
                        'quad_xy': region['quad_xy'], 'detection_status': 'generated_candidate',
                        'ocr_status': 'generated_string', 'text': item['text'], 'generated_text': item['text'],
                        'evidence': region['evidence']})
    return image, records


def _check_font_text(request):
    font = ImageFont.truetype(str(FONT), 98)
    for state in request['strata']:
        for item in state['target']['papers']:
            box = font.getbbox(item['text'])
            if box[2] - box[0] > 600 or box[3] - box[1] > 160:
                raise ValueError('requested text does not fit the existing paper graphic')


def current_detection_sources(request):
    """Bind the current full-frame plan to the recorded complete detections."""
    from datasets.scripts.building_name_paper.package_real_candidates import collect_selection
    selection = source_path('building_name_paper', 'direct_selection')
    rows = collect_selection(selection)
    indexed = {row['image']: (manifest, row, task, label)
               for manifest, index, row, task, image, label in rows}
    evidence = {selection, source_path('building_name_paper', 'combination_plan')}
    for state in request['strata']:
        for candidate in state['background']['candidates']:
            if candidate['image'] not in indexed:
                raise ValueError('current full-frame plan uses an image outside the complete-detection selection')
            manifest, row, task, label = indexed[candidate['image']]
            if (candidate['coverage'] != 'complete_candidate' or
                    candidate['parent_source'] != row['parent_video'] or
                    candidate['image_size'] != row['image_size'] or
                    [p['box_xyxy'] for p in candidate['existing_papers']] != task['boxes_xyxy']):
                raise ValueError('plan source boxes or parent differ from the complete source record')
            evidence.update((manifest, label))
    return sorted(evidence)


def augment_scene(image, papers, rng):
    """Warp the complete scene and every known target together; return actual parameters."""
    height, width = image.shape[:2]
    corners = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    # Zoom and skew the entire scene, retrying if any known paper would be cut.
    for _ in range(100):
        scale = rng.uniform(1.02, 1.12)
        crop_x = rng.uniform(.005, scale - 1 - .005) * (width - 1)
        crop_y = rng.uniform(.005, scale - 1 - .005) * (height - 1)
        left, top = -crop_x, -crop_y
        right = scale * (width - 1) - crop_x
        bottom = scale * (height - 1) - crop_y
        destination = [[left-rng.uniform(0,.003)*width, top-rng.uniform(0,.003)*height],
                       [right+rng.uniform(0,.003)*width, top-rng.uniform(0,.003)*height],
                       [right+rng.uniform(0,.003)*width, bottom+rng.uniform(0,.003)*height],
                       [left-rng.uniform(0,.003)*width, bottom+rng.uniform(0,.003)*height]]
        matrix = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination))
        if all(np.all((lambda q: (q[:,0] >= 1) & (q[:,0] < width-1) & (q[:,1] >= 1) & (q[:,1] < height-1))(
                cv2.perspectiveTransform(np.float32([p.get('quad_xy') or
                    [[p['box_xyxy'][0], p['box_xyxy'][1]], [p['box_xyxy'][2], p['box_xyxy'][1]],
                     [p['box_xyxy'][2], p['box_xyxy'][3]], [p['box_xyxy'][0], p['box_xyxy'][3]]]]), matrix)[0]))
               for p in papers):
            break
    else:
        raise ValueError('cannot keep known paper targets in the augmented image')
    image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    for paper in papers:
        x1, y1, x2, y2 = paper['box_xyxy']
        points = paper.get('quad_xy') or [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        transformed = cv2.perspectiveTransform(np.float32([points]), matrix)[0]
        paper['quad_xy'] = transformed.tolist()
        paper['box_xyxy'] = [max(0, math.floor(float(transformed[:,0].min()))),
                             max(0, math.floor(float(transformed[:,1].min()))),
                             min(width, math.ceil(float(transformed[:,0].max()))),
                             min(height, math.ceil(float(transformed[:,1].max())))]
    gain, bias = rng.uniform(.85, 1.15), rng.uniform(-10, 10)
    image = cv2.convertScaleAbs(image, alpha=gain, beta=bias)
    blur = rng.choice((0, 0, 3))
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image, {'homography': matrix.tolist(), 'destination_corners': destination,
                   'gain': gain, 'bias': bias, 'gaussian_kernel': blur,
                   'scale': scale, 'crop_xy': [crop_x, crop_y],
                   'sampling': 'uniform_scale_1.02_1.12_crop_inside_frame_outward_corner_jitter_0.003_gain_0.85_1.15_bias_-10_10; kernel_0_0_3; reject clipped known paper'}


def _generate(output, run):
    if output.exists() and any(output.iterdir()):
        raise ValueError('output already exists')
    (output / 'images').mkdir(parents=True)
    label_folder = 'labels' if output == split_dir('building_name_paper', 'train') else 'labels_candidate'
    (output / label_folder).mkdir()
    records = []
    for state in run.plan['strata']:
        rng = random.Random(state['parameters']['seed'])
        for index in range(state['planned_count']):
            candidate = choose_background(state['background']['candidates'], rng)
            targets = state['target']['papers']
            if state['augmentation']['mode'] == 'scene_bounded_v1':
                targets = [{'slot_id': p['slot_id'], 'text': make_name(rng.randrange(len(CHARS)*len(CHARS)*len(SUFFIX)))} for p in targets]
            image, papers = compose_one(read_image(ROOT / candidate['image']), candidate, targets)
            actual_augmentation = {'mode': 'none'}
            if state['augmentation']['mode'] == 'scene_bounded_v1':
                image, actual_augmentation = augment_scene(image, papers, rng)
            height, width = image.shape[:2]
            stem = f"{state['id']}_{index:06d}"
            image_path = output / 'images' / (stem + '.png')
            label_path = output / label_folder / (stem + '.txt')
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise ValueError('image encoding failed')
            encoded.tofile(str(image_path))
            label_path.write_text(''.join(yolo_row(p['box_xyxy'], width, height) + '\n' for p in papers), encoding='utf8')
            actual = {'source_image': candidate['image'], 'source_group': candidate['source_group'],
                      'parent_source': candidate['parent_source'],
                      'background_generation_method': candidate['generation_method'],
                      'generation_method': ('real_scene_plus_generated_name_paper' if state['target']['papers']
                                            else 'direct_real_with_recorded_detection_labels'),
                      'image_size': [width, height], 'coverage': candidate['coverage'],
                      'coverage_evidence': candidate['coverage_evidence'], 'papers': papers,
                      'augmentation': actual_augmentation,
                      'label_status': 'complete_candidate' if candidate['coverage'] == 'complete_candidate' else 'partial_candidate'}
            records.append({'image': image_path.relative_to(ROOT).as_posix(),
                            'label': label_path.relative_to(ROOT).as_posix(), 'state_id': state['id'], **actual})
            run.record_sample(state['id'], output=image_path, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
    complete = all(r['coverage'] == 'complete_candidate' for r in records)
    result = {'status': 'complete_detection_records' if complete else 'partial_detection_records', 'records': records,
              'source_refs': run.source_refs, 'scope': 'Source detections retain recorded coverage. Generated strings are known; original OCR and pixel readability retain their recorded status.'}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=load_task_config('building_name_paper')[1]['production_plan'], help='workspace-relative production declaration JSON')
    parser.add_argument('--output', required=True, help='current building_name_paper sample or train directory')
    args = parser.parse_args()
    plan_path = _path(args.plan)
    output = _path(args.output)
    if output not in {sample_dir('building_name_paper').resolve(), split_dir('building_name_paper', 'train').resolve()}:
        raise ValueError('output must be the building_name_paper sample or train directory')
    if not plan_path.is_file():
        raise ValueError('plan does not exist')
    request = validate_request(json.loads(plan_path.read_text(encoding='utf8')))
    if output == split_dir('building_name_paper', 'train') and any(
            c['coverage'] != 'complete_candidate' for s in request['strata'] for c in s['background']['candidates']):
        raise ValueError('current training plan excludes backgrounds without complete detection labels')
    source_evidence = current_detection_sources(request) if output == split_dir('building_name_paper', 'train') else []
    _check_font_text(request)
    planned = plan_production(request, ROOT)
    media = sorted({_path(c[key]) for s in request['strata'] for c in s['background']['candidates']
                    for key in ('image', 'parent_source')})
    code = [Path(__file__), ROOT / 'lab/training_common/bayesian_allocation.py', ROOT / 'datasets/scripts/common/bayesian.py',
            ROOT / 'datasets/scripts/common/paper.py', ROOT / 'datasets/scripts/common/perspective.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'datasets/scripts/common/names.py', ROOT / 'datasets/scripts/common/config.py',
            ROOT / 'datasets/scripts/common/paths.py', Path(__file__).with_name('package_real_candidates.py'),
            Path(__file__).with_name('plan_rooftop_batch.py')]
    configs = [ROOT/'datasets/scripts/building_name_paper/config'/name for name in ('dataset.yaml', 'source.yaml')]
    with generation_guard('building_name.compose_rooftop:' + args.output,
                          [*media, plan_path, FONT, *configs], [output], {}, code_paths=code,
                          procedural_inputs=[plan_path, FONT, *configs, *source_evidence], root=ROOT, production_plan=request,
                          sample_output_paths=[output / 'images']) as run:
        if run.plan != planned:
            raise ValueError('guard plan differs from requested allocation')
        if not run.reused:
            _generate(output, run)
    print(json.dumps({'planned': planned['planned_total'], 'output': args.output}, ensure_ascii=False))


if __name__ == '__main__':
    main()
