"""Compose one explicitly planned person on a reviewed scene as a candidate."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASK_HELPERS = ROOT / 'datasets/scripts/person'
sys.path.insert(0, str(ROOT))
from datasets.scripts.person.person_scene_plan import compose_scene, _polygon
from datasets.scripts.person.synthesize_real_ground import assets, read_color, write_image, yolo_row
from lab.training_common.production_guard import generation_guard
from datasets.scripts.person.prepare_v4_masked_train import source_role_inputs
from lab.training_common.camera_tone import source_role, apply_tone
from datasets.scripts.person.planned_sources import prepare_scene, scene_sources
from lab.scripts.data.v4_source_gate import build_queue
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir


def apply_scene_tone(image, *, value_gain, saturation_gain):
    """Apply the planned person tone, including the bounded bright variant."""
    if value_gain <= 1 and saturation_gain <= 1:
        return apply_tone(image, value_gain=value_gain, saturation_gain=saturation_gain)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.rint(hsv[:, :, 1].astype(np.float32) * saturation_gain).clip(0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.rint(hsv[:, :, 2].astype(np.float32) * value_gain).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def workspace_file(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('expected a workspace-relative input file')
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.relative_to(ROOT).as_posix() != value:
        raise ValueError('input is missing or is not a canonical workspace path')
    return path


def configured_artwork_root():
    """Return the task-configured person artwork directory inside the workspace."""
    _, source_config = load_task_config('person')
    value = source_config.get('person_artwork_root')
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('person_artwork_root must be workspace-relative')
    root = (ROOT / value).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('person_artwork_root must be inside the workspace')
    return root


def prepare(declaration):
    """Read source records and validate one state; never produce output."""
    if declaration.get("constraints", {}).get("foreground") == "planned_actors":
        return prepare_scene(declaration)
    if (not isinstance(declaration, dict) or declaration.get('model_key') != 'person' or
            declaration.get('target_split') not in ('train', 'val', 'test') or declaration.get('budget') != 1 or
            declaration.get('constraints') != {'scope': 'near_frontal', 'background': 'complete_existing_candidate',
                                               'foreground': 'single_unoccluded'}):
        raise ValueError('expected one single-person training candidate state')
    states = declaration.get('strata')
    if not isinstance(states, list) or len(states) != 1:
        raise ValueError('exactly one state is supported')
    state = states[0]
    fields = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight', 'failure_cost',
              'conditions', 'background', 'target', 'labels', 'augmentation', 'parameters', 'sources'}
    if (not isinstance(state, dict) or set(state) != fields or state['conditions'] != [] or
            state['min_count'] != 1 or state['max_count'] != 1):
        raise ValueError('unsupported state fields or quota')
    bg, target, params = state['background'], state['target'], state['parameters']
    if (not isinstance(bg, dict) or set(bg) != {'image', 'review_manifest', 'record_index', 'source_video'} or
            not isinstance(target, dict) or set(target) != {'artwork', 'category', 'identity'} or
            not isinstance(params, dict) or set(params) != {'resolution', 'ground_polygon', 'foot_xy',
                                                            'contact_width', 'size_wh', 'angles_degrees',
                                                            'geometry_basis'}):
        raise ValueError('unsupported source or geometry fields')
    if (state['labels'] != {'person': 'existing_and_new_visible_yolo', 'white_back': 'not_labeled'} or
            state['augmentation'] != {'kind': 'none'} or params['resolution'] != [640, 360]):
        raise ValueError('unsupported image, label or augmentation contract')
    image, review, video, art = (workspace_file(bg['image']), workspace_file(bg['review_manifest']),
                                 workspace_file(bg['source_video']), workspace_file(target['artwork']))
    expected_sources = sorted({bg['image'], bg['source_video'], target['artwork']})
    if state['sources'] != expected_sources:
        raise ValueError('declared sources differ from actual source files')
    classes = {'community': 0, 'noncommunity': 1}
    category = target['category']
    if category not in classes or not isinstance(target['identity'], str) or not target['identity']:
        raise ValueError('a known person class and identity are required')
    folder = '社区人员' if category == 'community' else '非社区人员'
    if art.parent != configured_artwork_root() / folder or art.stem != target['identity']:
        raise ValueError('artwork and requested identity/class do not match')
    info = json.loads(review.read_text(encoding='utf8'))
    index = bg['record_index']
    if (type(index) is not int or index < 0 or not isinstance(info.get('images'), list) or
            index >= len(info['images']) or info.get('source_video') != bg['source_video']):
        raise ValueError('invalid source review index or parent video')
    row = info['images'][index]
    if (row.get('image') != bg['image'] or row.get('state') != 'complete_positive' or
            row.get('split') != 'unassigned' or not isinstance(row.get('boxes'), list) or
            not isinstance(row.get('white_back_reference_boxes'), list)):
        raise ValueError('background is not the declared complete positive candidate')
    timestamp_source = row.get('timestamp_source')
    if not isinstance(timestamp_source, str) or not timestamp_source.endswith(' showinfo'):
        raise ValueError('background timestamp evidence is missing')
    timestamp_log = workspace_file(timestamp_source[:-len(' showinfo')])
    if not isinstance(row.get('label'), str):
        raise ValueError('background has no existing label')
    label = (review.parent / row['label']).resolve()
    if not label.is_relative_to(review.parent.resolve()) or not label.is_file():
        raise ValueError('background label is missing or outside the review directory')
    boxes = [{'class_id': obj['class_id'], 'xyxy': obj['xyxy']} for obj in row['boxes']]
    expected = [yolo_row(obj['class_id'], obj['xyxy'], 640, 360) for obj in boxes]
    actual = label.read_text(encoding='utf8').splitlines()
    if len(expected) != len(actual):
        raise ValueError('background person label count differs from review')
    for a, b in zip(expected, actual):
        if len(a.split()) != 5 or len(b.split()) != 5 or any(abs(float(x)-float(y)) > 1e-5 for x, y in zip(a.split(), b.split())):
            raise ValueError('background person labels differ from reviewed boxes')
    width, height = params['resolution']
    if not isinstance(params['geometry_basis'], str) or not params['geometry_basis'].strip():
        raise ValueError('placement geometry requires an evidence basis')
    poly = _polygon(params['ground_polygon'], width, height, 'ground polygon')
    foot, size, contact = params['foot_xy'], params['size_wh'], params['contact_width']
    if (not isinstance(foot, list) or len(foot) != 2 or any(type(v) is not int for v in foot) or
            not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 2 for v in size) or
            type(contact) is not int or not 1 <= contact <= size[0] or
            params['angles_degrees'] != {'yaw': 0, 'pitch': 0, 'roll': 0}):
        raise ValueError('unsupported placement geometry')
    x, y = foot
    if any(cv2.pointPolygonTest(poly, (float(px), float(y)), False) < 0 for px in (x-contact/2, x+contact/2)):
        raise ValueError('foreground contact falls outside ground')
    left, top = round(x-size[0]/2), y-size[1]
    bounds = [left, top, left+size[0], y]
    if not 0 <= bounds[0] < bounds[2] <= width or not 0 <= bounds[1] < bounds[3] <= height:
        raise ValueError('foreground lies outside image')
    protected = [obj['xyxy'] for obj in boxes] + row['white_back_reference_boxes']
    if any(min(bounds[2], b[2]) > max(bounds[0], b[0]) and min(bounds[3], b[3]) > max(bounds[1], b[1])
           for b in protected):
        raise ValueError('planned foreground overlaps a protected background object')
    if read_color(image).shape != (360, 640, 3):
        raise ValueError('unexpected background resolution')
    foreground = cv2.imdecode(np.fromfile(str(art), np.uint8), cv2.IMREAD_UNCHANGED)
    if (foreground is None or foreground.dtype != np.uint8 or foreground.ndim != 3 or
            foreground.shape[2] != 4 or not np.any(foreground[:, :, 3] > 127)):
        raise ValueError('person artwork requires a nonempty alpha channel')
    return {'state': state, 'row': row, 'image': image, 'review': review, 'video': video,
            'artwork': art, 'label': label, 'timestamp_log': timestamp_log,
            'existing': boxes, 'protected': row['white_back_reference_boxes']}


def prepare_batch(declaration):
    """Validate explicit, unique one-image states without generating images."""
    states = declaration.get('strata') if isinstance(declaration, dict) else None
    if (not isinstance(states, list) or not 1 <= len(states) <= 3000 or
            type(declaration.get('budget')) is not int or not 1 <= declaration['budget'] <= len(states)):
        raise ValueError('batch requires one image per state, up to 3000 states')
    prepared, signatures = {}, set()
    for state in states:
        one = dict(declaration, budget=1, strata=[state])
        item = prepare(one)
        state_id = state['id']
        signature = json.dumps({key: state.get(key) for key in ('background', 'target', 'parameters', 'augmentation')},
                               ensure_ascii=False, sort_keys=True, allow_nan=False)
        if state_id in prepared or signature in signatures:
            raise ValueError('batch states must have unique IDs and image layouts')
        signatures.add(signature)
        prepared[state_id] = item
    return prepared


def _batch_layout(declaration):
    """Check the batch shape without decoding any image."""
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'person':
        raise ValueError('expected a person production plan')
    states = declaration.get('strata')
    if (not isinstance(states, list) or not 1 <= len(states) <= 3000 or
            type(declaration.get('budget')) is not int or declaration['budget'] != len(states)):
        raise ValueError('batch requires one image per state, up to 3000 states')
    identifiers, signatures = set(), set()
    for state in states:
        if (not isinstance(state, dict) or type(state.get('min_count')) is not int or
                type(state.get('max_count')) is not int or state['min_count'] != 1 or state['max_count'] != 1):
            raise ValueError('each explicit state must allocate exactly one image')
        state_id = state.get('id')
        if not isinstance(state_id, str) or not state_id or state_id in identifiers:
            raise ValueError('batch states must have unique IDs')
        identifiers.add(state_id)
        signature = json.dumps({key: state.get(key) for key in ('background', 'target', 'parameters', 'augmentation')},
                               ensure_ascii=False, sort_keys=True, allow_nan=False)
        if signature in signatures:
            raise ValueError('batch states must have unique IDs and image layouts')
        signatures.add(signature)
    return states


def produce(plan_path, output):
    plan_path = workspace_file(plan_path)
    output = (ROOT / output).resolve()
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    if output not in {sample_dir('person').resolve(), split_dir('person', declaration['target_split']).resolve()}:
        raise ValueError('output must be the person sample or declared split directory')
    states = _batch_layout(declaration)
    # Source resolution for the receipt is path-only; pixels are decoded one state at a time
    # so a full train batch does not hold every background and foreground in memory at once.
    declared_media, declared_procedural = set(), set()
    for state in states:
        media_paths, procedural_paths = scene_sources(dict(declaration, budget=1, strata=[state]))
        declared_media.update(ROOT / value for value in media_paths)
        declared_procedural.update(ROOT / value for value in procedural_paths)
    # Queue construction and the production receipt are only invoked at actual production time.
    media, procedural, code = source_role_inputs(ROOT)
    queue = build_queue(ROOT)
    unavailable = {row['path'] for row in queue.get('unavailable_exposure_paths', [])}
    # source_role itself skips these lifecycle-excluded historical images.
    media = {path for path in media if path.relative_to(ROOT).as_posix() not in unavailable}
    media |= {path for path in declared_media if path.is_file()}
    procedural = set(procedural) | declared_procedural | {
        plan_path, ROOT/'datasets/scripts/person/config/dataset.yaml',
        ROOT/'datasets/scripts/person/config/source.yaml'}
    # Queue dependencies may also be selected media; declare each path once.
    procedural.difference_update(media)
    roles = {}
    for state in states:
        image = ROOT / state['background']['image']
        if image in roles:
            continue
        roles[image] = source_role(ROOT, image, queue)
        if declaration['target_split'] == 'train' and roles[image][0] == 'regression':
            raise ValueError('background is reserved for regression: ' + roles[image][1])
    with generation_guard('person.single_ground:' + output.relative_to(ROOT).as_posix(),
                          [*media, *procedural], [output], {'version': 1},
                          code_paths=[Path(__file__), TASK_HELPERS/'person_scene_plan.py', TASK_HELPERS/'planned_sources.py',
                                      TASK_HELPERS/'synthesize_real_ground.py',
                                      ROOT/'datasets/scripts/person/prepare_v4_masked_train.py',
                                      ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py',
                                      ROOT/'datasets/scripts/common/images.py', ROOT/'datasets/scripts/common/perspective.py', ROOT/'datasets/scripts/common/composite.py', ROOT/'datasets/scripts/common/annotations.py',
                                      ROOT/'lab/training_common/camera_tone.py', *code],
                          procedural_inputs=procedural, root=ROOT, production_plan=declaration,
                          sample_output_paths=[output/'images']) as run:
        if run.reused:
            return json.loads((output/'manifest.json').read_text(encoding='utf8'))
        if output.exists() and any(output.iterdir()):
            raise ValueError('output is not empty')
        (output/'images').mkdir(parents=True)
        (output/'labels').mkdir()
        records = []
        for number, state in enumerate(run.plan['strata'], 1):
            if state['planned_count'] == 0:
                continue
            if state['planned_count'] != 1:
                raise ValueError('each explicit state must allocate at most one image')
            item = prepare(dict(declaration, budget=1, strata=[state]))
            role, reason = roles[item['image']]
            actual = _produce_one(output, number, state, item, role, reason)
            run.record_sample(state['id'], output=ROOT/actual['image'], sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            records.append(actual)
            del item
        manifest = {'status': 'candidate_not_semantically_verified', 'records': records}
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
                                            encoding='utf8', newline='\n')
        return manifest


def _produce_one(output, number, state, prepared, role, reason):
    """Execute one validated state within the active production guard."""
    param, target = state['parameters'], state['target']
    if 'scene' in prepared:
        result = compose_scene(prepared['background_image'], prepared['scene'], prepared['foregrounds'],
                               prepared['identity_masks'])
        result['image'] = apply_scene_tone(result['image'], **state['augmentation'])
        return _write_result(output, number, state, prepared, role, reason, result)
    rgba = assets.load_rgba(prepared['artwork'])
    actor = {'id': 'added', 'kind': 'person', 'source': target['artwork'], 'view': 'front',
             'category': target['category'], 'identity': target['identity'],
             'identity_evidence': target['artwork'], 'min_identity_visible': 1.0,
             'foot_xy': param['foot_xy'], 'contact_width': param['contact_width'],
             'size_wh': param['size_wh'], 'angles_degrees': param['angles_degrees']}
    scene = {'background_source': state['background']['image'],
             'background_assessment': {'person_scope': 'complete_existing',
                'evidence': state['background']['review_manifest'], 'existing_people': prepared['existing'],
                'avoid_boxes': prepared['protected']},
             'ground_polygon': param['ground_polygon'], 'obstacle_polygons': [],
             'actors': [actor], 'required_overlaps': []}
    result = compose_scene(read_color(prepared['image']), scene, {'added': rgba},
                           {'added': np.where(rgba[:, :, 3] > 127, 255, 0).astype(np.uint8)})
    return _write_result(output, number, state, prepared, role, reason, result)


def _write_result(output, number, state, prepared, role, reason, result):
    image = output/'images'/f'person_{number:06d}.png'
    label = output/'labels'/f'person_{number:06d}.txt'
    write_image(image, result['image'])
    body = '\n'.join(result['labels'])
    label.write_text(body + ('\n' if body else ''), encoding='utf8', newline='\n')
    actual = {'image': image.relative_to(ROOT).as_posix(), 'state_id': state['id'],
              'failure_event': state.get('failure_event'),
              'background': state['background'], 'source_role': role, 'source_role_reason': reason,
              'source_time_seconds': prepared['row'].get('timestamp_seconds'),
              'background_human_accepted': prepared['row'].get('human_accepted'),
              'background_people': prepared['existing'], 'background_white_backs': prepared['protected'],
              'objects': result['objects'], 'label': label.relative_to(ROOT).as_posix(),
              'label_status': 'ai_reviewed_background_and_unreviewed_composite_candidate'}
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(produce(args.production_plan, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
