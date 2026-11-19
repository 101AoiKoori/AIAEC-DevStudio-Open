"""Compose one explicitly planned person on a reviewed scene as a candidate."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASK_HELPERS = ROOT / 'datasets/scripts/person'
sys.path.insert(0, str(ROOT))
from datasets.scripts.person.person_scene_plan import compose_scene, _polygon
from datasets.scripts.person.synthesize_real_ground import assets, read_color, write_image, yolo_row
from lab.training_common.production_guard import generation_guard, plan_production, receipt_completed
from datasets.scripts.common import recoverable_publish
from datasets.scripts.person import sample_replace
from datasets.scripts.person.prepare_v4_masked_train import source_role_inputs
from lab.training_common.camera_tone import source_role, apply_tone
from datasets.scripts.person.planned_sources import TASK_COMPLETE, prepare_scene, scene_sources, verify_current_backgrounds
from datasets.scripts.person import blender_existing
from lab.scripts.data.v4_source_gate import build_queue, queue_inputs
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir
from datasets.scripts.common.working_inputs import CODE as WORKING_INPUTS_CODE, WorkingInputs


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


def prepare(declaration, working=None, *, historical_replay=False):
    """Read source records and validate one state; never produce output."""
    if (isinstance(declaration, dict) and declaration.get('strata') and
            declaration['strata'][0].get('background', {}).get('role') == 'blender_existing'):
        if declaration.get('constraints', {}).get('background') == TASK_COMPLETE:
            raise ValueError('task-complete plans require a configured background record')
        return blender_existing.prepare_scene(declaration)
    if declaration.get("constraints", {}).get("foreground") == "planned_actors":
        return prepare_scene(declaration, working=working, historical_replay=historical_replay)
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
    if working:
        art = working.path(art)
    if read_color(working.path(image) if working else image).shape != (360, 640, 3):
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
    if (not isinstance(states, list) or not 1 <= len(states) <= 10000 or
            type(declaration.get('budget')) is not int or not 1 <= declaration['budget'] <= min(3000, len(states))):
        raise ValueError('batch budget must fit the available one-image states, up to 3000 images')
    identifiers, signatures = set(), set()
    for state in states:
        if (not isinstance(state, dict) or type(state.get('min_count')) is not int or
                type(state.get('max_count')) is not int or state['min_count'] not in (0, 1) or state['max_count'] != 1):
            raise ValueError('each explicit state must allocate at most one image')
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


def produce(plan_path, output, *, replace=False):
    plan_path = workspace_file(plan_path)
    output_path = ROOT / output
    if output_path.is_symlink():
        raise ValueError('person output must not be a symlink')
    output = output_path.resolve()
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    split = declaration['target_split']
    formal_output = split_dir('person', split).resolve()
    if output not in {sample_dir('person').resolve(), formal_output}:
        raise ValueError('output must be the person sample or declared split directory')
    task_complete = declaration.get('constraints', {}).get('background') == TASK_COMPLETE
    unified = 'direct_capture_ids' in declaration
    images, labels = output/'images', output/'labels'
    direct_media, direct_procedural, direct_code = set(), set(), ()
    run_parameters = {'version': 1}
    operation = 'person.single_ground'
    auxiliary_outputs = []
    if unified:
        from datasets.scripts.person import package_person_dataset as package
        if not task_complete:
            raise ValueError('unified export requires task-complete labels')
        selection = package.prepare_direct(split, declaration['direct_capture_ids'])
        direct_media, direct_procedural, direct_code = package.direct_dependencies(selection)
        run_parameters['direct_capture_ids'] = declaration['direct_capture_ids']
        run_parameters['frame_environment_record_ids'] = [item[1]['frame_environment']['record_id']
                                                          for item in selection if 'frame_environment' in item[1]]
        auxiliary_outputs = [images/package.DIRECT_DIR]
        images, labels = images/'synthetic', labels/'synthetic'
        operation = 'person.unified_ground'
    _batch_layout(declaration)
    states = [state for state in plan_production(declaration, ROOT)['strata'] if state['planned_count']]
    verify_current_backgrounds(states, split)
    # Source resolution for the receipt is path-only; pixels are decoded one state at a time
    # so a full train batch does not hold every background and foreground in memory at once.
    declared_media, declared_procedural = set(), set()
    verified_backgrounds = set()
    for state in states:
        one = dict(declaration, budget=1, strata=[state])
        background = state['background']
        if 'record_id' in background and background['record_id'] not in verified_backgrounds:
            from datasets.scripts.common.frame_environment import verified_image

            verified_image(background)
            verified_backgrounds.add(background['record_id'])
        if state['background'].get('role') == 'blender_existing':
            media_paths, procedural_paths = blender_existing.scene_sources(one)
        else:
            media_paths, procedural_paths = scene_sources(one)
        declared_media.update(ROOT / value for value in media_paths)
        declared_procedural.update(ROOT / value for value in procedural_paths)
    background_inputs = {ROOT / s['background']['image'] for s in states}
    real_backgrounds = {ROOT / s['background']['image'] for s in states
                        if s['background'].get('role') != 'blender_existing'}
    working = WorkingInputs('person', backgrounds=real_backgrounds,
                            targets=declared_media - background_inputs, split=declaration['target_split'])
    # Queue construction and the production receipt are only invoked at actual production time.
    queue_dependencies = queue_inputs(ROOT)
    queue = build_queue(ROOT, inputs=queue_dependencies)
    media, procedural, code = source_role_inputs(ROOT, known=queue, queue_dependencies=queue_dependencies)
    unavailable = {row['path'] for row in queue.get('unavailable_exposure_paths', [])}
    # source_role itself skips these lifecycle-excluded historical images.
    media = {path for path in media if path.relative_to(ROOT).as_posix() not in unavailable}
    media |= {path for path in declared_media if path.is_file()} | working.media | direct_media
    procedural = set(procedural) | declared_procedural | working.evidence | direct_procedural | {
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
        # Keep prior regression exposure in each output record; source roles are
        # provenance, not an automatic block on the user's training plan.
    code_paths = {Path(__file__), Path(sample_replace.__file__), Path(recoverable_publish.__file__),
                                      TASK_HELPERS/'person_scene_plan.py', TASK_HELPERS/'planned_sources.py',
                                      TASK_HELPERS/'blender_existing.py',
                                      TASK_HELPERS/'synthesize_real_ground.py',
                                      ROOT/'datasets/scripts/person/prepare_v4_masked_train.py',
                                      ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py',
                                      ROOT/'datasets/scripts/common/images.py', ROOT/'datasets/scripts/common/perspective.py', ROOT/'datasets/scripts/common/composite.py', ROOT/'datasets/scripts/common/annotations.py',
                                      ROOT/'datasets/scripts/common/review/placement_record.py',
                                      ROOT/'datasets/scripts/common/review/placement.py',
                                      ROOT/'lab/training_common/camera_tone.py', WORKING_INPUTS_CODE, *code, *direct_code}
    if verified_backgrounds:
        code_paths.add(ROOT/'datasets/scripts/common/frame_environment.py')
    run_parameters['background_frame_environment_record_ids'] = sorted(verified_backgrounds)
    operation = f'{operation}:{output.relative_to(ROOT).as_posix()}'
    committed = lambda token: receipt_completed(ROOT, operation, token)
    with recoverable_publish.sample_lock(output):
        recoverable_publish.recover(output, committed)
        owned = set()

        def verify_existing(workspace, output_state, previous):
            owned.update(sample_replace.owned_files(output, previous, output_state, workspace))
            root = output.relative_to(workspace).as_posix()
            retained = [row['path'] for row in output_state if row.get('type') == 'file'
                        and row['path'].startswith(root + '/')
                        and row['path'][len(root) + 1:] not in owned]
            return {'preserved_paths': retained}

        stage = None
        try:
            with generation_guard(operation, [*media, *procedural], [output], run_parameters,
                                  code_paths=code_paths, procedural_inputs=procedural, root=ROOT,
                                  production_plan=declaration, sample_output_paths=[images],
                                  auxiliary_output_paths=auxiliary_outputs, replace_outputs=replace,
                                  replace_verifier=verify_existing if replace else None) as run:
                verify_current_backgrounds([state for state in run.plan['strata'] if state['planned_count']], split)
                if run.reused:
                    return json.loads((output/'manifest.json').read_text(encoding='utf8'))
                if not replace and output.exists() and any(path.name != '.gitkeep' for path in output.iterdir()):
                    raise ValueError('output is not empty')
                stage = recoverable_publish.prepare(output, run.token)
                recoverable_publish.copy_unowned(output, stage, owned)
                staged_images = stage / images.relative_to(output)
                staged_labels = stage / labels.relative_to(output)
                staged_images.mkdir(parents=True, exist_ok=True)
                staged_labels.mkdir(parents=True, exist_ok=True)
                records, assignments = [], []
                for state in run.plan['strata']:
                    if state['planned_count'] == 0:
                        continue
                    if state['planned_count'] != 1:
                        raise ValueError('each explicit state must allocate at most one image')
                    number = len(records) + 1
                    if ((staged_images / f'person_{number:06d}.png').exists() or
                            (staged_labels / f'person_{number:06d}.txt').exists()):
                        raise ValueError('generated person sample would overwrite an unowned file')
                    item = prepare(dict(declaration, budget=1, strata=[state]), working=working)
                    role, reason = roles[item['image']]
                    actual = _produce_one(stage, number, state, item, role, reason, working=working,
                                          image_dir=staged_images, label_dir=staged_labels)
                    for key in ('image', 'label'):
                        actual[key] = (output / (ROOT / actual[key]).relative_to(stage)).relative_to(ROOT).as_posix()
                    if state['background'].get('role') != 'blender_existing':
                        actual['working_background'] = working.actual(item['image'])
                        actual['working_targets'] = {value: working.actual(value) for value in state['sources']
                                                     if value in working.aliases and value != state['background']['image']}
                    records.append(actual)
                    assignments.append((state, actual))
                    del item
                if unified:
                    for index, item in enumerate(selection, 1):
                        suffix = item[3].suffix.lower()
                        if ((stage / 'images' / package.DIRECT_DIR / f'person_real_{index:06d}{suffix}').exists() or
                                (stage / 'labels' / package.DIRECT_DIR / f'person_real_{index:06d}.txt').exists()):
                            raise ValueError('direct person sample would overwrite an unowned file')
                    direct_records = package.write_direct_records(stage, split, selection)
                    for record in direct_records:
                        for key in ('image', 'label'):
                            record[key] = (output / (ROOT / record[key]).relative_to(stage)).relative_to(ROOT).as_posix()
                    records.extend(direct_records)
                manifest = {'status': 'complete_task_yolo' if task_complete else 'candidate_not_semantically_verified',
                            'records': records}
                if unified:
                    manifest.update({'split': split, 'coverage': package.split_coverage(records)})
                if (stage/'manifest.json').exists():
                    raise ValueError('person manifest would overwrite an unowned file')
                (stage/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
                                                   encoding='utf8', newline='\n')
                recoverable_publish.publish(output, stage, run.token)
                for state, actual in assignments:
                    run.record_sample(state['id'], output=ROOT/actual['image'], sources=state['sources'],
                                      parameters=state['parameters'], actual=actual)
                if unified:
                    run.set_result({'manifest': (output/'manifest.json').relative_to(ROOT).as_posix(),
                                    'coverage': manifest['coverage']})
            recoverable_publish.recover(output, committed)
        except BaseException:
            if recoverable_publish._metadata(output)[1].exists():
                recoverable_publish.recover(output, committed)
            elif stage is not None and stage.exists():
                shutil.rmtree(stage)
            raise
    return manifest


def _produce_one(output, number, state, prepared, role, reason, working=None, image_dir=None, label_dir=None):
    """Execute one validated state within the active production guard."""
    param, target = state['parameters'], state['target']
    if state['background'].get('role') == 'blender_existing':
        result = blender_existing.produce_scene(prepared, state)
        return _write_result(output, number, state, prepared, role, reason, result, image_dir, label_dir)
    if 'scene' in prepared:
        result = compose_scene(prepared['background_image'], prepared['scene'], prepared['foregrounds'],
                               prepared['identity_masks'])
        result['image'] = apply_scene_tone(result['image'], **state['augmentation'])
        return _write_result(output, number, state, prepared, role, reason, result, image_dir, label_dir)
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
    image = working.path(prepared['image']) if working else prepared['image']
    result = compose_scene(read_color(image), scene, {'added': rgba},
                           {'added': np.where(rgba[:, :, 3] > 127, 255, 0).astype(np.uint8)})
    return _write_result(output, number, state, prepared, role, reason, result, image_dir, label_dir)


def _write_result(output, number, state, prepared, role, reason, result, image_dir=None, label_dir=None):
    image = (image_dir or output/'images')/f'person_{number:06d}.png'
    label = (label_dir or output/'labels')/f'person_{number:06d}.txt'
    write_image(image, result['image'])
    body = '\n'.join(result['labels'])
    label.write_text(body + ('\n' if body else ''), encoding='utf8', newline='\n')
    actual = {'image': image.relative_to(ROOT).as_posix(), 'state_id': state['id'],
              'failure_event': state.get('failure_event'),
              'background': state['background'], 'source_role': role, 'source_role_reason': reason,
              'source_time_seconds': prepared['row'].get('timestamp_seconds'),
              'background_review_state': prepared['row'].get('state', 'unknown'),
              'background_human_accepted': prepared['row'].get('human_accepted'),
              'background_people': prepared['existing'], 'background_white_backs': prepared['protected'],
              'objects': result['objects'], 'label': label.relative_to(ROOT).as_posix(),
              'label_status': ('programmatic_blender_render' if state['background'].get('role') == 'blender_existing'
                               else 'background_unknown_and_unreviewed_composite_candidate'
                               if prepared['row'].get('state') == 'unknown'
                               else 'ai_reviewed_background_and_unreviewed_composite_candidate'),
              'augmentation': state['augmentation'], 'parameters': state['parameters']}
    if prepared.get('task_complete'):
        row = prepared['row']
        source_group = row.get('source_group') or row.get('source_video') or row.get('source_scene')
        actual.update({
            'label_status': 'complete_task_yolo',
            'generation_method': 'blender_background_composite'
            if row.get('generation_method') == 'blender_render' else 'real_background_composite',
            'background_evidence': prepared['review'].relative_to(ROOT).as_posix(),
            'source_group': source_group,
            'parent_video': row.get('source_video'),
            'mother_scene': row.get('mother_scene') or row.get('source_scene'),
            'source_assets': state['sources'],
            'effective_image_height': prepared['effective_image_height'],
        })
    if prepared.get('white_back_sources'):
        actual['white_back_sources'] = prepared['white_back_sources']
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replace', action='store_true',
                        help='replace generated images and labels in the selected person output')
    args = parser.parse_args()
    print(json.dumps(produce(args.production_plan, args.output, replace=args.replace), ensure_ascii=False))


if __name__ == '__main__':
    main()
