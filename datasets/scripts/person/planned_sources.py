"""Resolve explicit person scene inputs through this task's source roles."""
import json
from pathlib import Path

import cv2
import numpy as np

from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import WORKSPACE as ROOT
from datasets.scripts.common.source_history import RELOCATIONS, read_source_history
from datasets.scripts.common.images import read_image
from datasets.scripts.common.annotations import class_yolo_row
from datasets.scripts.common.review.placement_record import contact_status_in_view, decode_placement_record


TASK_COMPLETE = 'task_complete'


def file(value):
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'missing workspace source: {value}')
    return path


def role_paths(source, role):
    roles = source['role_records']
    values = roles[role]
    paths = values if isinstance(values, list) else [values]
    if role == 'reviewed_background' and (environment := roles.get('reviewed_background_environment')):
        return [*paths, environment]
    return paths


def configured_background_dispatch(manifest=None):
    _, source = load_task_config('person')
    environment = source['role_records'].get('reviewed_background_environment')
    selected = manifest or environment
    if selected not in role_paths(source, 'reviewed_background'):
        raise ValueError('Register a current person background dispatch in source.yaml or explicitly select a registered one')
    path = file(selected)
    if selected == environment:
        snapshot = json.loads(path.read_text(encoding='utf-8-sig'))
        rows = snapshot.get('records')
        if (snapshot.get('source_relation_basis') != 'frame_environment' or not isinstance(rows, list) or
                not all(_eligible_environment_background(row) for row in rows)):
            raise ValueError('The configured person dispatch contains unqualified background records')
    return selected, selected == environment


def _eligible_environment_background(row):
    return (row.get('status') == 'eligible' and row.get('role') == 'composite_background'
            and row.get('scope', {}).get('kind') == 'frame'
            and row.get('record_id') and row.get('frame_id') and row.get('content_sha256')
            and (row.get('geometry') or {}).get('placeable_ground'))


def verify_current_backgrounds(states, split):
    """Check only the unique frame revisions selected for a new person batch."""
    from datasets.scripts.common.frame_environment import query_frames

    _, source = load_task_config('person')
    database = file(source['databases']['frame_environment'])
    checked = set()
    for state in states:
        bg = state['background']
        frame_id, record_id = bg.get('frame_id'), bg.get('record_id')
        if not frame_id or not record_id:
            raise ValueError(f'New person batches require a frame-environment qualification: {bg["image"]}')
        key = (frame_id, record_id, bg['image'], bg.get('content_sha256'))
        if key in checked:
            continue
        current = query_frames(task='person', role='composite_background', split=split,
                               frame_id=frame_id, database=database)
        if (len(current) != 1 or current[0]['record_id'] != record_id or
                current[0]['content_sha256'] != bg.get('content_sha256') or bg['image'] not in current[0]['paths']):
            raise ValueError(f'Person background qualification is no longer current for {split}: {bg["image"]}. '
                             'Refresh the background dispatch and recompile the plan.')
        checked.add(key)


def environment_background_row(decision):
    """Adapt one retained person decision to the explicit scene inputs."""
    width, height = decision['image_size']
    boxes = decision['label']['instances']
    quad = decision['geometry']['placeable_ground']['reference_quad_px']
    return {**decision, 'source_image_size': [width, height],
            'source_group': decision.get('source_group') or decision['parent_group'],
            'source_video': decision.get('parent_video'), 'source_scene': decision.get('source_blend'),
            'state': 'complete_positive' if boxes else 'complete_negative', 'boxes': boxes,
            'ground_polygon_640x360': [[round(x * 639 / (width - 1)), round(y * 359 / (height - 1))]
                                      for x, y in quad]}


def _historical_background(source, bg, split):
    from datasets.scripts.common.frame_environment import query_frames

    manifest = bg.get('manifest')
    if manifest not in role_paths(source, bg['role']):
        raise ValueError('historical person background manifest is not registered')
    rows = query_frames(task='person', role='composite_background', split=split, frame_id=bg.get('frame_id'),
                        record_id=bg['record_id'], database=file(source['databases']['frame_environment']))
    if len(rows) != 1 or rows[0]['recorded_status'] != 'eligible' or bg['image'] not in rows[0]['paths']:
        raise ValueError(f'Historical person background qualification is unavailable: {bg["image"]}')
    decision = {**rows[0], 'status': rows[0]['recorded_status'], 'image': bg['image']}
    if not _eligible_environment_background(decision):
        raise ValueError(f'Historical person qualification lacks full-frame placement evidence: {bg["image"]}')
    return file(manifest), {'source_relation_basis': 'frame_environment'}, environment_background_row(decision)


def role_record(source, role, reference):
    """Resolve a configured record, including its registered former path."""
    manifest = reference.get('manifest')
    allowed = role_paths(source, role)
    if role == 'white_back' and manifest not in allowed:
        # Existing plans keep the exact reference they were compiled with.
        manifest = next((current for current, former in read_source_history(ROOT)
                         if former == manifest and current in allowed), manifest)
    if manifest not in allowed:
        raise ValueError(f'manifest is not configured for person role {role}')
    path = file(manifest)
    text = path.read_text(encoding='utf-8-sig')
    info = {'records': [json.loads(line) for line in text.splitlines() if line.strip()]} if path.suffix == '.jsonl' else json.loads(text)
    rows = info.get('records', info.get('images', info.get('samples')))
    if role == 'white_back' and 'foreground' in info and rows is None:
        rows = [info]
    index = reference.get('record_index')
    if not isinstance(rows, list) or type(index) is not int or not 0 <= index < len(rows):
        raise ValueError('source record index is invalid')
    return path, info, rows[index]


def complete_background(row, state_name, split):
    if row.get('split') != split or row.get('semantic_status') == 'unknown':
        return False
    raw_state = row.get('state', row.get('task_state'))
    if raw_state in ('confirmed_empty', 'complete_positive', 'complete_negative'):
        return True
    semantic = row.get('semantic_status')
    if state_name in ('complete_positive', 'complete_negative') and semantic in (
        state_name, f'auto_candidate_{state_name}',
    ):
        return True
    return (state_name == 'complete_negative' and row.get('generation_method') == 'blender_render'
            and bool(row.get('absence_evidence')))


def recorded_ground(row):
    """Load the placement observation on the selected, full source frame."""
    record = (row.get('geometry') or {}).get('placeable_ground')
    if record is None:
        if 'record_id' in row:
            raise ValueError('missing persisted person placement evidence')
        return None
    width, height = row['source_image_size']
    if record.get('image_shape') != [height, width]:
        raise ValueError('person placement evidence must use the selected source frame coordinates')
    return record, decode_placement_record(record)


def recorded_contact(assessment, actor, resolution):
    height, width = assessment.candidate_mask.shape
    output_width, output_height = resolution
    if resolution == (640, 480) and width * 9 == height * 16:
        content = (0, 60, 640, 420)
    else:
        content = (0, 0, output_width, output_height)
    return contact_status_in_view(assessment, actor['foot_xy'], contact_width=actor['contact_width'],
                                  crop_xyxy=(0, 0, width, height), content_xyxy=content)


def _resolve(declaration, check_sources=True, *, historical_replay=False):
    """Validate one declared scene and return paths and records without pixels."""
    _, source = load_task_config('person')
    if declaration.get('model_key') != 'person' or declaration.get('target_split') not in ('train', 'val', 'test'):
        raise ValueError('invalid person task or split')
    if declaration.get('constraints') not in (
        {'foreground': 'planned_actors', 'identity': 'front_visible', 'background': 'complete_existing_or_empty'},
        {'foreground': 'planned_actors', 'identity': 'front_visible', 'background': 'recorded_background_scope'},
        {'foreground': 'planned_actors', 'identity': 'front_visible', 'background': TASK_COMPLETE},
    ):
        raise ValueError('unsupported scene constraints')
    state = declaration['strata'][0]
    if state.get('min_count') not in (0, 1) or state.get('max_count') != 1:
        raise ValueError('each concrete scene may be allocated at most once')
    bg, params = state['background'], state['parameters']
    role = bg['role']
    if role not in ('annotated_real_background', 'annotated_real_reference', 'background_candidates', 'shared_background', 'reviewed_background'):
        raise ValueError('unsupported background role')
    if historical_replay and bg.get('record_id'):
        review, info, row = _historical_background(source, bg, declaration['target_split'])
    else:
        review, info, row = role_record(source, role, bg)
    if bg.get('manifest') == source['role_records'].get('reviewed_background_environment'):
        if info.get('source_relation_basis') != 'frame_environment' or not _eligible_environment_background(row):
            raise ValueError('the current person plan requires eligible frame-environment records')
    if 'record_id' in row or 'record_id' in bg:
        if (not row.get('record_id') or not row.get('frame_id') or
                not row.get('content_sha256') or
                any(row.get(key) != bg.get(key) for key in ('frame_id', 'record_id', 'content_sha256'))):
            raise ValueError('person background frame or review revision differs from the plan')
    image = file(bg['image'])
    declared_image = row.get('image', row.get('path'))
    if declared_image != bg['image'] and (review.parent / str(declared_image)).resolve() != image:
        raise ValueError('background does not match the selected record')
    state_name = row.get('state', row.get('task_state', ''))
    state_name = state_name.removeprefix('candidate_')
    if declaration['target_split'] == 'train' and row.get('split') == 'regression':
        raise ValueError('background record is reserved for regression')
    if state_name not in ('complete_positive', 'complete_negative', 'confirmed_empty', 'unknown'):
        raise ValueError('background record is not an accepted candidate')
    task_complete = declaration['constraints']['background'] == TASK_COMPLETE
    if task_complete and not complete_background(row, state_name, declaration['target_split']):
        raise ValueError(f'background must have complete person evidence for this split: {bg["image"]}')
    rows = row.get('boxes', [])
    if state_name == 'complete_positive' and not rows:
        raise ValueError('positive background requires existing person boxes')
    if state_name not in ('complete_positive', 'unknown') and rows:
        raise ValueError('empty background conflicts with existing person boxes')
    existing = [{'class_id': item['class_id'], 'xyxy': item['xyxy']} for item in rows]
    width, height = params['resolution']
    procedural, media = {review}, {image}
    for key in ('source_video', 'parent_image'):
        value = row.get(key, info.get(key))
        if value:
            media.add(file(value))
    label_path = row['label'].get('label') if 'record_id' in row else row.get('label')
    if existing and (label_path or 'record_id' not in row):
        label = (ROOT / label_path if 'record_id' in row else review.parent / label_path).resolve()
        if not label.is_relative_to(ROOT) or not label.is_file():
            raise ValueError('background label is missing')
        label_width, label_height = row['source_image_size'] if 'record_id' in row else (width, height)
        expected = [class_yolo_row(item['class_id'], item['xyxy'], label_width, label_height) for item in existing]
        actual = label.read_text(encoding='utf8').splitlines()
        if len(actual) != len(expected) or any(len(a.split()) != 5 or
            any(abs(float(x)-float(y)) > 1e-5 for x,y in zip(a.split(), b.split())) for a,b in zip(actual, expected)):
            raise ValueError('background labels do not match the selected review')
        procedural.add(label)
    actors, regions, back_sources = [], {}, {}
    for entry in state['target']['actors']:
        actor = dict(entry)
        reference = actor.pop('source_record', None)
        region = actor.pop('identity_region_xyxy', None)
        role = actor.pop('source_role')
        path = file(actor['source'])
        if role == 'person_artwork':
            if not path.is_relative_to(file_root(source['person_artwork_root'])) or actor['kind'] != 'person':
                raise ValueError('person artwork must use the configured root')
            if actor.get('view') != 'front':
                raise ValueError('frontal artwork cannot be declared a real side/back view')
            folder = {'community': '社区人员', 'noncommunity': '非社区人员'}.get(actor.get('category'))
            if not folder or path.parent.name != folder or path.stem != actor.get('identity'):
                raise ValueError('person identity/class conflicts with the artwork')
        elif role == 'white_back' and reference:
            manifest, _, record = role_record(source, role, reference)
            if (actor['kind'] != 'person' or actor.get('view') != 'back'
                    or actor.get('category') is not None or actor.get('identity') is not None
                    or record.get('status_for_model') != 'hard_negative_no_person_box'
                    or record.get('foreground') != actor['source']):
                raise ValueError('white back actor must use its configured hard-negative crop without a class or identity')
            if record.get('generation_method') == 'mirrored_white_rgba':
                media_keys = ('parent_image', 'canonical_parent_image', 'parent_artwork')
                evidence_keys = ('parent_manifest',)
                metadata_keys = ('generation_method', 'parent_sha256')
            else:
                media_keys = ('parent_image', 'parent_video')
                evidence_keys = ('parent_sequence', 'review_evidence')
                metadata_keys = ('source_time_seconds',)
            media.update(file(record[key]) for key in media_keys)
            procedural.add(manifest)
            if manifest.relative_to(ROOT).as_posix() != reference['manifest']:
                procedural.add(file(RELOCATIONS))
            procedural.update(file(record[key]) for key in evidence_keys)
            back_sources[actor['id']] = {
                **reference, **{key: record[key] for key in media_keys + evidence_keys + metadata_keys},
            }
        else:
            if role not in ('foreground_texture', 'vehicle_occluder') or not reference:
                raise ValueError('foreground requires a configured source role and record')
            manifest, _, record = role_record(source, role, reference)
            expected = record.get('cropped_image', record.get('image'))
            if expected != actor['source'] or record.get('class') != actor.get('category'):
                raise ValueError('actor source or class differs from the configured record')
            if (role == 'vehicle_occluder') != (actor['kind'] == 'vehicle'):
                raise ValueError('actor kind conflicts with its source role')
            if actor['kind'] == 'person' and actor.get('view') != 'front':
                raise ValueError('frontal texture cannot be declared a side/back view')
            procedural.add(manifest)
            for key in ('artwork', 'path', 'source_frame', 'source_video'):
                if record.get(key):
                    media.add(file(record[key]))
        media.add(path)
        if actor['kind'] == 'person' and actor['view'] == 'front':
            if not isinstance(region, list) or len(region) != 4 or any(type(v) is not int for v in region):
                raise ValueError('frontal identity needs an explicit source-pixel evidence region')
            x1, y1, x2, y2 = region
            if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                raise ValueError('identity evidence region is outside the artwork')
            regions[actor['id']] = region
        actors.append(actor)
    augmentation = state['augmentation']
    if set(augmentation) != {'value_gain', 'saturation_gain'} or any(
        isinstance(v, bool) or not isinstance(v, (int,float)) or not np.isfinite(v) or not .3 <= v <= 1.6
        for v in augmentation.values()):
        raise ValueError('invalid explicit brightness/saturation')
    if state['labels'] != {'person': 'existing_and_new_visible_yolo', 'white_back': 'not_labeled'}:
        raise ValueError('unsupported person labels')
    if check_sources and sorted(p.relative_to(ROOT).as_posix() for p in media) != state['sources']:
        raise ValueError('declared scene sources must include every actual parent and foreground')
    return {'source': source, 'state': state, 'row': row, 'bg': bg, 'params': params,
            'review': review, 'image': image, 'existing': existing, 'actors': actors,
            'regions': regions, 'media': media, 'procedural': procedural,
            'task_complete': task_complete, 'white_back_sources': back_sources}


def scene_sources(declaration):
    """Return declared sources of one scene without decoding images."""
    context = _resolve(declaration, check_sources=False)
    return (sorted(path.relative_to(ROOT).as_posix() for path in context['media']),
            sorted(path.relative_to(ROOT).as_posix() for path in context['procedural']))


def prepare_scene(declaration, load=True, working=None, *, historical_replay=False):
    """Read and check a multi-actor state without compositing or writing images."""
    context = _resolve(declaration, historical_replay=historical_replay)
    if not load:
        return {'scene': None, 'state': context['state'], 'row': context['row'],
                'image': context['image'], 'review': context['review'],
                'existing': context['existing'], 'protected': context['row'].get('white_back_reference_boxes', []),
                'media': context['media'], 'procedural': context['procedural']}
    state, params = context['state'], context['params']
    image, bg, row, existing = context['image'], context['bg'], context['row'], context['existing']
    width, height = params['resolution']
    saved_ground = recorded_ground(row) if context['actors'] else None
    if saved_ground and any(recorded_contact(saved_ground[1], actor, (width, height)) != 'candidate'
                            for actor in context['actors']):
        raise ValueError('planned contact is outside the recorded person placement area')
    effective_height = height
    background_image = read_image(working.path(image) if working else image)
    if saved_ground and background_image.shape[:2] != saved_ground[1].candidate_mask.shape:
        raise ValueError('background dimensions differ from the saved person placement evidence')
    if background_image.shape != (height, width, 3):
        source_height, source_width = background_image.shape[:2]
        if 'record_id' in row and [source_width, source_height] != row['source_image_size']:
            raise ValueError('background dimensions differ from the saved person review')
        if ('record_id' in row and (width, height) == (640, 480) and source_width * 9 == source_height * 16):
            background_image = cv2.resize(background_image, (640, 360), interpolation=cv2.INTER_AREA)
            background_image = cv2.copyMakeBorder(background_image, 60, 60, 0, 0, cv2.BORDER_CONSTANT)
            effective_height = 360
        elif 'record_id' in row and width * source_height == height * source_width:
            background_image = cv2.resize(background_image, (width, height), interpolation=cv2.INTER_AREA)
        elif (bg['role'] == 'reviewed_background' and (width, height) == (640, 480) and
                background_image.shape in ((360, 640, 3), (720, 1280, 3), (1080, 1920, 3))):
            if background_image.shape != (360, 640, 3):
                background_image = cv2.resize(background_image, (640, 360), interpolation=cv2.INTER_AREA)
            background_image = cv2.copyMakeBorder(background_image, 60, 60, 0, 0, cv2.BORDER_CONSTANT)
            effective_height = 360
        elif bg['role'] == 'reviewed_background' and background_image.shape == (height * 3, width * 3, 3):
            background_image = cv2.resize(background_image, (width, height), interpolation=cv2.INTER_AREA)
        else:
            raise ValueError('background dimensions differ from the plan')
    if existing and 'record_id' in row:
        source_width, source_height = row['source_image_size']
        scale_x, scale_y = width / source_width, effective_height / source_height
        pad = (height - effective_height) / 2
        existing = [{'class_id': item['class_id'],
                     'xyxy': [x1 * scale_x, y1 * scale_y + pad, x2 * scale_x, y2 * scale_y + pad]}
                    for item in existing for x1, y1, x2, y2 in [item['xyxy']]]
    actors, foregrounds, masks = [], {}, {}
    for actor in context['actors']:
        path = file(actor['source'])
        rgba = read_image(working.path(path) if working else path, cv2.IMREAD_UNCHANGED)
        if rgba.ndim != 3 or rgba.shape[2] != 4 or not np.any(rgba[:,:,3] > 127):
            raise ValueError('foreground requires nonempty RGBA')
        foregrounds[actor['id']] = rgba
        region = context['regions'].get(actor['id'])
        if region is not None:
            x1, y1, x2, y2 = region
            if not 0 <= x1 < x2 <= rgba.shape[1] or not 0 <= y1 < y2 <= rgba.shape[0]:
                raise ValueError('identity evidence region is outside the artwork')
            mask = np.zeros(rgba.shape[:2], np.uint8)
            mask[y1:y2,x1:x2] = np.where(rgba[y1:y2,x1:x2,3] > 127, 255, 0)
            if not mask.any():
                raise ValueError('identity evidence region contains no visible artwork')
            masks[actor['id']] = mask
        actors.append(actor)
    person_scope = 'complete_existing' if existing else 'confirmed_empty'
    if row.get('state') == 'candidate_complete_negative':
        person_scope = 'candidate_empty'
    elif row.get('state') == 'unknown':
        person_scope = 'unknown'
    scene = {'background_source': bg['image'], 'background_assessment': {
        'person_scope': person_scope, 'evidence': bg['manifest'],
        'existing_people': existing, 'avoid_boxes': row.get('white_back_reference_boxes', [])},
        'ground_polygon': params['ground_polygon'], 'obstacle_polygons': params['obstacle_polygons'],
        'actors': actors, 'required_overlaps': params['required_overlaps']}
    return {'scene': scene, 'state': state, 'row': row, 'image': image, 'background_image': background_image,
            'review': context['review'], 'existing': existing,
            'protected': row.get('white_back_reference_boxes', []),
            'media': context['media'], 'procedural': context['procedural'],
            'foregrounds': foregrounds, 'identity_masks': masks,
            'task_complete': context['task_complete'], 'effective_image_height': effective_height,
            'white_back_sources': context['white_back_sources']}


def file_root(value):
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT) or not path.is_dir():
        raise ValueError('configured root is missing or outside workspace')
    return path
