"""Resolve explicit person scene inputs through this task's source roles."""
import json
from pathlib import Path

import cv2
import numpy as np

from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import WORKSPACE as ROOT
from datasets.scripts.common.images import read_image
from datasets.scripts.common.annotations import class_yolo_row


def file(value):
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'missing workspace source: {value}')
    return path


def role_paths(source, role):
    values = source['role_records'][role]
    return values if isinstance(values, list) else [values]


def role_record(source, role, reference):
    """Only explicitly selected records are loaded; no discovery scan or fallback."""
    if reference.get('manifest') not in role_paths(source, role):
        raise ValueError(f'manifest is not configured for person role {role}')
    path = file(reference['manifest'])
    text = path.read_text(encoding='utf-8-sig')
    info = {'records': [json.loads(line) for line in text.splitlines() if line.strip()]} if path.suffix == '.jsonl' else json.loads(text)
    rows = info.get('records', info.get('images', info.get('samples')))
    index = reference.get('record_index')
    if not isinstance(rows, list) or type(index) is not int or not 0 <= index < len(rows):
        raise ValueError('source record index is invalid')
    return path, info, rows[index]


def _resolve(declaration, check_sources=True):
    """Validate one declared scene and return paths and records without pixels."""
    _, source = load_task_config('person')
    if declaration.get('model_key') != 'person' or declaration.get('target_split') not in ('train', 'val', 'test'):
        raise ValueError('invalid person task or split')
    if declaration.get('constraints') != {'foreground': 'planned_actors', 'identity': 'front_visible',
                                          'background': 'complete_existing_or_empty'}:
        raise ValueError('unsupported scene constraints')
    state = declaration['strata'][0]
    if state.get('min_count') not in (0, 1) or state.get('max_count') != 1:
        raise ValueError('each concrete scene may be allocated at most once')
    bg, params = state['background'], state['parameters']
    role = bg['role']
    if role not in ('annotated_real_background', 'annotated_real_reference', 'background_candidates', 'shared_background', 'reviewed_background'):
        raise ValueError('unsupported background role')
    review, info, row = role_record(source, role, bg)
    image = file(bg['image'])
    declared_image = row.get('image', row.get('path'))
    if declared_image != bg['image'] and (review.parent / str(declared_image)).resolve() != image:
        raise ValueError('background does not match the selected record')
    state_name = row.get('state', row.get('task_state', ''))
    state_name = state_name.removeprefix('candidate_')
    if declaration['target_split'] == 'train' and row.get('split') == 'regression':
        raise ValueError('background record is reserved for regression')
    if state_name not in ('complete_positive', 'complete_negative', 'confirmed_empty'):
        raise ValueError('background record lacks complete person labels or explicit empty state')
    rows = row.get('boxes', [])
    if state_name == 'complete_positive' and not rows:
        raise ValueError('positive background requires existing person boxes')
    if state_name != 'complete_positive' and rows:
        raise ValueError('empty background conflicts with existing person boxes')
    existing = [{'class_id': item['class_id'], 'xyxy': item['xyxy']} for item in rows]
    width, height = params['resolution']
    procedural, media = {review}, {image}
    for key in ('source_video', 'parent_image'):
        value = row.get(key, info.get(key))
        if value:
            media.add(file(value))
    if existing:
        label = (review.parent / row['label']).resolve()
        if not label.is_relative_to(ROOT) or not label.is_file():
            raise ValueError('background label is missing')
        expected = [class_yolo_row(item['class_id'], item['xyxy'], width, height) for item in existing]
        actual = label.read_text(encoding='utf8').splitlines()
        if len(actual) != len(expected) or any(len(a.split()) != 5 or
            any(abs(float(x)-float(y)) > 1e-5 for x,y in zip(a.split(), b.split())) for a,b in zip(actual, expected)):
            raise ValueError('background labels do not match the selected review')
        procedural.add(label)
    actors, regions = [], {}
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
            'regions': regions, 'media': media, 'procedural': procedural}


def scene_sources(declaration):
    """Return declared sources of one scene without decoding images."""
    context = _resolve(declaration, check_sources=False)
    return (sorted(path.relative_to(ROOT).as_posix() for path in context['media']),
            sorted(path.relative_to(ROOT).as_posix() for path in context['procedural']))


def prepare_scene(declaration, load=True):
    """Read and check a multi-actor state without compositing or writing images."""
    context = _resolve(declaration)
    if not load:
        return {'scene': None, 'state': context['state'], 'row': context['row'],
                'image': context['image'], 'review': context['review'],
                'existing': context['existing'], 'protected': context['row'].get('white_back_reference_boxes', []),
                'media': context['media'], 'procedural': context['procedural']}
    state, params = context['state'], context['params']
    image, bg, row, existing = context['image'], context['bg'], context['row'], context['existing']
    width, height = params['resolution']
    background_image = read_image(image)
    if background_image.shape != (height, width, 3):
        if bg['role'] != 'reviewed_background' or background_image.shape != (height * 3, width * 3, 3):
            raise ValueError('background dimensions differ from the plan')
        background_image = cv2.resize(background_image, (width, height), interpolation=cv2.INTER_AREA)
    actors, foregrounds, masks = [], {}, {}
    for actor in context['actors']:
        path = file(actor['source'])
        rgba = read_image(path, cv2.IMREAD_UNCHANGED)
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
    scene = {'background_source': bg['image'], 'background_assessment': {
        'person_scope': 'complete_existing' if existing else 'confirmed_empty', 'evidence': bg['manifest'],
        'existing_people': existing, 'avoid_boxes': row.get('white_back_reference_boxes', [])},
        'ground_polygon': params['ground_polygon'], 'obstacle_polygons': params['obstacle_polygons'],
        'actors': actors, 'required_overlaps': params['required_overlaps']}
    return {'scene': scene, 'state': state, 'row': row, 'image': image, 'background_image': background_image,
            'review': context['review'], 'existing': existing,
            'protected': row.get('white_back_reference_boxes', []),
            'media': context['media'], 'procedural': context['procedural'],
            'foregrounds': foregrounds, 'identity_masks': masks}


def file_root(value):
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT) or not path.is_dir():
        raise ValueError('configured root is missing or outside workspace')
    return path
