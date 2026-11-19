"""Materialize shared backgrounds and carry forward task-specific review evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import filecmp
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sqlite3
from tempfile import TemporaryDirectory

from .config import load_task_config
from .paths import TASKS, WORKSPACE
from .source_history import historical_source, read_source_history


POOL_MANIFEST = 'datasets/annotated/common/sources/background_pool_manifest.json'
MACHINE_REVIEW_INDEX = 'datasets/annotated/common/sources/background_pool_machine_review_index.json'
POOL_ROOT = 'datasets/raw/shared/backgrounds'
LEGACY_POOL = 'datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113/manifest.json'
PERSON_DISPATCH = 'datasets/scripts/person/config/background_dispatch.json'
CAR_CATALOG = 'datasets/annotated/car_real_capture_20261112/direct_real_manifest.json'
CAR_SURFACES = 'datasets/annotated/car_real_capture_20261112/waypoint_surface_manifest.jsonl'
SOURCE_COPIES = 'datasets/source_copy_manifest.jsonl'
BLENDER_REGISTRY = 'datasets/blender_source_registry.sqlite3'
RENDER_BATCHES = (
    'datasets/annotated/common/sources/aiaec_eight_routes_20261117/manifest.json',
    'datasets/annotated/common/sources/aiaec_eight_routes_indoor_20261117/manifest.json',
    'datasets/annotated/common/sources/aiaec_eight_routes_detailed_20261117/manifest.json',
    'datasets/annotated/common/sources/aiaec_fire_c_facade_20261117/manifest.json',
    'datasets/annotated/waste_bin/waste_sticker_arena_surfaces_20261117/manifest.json',
)
PILOT_TASKS = frozenset({'person', 'e_bike', 'fire', 'building_name_paper'})
GROUND_TARGETS = {'person': {'height_fraction': 0.30, 'aspect': 0.42},
                  'e_bike': {'height_fraction': 0.18, 'aspect': 1.5},
                  'sign': {'height_fraction': 0.20, 'aspect': 0.50},
                  'waste_bin_state': {'height_fraction': 0.30, 'aspect': 0.75}}


def _file(path):
    relative = Path(path)
    resolved = (WORKSPACE / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(WORKSPACE) or not resolved.is_file():
        raise ValueError(f'Expected an existing workspace file: {path}')
    return resolved


def _json(path):
    return json.loads(_file(path).read_text(encoding='utf-8-sig'))


def _jsonl(path):
    with _file(path).open(encoding='utf-8-sig') as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def _source(task):
    return load_task_config(task)[1]


def _evidence(manifest, index, **fields):
    return {'manifest': manifest, 'record_index': index, **fields}


def _review(evidence, *, geometry, label_state, split=None, state='candidate',
            uses=('composite_background',), existing_targets=None, **fields):
    return {'state': state, 'uses': list(uses), 'split': split, 'label_state': label_state,
            'geometry': geometry, 'existing_targets': existing_targets, 'evidence': evidence, **fields}


class Pool:
    def __init__(self):
        self.by_image = {}
        self.source_history = read_source_history(WORKSPACE)
        self.source_copies = {row['path']: row['original_path'] for row in _jsonl(SOURCE_COPIES)
                              if row.get('path') and row.get('original_path')}
        with sqlite3.connect(_file(BLENDER_REGISTRY).as_uri() + '?mode=ro', uri=True) as db:
            self.scene_assets = {path: {'asset_id': asset_id, 'classification': classification,
                                        'lifecycle_state': state}
                                 for path, asset_id, classification, state in db.execute(
                                     'SELECT current_path, asset_id, classification, lifecycle_state FROM assets')}

    def add(self, image, evidence, *, source_group=None, video=None, scene=None, mother_scene=None,
            generation_method=None, capture_identity=None, image_size=None, timestamp_seconds=None, **fields):
        src = _file(image)
        if image_size is None:
            from PIL import Image

            with Image.open(src) as pixels:
                image_size = list(pixels.size)
        if image in self.by_image:
            row = self.by_image[image]
            if row['image_size'] != list(image_size):
                raise ValueError(f'Conflicting dimensions for shared background: {image}')
            row['source_records'].append(evidence)
            return row
        parent = mother_scene or scene or video or source_group or image
        parent = self.source_copies.get(parent, parent)
        parent = historical_source(parent, self.source_history)
        group = historical_source(source_group, self.source_history) if source_group else parent
        if scene:
            _file(scene)
            if mother_scene:
                _file(mother_scene)
            fields['blender_registry'] = {'database': BLENDER_REGISTRY, 'source_scene': self.scene_assets.get(scene),
                                          'mother_scene': self.scene_assets.get(mother_scene)}
        historical_image = historical_source(image, self.source_history)
        identity = capture_identity or historical_image
        kind = 'rendered' if scene else 'camera'
        group_stem = re.sub(r'[^\w.-]+', '_', Path(parent).stem)[:60].strip('._') or 'parent'
        group_folder = group_stem + '__' + sha256(parent.encode('utf-8')).hexdigest()[:10]
        alias = f'{POOL_ROOT}/{kind}/{group_folder}/{src.stem}__{sha256(historical_image.encode("utf-8")).hexdigest()[:10]}{src.suffix.lower()}'
        row = {'id': identity, 'capture_identity': identity, 'image': alias, 'source_image': image,
               'image_size': list(image_size), 'source_group': group, 'parent_group': parent,
               'parent_video': video, 'source_blend': scene, 'mother_scene': mother_scene,
               'source_kind': ('blender_render' if scene else
                               'video_frame' if video and generation_method != 'camera_capture' else 'camera_photo'),
               'generation_method': generation_method or ('provided_video_frame' if video else 'camera_capture'),
               'timestamp_seconds': timestamp_seconds, 'evidence': evidence, 'source_records': [evidence],
               'tasks': {}, **fields}
        self.by_image[image] = row
        return row

    def review(self, image, task, observation, **media):
        row = self.add(image, observation['evidence'], **media)
        old = row['tasks'].get(task)
        if old and old != observation:
            old.setdefault('additional_evidence', []).append(observation)
        else:
            row['tasks'][task] = observation


def _legacy_pool(pool):
    for i, row in enumerate(_json(LEGACY_POOL)['records']):
        pool.add(row['image'], _evidence(LEGACY_POOL, i), source_group=row['source_group'],
                 video=row['source_video'], image_size=row['image_size'], timestamp_seconds=row['source_time_seconds'],
                 generation_method=row['generation_method'], source_pts=row.get('source_pts'),
                 source_time_base=row.get('source_time_base'), strata=row.get('strata'),
                 former_review_status=row.get('review_status'), image_processing=row.get('image_processing'))


def _waypoints(pool, pilot):
    rows = {row['image']: row for row in _json(CAR_CATALOG)['records'] if row.get('waypoint_label')
            and '/frames_' not in row['image']}
    if pilot:
        rows = {path: row for path, row in rows.items() if row['point'] in {'P02', 'P09'}}
    for i, source in enumerate(_jsonl(CAR_SURFACES)):
        media = rows.get(source['path'])
        if media is None:
            continue
        for task in sorted(TASKS):
            if pilot and task not in PILOT_TASKS:
                continue
            target = 'fire_sticker' if task == 'fire' else task
            regions = [region for region in source['regions'] if target in region['task_candidates']]
            if not regions:
                continue
            labels = media['tasks'].get(task) or {}
            pool.review(source['path'], task, _review(_evidence(CAR_SURFACES, i, capture_manifest=CAR_CATALOG),
                        geometry={'regions': regions}, label_state=labels.get('state', 'unknown'),
                        state='partial', uses=('geometry_reference',), existing_targets=labels.get('instances')),
                        source_group=media['source_group'], video=media.get('parent_video'),
                        capture_identity=media.get('capture_identity'), image_size=media['image_size'],
                        timestamp_seconds=media.get('timestamp_seconds'), frame_index=media.get('frame_index'),
                        point=media.get('point'), waypoint_label=media.get('waypoint_label'),
                        generation_method='camera_capture')


def _person(pool):
    for i, row in enumerate(_json(PERSON_DISPATCH)['records']):
        polygon, foot = row.get('ground_polygon_640x360'), row.get('foot_xy_640x360')
        if not polygon or not foot or row.get('state') == 'excluded':
            continue
        pool.review(row['image'], 'person', _review(_evidence(PERSON_DISPATCH, i, detail=row.get('review_evidence')),
                    geometry={'ground_polygon': polygon, 'foot_xy': foot, 'coordinate_size': [640, 360]},
                    label_state=row.get('state', 'unknown'), split=row.get('split'),
                    state='ready' if row.get('review_verdict') == 'ready' else 'candidate',
                    existing_targets=row.get('boxes'), accepted_by=row.get('accepted_by'),
                    review_verdict=row.get('review_verdict'), historical_role=row.get('source_role')),
                    source_group=row.get('source_group'), video=row.get('source_video'),
                    scene=row.get('source_scene'), mother_scene=row.get('mother_scene'),
                    generation_method=row.get('generation_method'), timestamp_seconds=row.get('source_time_seconds'),
                    camera=row.get('camera'), render_frame=row.get('frame'))


def _e_bike(pool):
    from datasets.scripts.e_bike.compose_v4_grounded_train import ZONES

    config = _source('e_bike')
    manifest = config['ground_background_manifest']
    plan_path = config['production_plan']
    plan = _json(plan_path)
    selected = {row['background']['image'] for row in plan['strata']}
    for i, row in enumerate(_json(manifest)['backgrounds']):
        zone = ZONES.get(row['id'])
        if not zone or row['image'] not in selected:
            continue
        if row.get('task_state') != 'candidate_complete_negative':
            continue
        pool.review(row['image'], 'e_bike', _review(_evidence(manifest, i, plan=plan_path),
                    geometry={'ground_polygon': zone['poly'], 'foot_x_range': zone['x'],
                              'foot_y_range': zone['y'], 'coordinate_size': [640, 480]},
                    label_state=row['task_state'], split=plan['target_split'], existing_targets=[]),
                    video=row['source_video'], parent_image=row.get('parent_image'),
                    timestamp_seconds=row.get('source_time_seconds'), generation_method='camera_crop_resize')


def _fire(pool):
    config = _source('fire')
    plan_path = config['production_plan']
    plan = _json(plan_path)
    regions = defaultdict(list)
    for i, stratum in enumerate(plan['strata']):
        background = stratum['background']
        if (stratum['target']['count'] != 1 or
                background.get('review_state') != 'no_recognizable_target_in_crop_candidate'):
            continue
        regions[background['image']].append((i, stratum))
    for image, selected in regions.items():
        crop = selected[0][1]['background']['crop_xyxy']
        rw, rh = config['real_facades']['region_reference_size']
        x0, y0, x1, y1 = crop
        patches = []
        for i, state in selected:
            quad = state['target']['quad']
            patches.append({'quad_xy': [[round(x0 + x * (x1-x0)/rw, 4), round(y0 + y * (y1-y0)/rh, 4)]
                                        for x, y in quad], 'plan_region_xy': quad,
                            'evidence': _evidence(plan_path, i, collection='strata')})
        pool.review(image, 'fire', _review(_evidence(plan_path, selected[0][0], collection='strata'),
                    geometry={'crop_xyxy': crop, 'regions': patches, 'scope': 'reviewed_crop_only',
                              'region_reference_size': [rw, rh]}, label_state='candidate_complete_negative_in_crop',
                    split=plan['target_split'], existing_targets=[]), generation_method='camera_capture')


def _paper(pool):
    config = _source('building_name_paper')
    for key in ('production_plan', 'rooftop_sources'):
        path = config[key]
        plan = _json(path)
        for i, stratum in enumerate(plan['strata']):
            for row in stratum['background']['candidates']:
                if not row['regions']:
                    continue
                pool.review(row['image'], 'building_name_paper',
                            _review(_evidence(path, i, collection='strata', detail=row['coverage_evidence']),
                                    geometry={'regions': row['regions'], 'coordinate_size': row['image_size']},
                                    label_state=row['coverage'], split=plan['target_split'],
                                    existing_targets=row['existing_papers']),
                            source_group=row['source_group'], video=row['parent_source'],
                            generation_method=row['generation_method'], image_size=row['image_size'])


def _sign(pool):
    from datasets.scripts.sign.plan_sign_sources import legacy_background_inputs

    records, frame_inputs = legacy_background_inputs(_source('sign'))
    for original, row in zip(records, frame_inputs, strict=True):
        support = row.get('support_rectangle_xyxy')
        if not support or row.get('sign_absence') != 'confirmed':
            continue
        evidence = {**row['source_evidence'], 'detail': row.get('ground_evidence'),
                    'authoring_entry': 'datasets/scripts/sign/plan_sign_sources.py'}
        pool.review(row['image'], 'sign', _review(evidence,
                    geometry={'support_rectangle_xyxy': support, 'coordinate_size': [640, 480],
                              'protected_rectangles_xyxy': row.get('protected_rectangles_xyxy', []),
                              'crop_xyxy': row.get('crop_xyxy')}, label_state='candidate_complete_negative',
                    split='train', existing_targets=[]), video=row.get('parent_video'),
                    source_group=original['source_group'],
                    timestamp_seconds=row.get('pts_seconds'))


def _bin_state(pool):
    path = _source('waste_bin_state')['background_manifest']
    for i, row in enumerate(_json(path)['frames']):
        polygon = row.get('ground_polygon_xy')
        if not polygon or not row.get('state_review'):
            continue
        pool.review(row['image'], 'waste_bin_state', _review(_evidence(path, i, collection='frames',
                    detail=row['state_review']), geometry={'ground_polygon': polygon, 'coordinate_size': [640, 480]},
                    label_state='candidate_complete_negative', split=row.get('split'), existing_targets=[]),
                    video=row['video'], parent_image=row.get('parent_image'),
                    timestamp_seconds=row.get('source_time_seconds'), generation_method='camera_crop_resize')


def _sticker(pool):
    config = _source('waste_sticker')
    path = config['real_photo_geometry_manifest']
    role_path = config['surface_background_roles']
    roles = list(_jsonl(role_path))
    for i, row in enumerate(_jsonl(path)):
        evidence = row['background_role_evidence']
        source = roles[evidence['record_index']]
        if source['image'] != row['path'] or not row.get('bins'):
            raise ValueError(f'Waste sticker surface and role do not match: {row["path"]}')
        uses = [use for use in source['task_use_status'] if use.get('role') == 'background' and use.get('status') == 'ready']
        if not uses:
            continue
        pool.review(row['path'], 'waste_sticker', _review(_evidence(path, i, role=evidence),
                    geometry={'bins': row['bins'], 'coordinate_size': row['image_size']},
                    label_state='candidate_complete_negative', split=uses[0]['split'],
                    existing_targets=[]), video=row['source_video'], image_size=row['image_size'],
                    frame_index=source.get('source_frame'))


def _meter(pool):
    settings = _source('meter')['real_composite']
    path, plan_path = settings['background_review'], settings['production_plan']
    rows = _json(path)['images']
    split = _json(plan_path)['target_split']
    for selected in settings['selected_background_records']:
        i = selected['record_index']
        row = rows[i]
        if row['id'] != selected['id'] or row['pose_state'] != 'candidate_complete_negative':
            raise ValueError(f'Changed configured meter background: {row["image"]}')
        pool.review(row['image'], 'meter', _review(_evidence(path, i, collection='images', plan=plan_path),
                    geometry={'mode': 'alpha_composite_identity', 'coordinate_size': [640, 480],
                              'scope': 'existing center crop and fixed station projection'},
                    label_state=row['pose_state'], split=split, existing_targets=[], historical_role=row.get('split')),
                    video=row['parent_video'], image_size=row['image_size'], timestamp_seconds=row['source_time_seconds'],
                    generation_method=row['generation_method'])


def _review_source_frames(pool, tasks):
    """Measure local ground fit on the original pool without inferring target labels."""
    import cv2
    import numpy as np
    from .review.geometry import estimate_ground_region
    from .review.placement import assess_placement, placement_decision

    for row in pool.by_image.values():
        if row['evidence']['manifest'] != LEGACY_POOL:
            continue
        image = cv2.imdecode(np.fromfile(_file(row['source_image']), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not decode shared background: {row["source_image"]}')
        ground = estimate_ground_region(image, params={'ransac_iters': 100})
        for task in sorted(set(tasks) - row['tasks'].keys()):
            geometry = {}
            evidence = {**row['evidence'], 'scope': 'existing frame and task-specific placement only'}
            uses, state = (), 'unknown'
            if task in GROUND_TARGETS:
                geometry.update(ground_polygon=ground.polygon, placement_patch=ground.placement_patch,
                                coordinate_size=row['image_size'], ground_reason_codes=ground.reason_codes,
                                target_probe=GROUND_TARGETS[task])
                evidence['method'] = 'shared_ground_geometry_and_placement_review'
                if ground.placement_patch:
                    check = assess_placement(image, ground.placement_patch, target=GROUND_TARGETS[task],
                                             params={'rectified_size': 256, 'stability_probes': 0})
                    decision, detail = placement_decision(check.metrics)
                    geometry.update(foot_xy=check.metrics.get('best_footpoint'), placement_decision=decision)
                    evidence['placement'] = detail
                    if decision == 'usable':
                        uses, state = ('geometry_reference',), 'partial'
            else:
                evidence['method'] = 'task_surface_evidence_unavailable'
            row['tasks'][task] = _review(evidence, geometry=geometry, label_state='unknown', state=state, uses=uses)


def _task_allocations(records, tasks):
    allocations = {}
    for task in sorted(tasks):
        groups = defaultdict(deque)
        used_captures = set()
        for row in records:
            use = row['tasks'].get(task, {})
            key = (use.get('split') or 'unassigned', row['capture_identity'])
            if 'composite_background' in use.get('uses', ()) and key not in used_captures:
                used_captures.add(key)
                groups[(key[0], row['parent_group'])].append(row['image'])
        splits = {}
        for split in sorted({key[0] for key in groups}):
            by_parent = {parent: group for (purpose, parent), group in groups.items() if purpose == split}
            counts = {parent: len(queue) for parent, queue in sorted(by_parent.items())}
            ordered = []
            while any(by_parent.values()):
                ordered.extend(by_parent[parent].popleft() for parent in counts if by_parent[parent])
            splits[split] = {'images': ordered, 'parent_counts': counts}
        allocations[task] = splits
    return allocations


def _materialize(row):
    source = _file(row['source_image'])
    target = (WORKSPACE / row['image']).resolve()
    pool_root = (WORKSPACE / POOL_ROOT).resolve()
    if not pool_root.is_relative_to(WORKSPACE) or not target.is_relative_to(pool_root):
        raise ValueError(f'Background alias outside the pool: {row["image"]}')
    if target.exists():
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            raise ValueError(f'Existing background alias differs from its source: {row["image"]}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as work:
        staged = Path(work) / target.name
        shutil.copyfile(source, staged)
        staged.rename(target)


def build(*, pilot=False):
    pool = Pool()
    _legacy_pool(pool)
    _waypoints(pool, pilot)
    for collect in (_person, _e_bike, _fire, _paper):
        collect(pool)
    if not pilot:
        for collect in (_sign, _bin_state, _sticker, _meter):
            collect(pool)
    tasks = PILOT_TASKS if pilot else TASKS
    _review_source_frames(pool, tasks)
    records = sorted(pool.by_image.values(), key=lambda row: row['image'])
    allocations = _task_allocations(records, tasks)
    return {'schema_version': 1, 'role': 'shared_background_pool', 'scope': 'pilot' if pilot else 'eight_tasks',
            'legacy_pool': LEGACY_POOL, 'allocation_basis': 'task and declared split, round robin by physical parent',
            'machine_review_index': MACHINE_REVIEW_INDEX,
            'extensions': [] if pilot else [path for path in RENDER_BATCHES if (WORKSPACE / path).is_file()],
            'counts': {'records': len(records), 'source_kinds': dict(Counter(row['source_kind'] for row in records)),
                       'parents': len({row['parent_group'] for row in records}),
                       'tasks': {task: {split: len(value['images']) for split, value in splits.items()}
                                 for task, splits in allocations.items()},
                       'task_review_states': {task: dict(Counter(row['tasks'][task]['state'] for row in records
                                                                if task in row['tasks'])) for task in sorted(tasks)}},
            'allocations': allocations, 'records': records}


def load_task_backgrounds(task, manifest=POOL_MANIFEST, split=None):
    """Return the explicitly used backgrounds in the recorded parent-balanced order."""
    if task not in TASKS:
        raise ValueError(f'Unknown background task: {task}')
    path = Path(manifest)
    path = path.relative_to(WORKSPACE) if path.is_absolute() else path
    data = _json(path)
    if data.get('schema_version') != 1 or data.get('role') != 'shared_background_pool':
        raise ValueError('Expected a version 1 shared background pool')
    records = {row['image']: row for row in data['records']}
    assignment = data['allocations'].get(task, {})
    splits = assignment.keys() if split is None else (split,)
    return [records[image] for purpose in splits
            for image in assignment.get(purpose, {}).get('images', [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Build the four-task P02/P09 pilot')
    parser.add_argument('--write', action='store_true', help='Copy approved source representations and publish the catalog')
    args = parser.parse_args()
    data = build(pilot=args.pilot)
    if args.write:
        output = WORKSPACE / POOL_MANIFEST
        if args.pilot and output.exists() and _json(POOL_MANIFEST).get('scope') == 'eight_tasks':
            raise ValueError('The eight-task catalog already exists; run the pilot as a dry run')
        for kind in ('camera', 'rendered'):
            (WORKSPACE / POOL_ROOT / kind).mkdir(parents=True, exist_ok=True)
        for row in data['records']:
            _materialize(row)
        with TemporaryDirectory(dir=output.parent) as work:
            staged = Path(work) / output.name
            staged.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            staged.replace(output)
    print(json.dumps(data['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
