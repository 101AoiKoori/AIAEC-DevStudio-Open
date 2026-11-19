"""Materialize task-local target inputs from the existing evidence catalogs."""

from __future__ import annotations

import argparse
from collections import Counter
import filecmp
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from PIL import Image

from .config import read_yaml
from .paths import TASKS, WORKSPACE
from .source_history import capture_identity, historical_source, read_source_history


CAR = 'datasets/annotated/car_real_capture_20261112/direct_real_manifest.json'
ARCHIVE = 'datasets/annotated/common/sources/archive_real_capture.json'
OUTPUT = Path('datasets/annotated/common/sources/targets')
PILOT = {'person': 'P02_PS1.png', 'e_bike': 'P02_PS1.png',
         'fire': 'P09_B1.png', 'building_name_paper': 'P09_B1.png'}
PERSON_ARTWORK = Path('Archive/Assets/working_sources/digital/person/team_assets')
PERSON_TEXTURES = 'datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json'
FIRE_ARTWORK = 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'
PAPER_FONT = Path('Archive/technical/revisions/fire/blender_sources_20261116/blender_buildings_v13/NotoSansSC.ttf')
STICKER_ARTWORK = 'datasets/annotated/waste_bin/waste_sticker_artwork_source_manifest_20261114.jsonl'
METER_FACE = 'Archive/technical/revisions/meter/blender_sources_20261116/v4_parametric_station_v2_20261113/face.png'
BIKE_CUTOUTS = 'datasets/annotated/e_bike/sam2_orbit_candidates_20261110/selection_review_candidate.json'
BIKE_FITTED_TEXTURES = 'datasets/annotated/person/v4_vehicle_occluders_20261113/manifest.json'
SIGN_CUTOUTS = Path('datasets/annotated/sign/sam2_fixture_candidates_20261110/manifest.json')
PERSON_DIRECT = 'datasets/scripts/person/config/direct_annotation_layers_20261115.json'
SITE = 'datasets/annotated/site_capture_20261112'
PAPER_DIRECT_SELECTION = 'datasets/scripts/building_name_paper/config/direct_selection.json'
SITE_REVIEWS = (
    (f'{SITE}/fire_name_v4_review/review_manifest.json', 'images', {'fire', 'building_name_paper'}),
    (f'{SITE}/fire_name_v4_review/batch2/review_manifest.json', 'images', {'fire', 'building_name_paper'}),
    (f'{SITE}/person_vehicle_v4_review/review_manifest.json', 'images', {'person', 'e_bike'}),
    (f'{SITE}/sign_plate_v4_review/review_manifest.json', 'samples', {'sign'}),
)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def relative(value, root):
    path = Path(value)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError(f'Expected a workspace-relative path: {value}')
    return resolved


def copy_once(source, target):
    """Publish a byte-identical task working copy without overwriting another file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_copy(source, target)
        return
    with TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / target.name
        shutil.copyfile(source, stage)
        try:
            if os.name == 'nt':
                stage.rename(target)  # Windows rename requires the destination to be absent.
            else:
                os.link(stage, target)
        except FileExistsError:
            verify_copy(source, target)


def verify_copy(source, target):
    if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
        raise ValueError(f'Target working copy conflicts with its source: {target}')


def group_folder(group, parent):
    if re.fullmatch(r'[a-z0-9_]+', group):
        return group
    stem = Path(parent or group).stem
    safe = re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_') or 'asset'
    digest = sha256(group.encode('utf-8')).hexdigest()[:10]
    return f'archive_{safe[:36]}_{digest}'


def camera_asset(row, task, source_manifest, index, history):
    image = Path(row['image'])
    source_kind = row.get('source_kind') or ''
    is_frame = source_kind.endswith('video_frame') or any(p.startswith('frames_') for p in image.parts)
    frame = row.get('frame_index')
    parent = row.get('parent_video') if is_frame else None
    folder = group_folder(historical_source(row['source_group'], history),
                          historical_source(parent, history) if parent else None)
    if is_frame and (source_manifest == CAR or source_kind == 'site_camera_video_frame') and frame is not None:
        name = f'f{frame:06d}{image.suffix.lower()}'
    elif is_frame:
        # Archive views may be crops or re-indexed samples of the same decoded frame.
        suffix = re.sub(r'[^a-z0-9_]+', '_', row['id'].lower()).strip('_')
        name = f'f{frame:06d}__{suffix}{image.suffix.lower()}' if frame is not None else image.name
        if frame is None:
            name = f'{suffix}{image.suffix.lower()}'
    else:
        name = image.name
    target = Path('camera', folder, 'frames' if is_frame else 'photos', name)
    copied = dict(row)
    copied.update(tasks={task: row['tasks'][task]}, source_image=row['image'],
                  source_kind='camera',
                  generation_method=row.get('generation_method') or ('video_frame' if is_frame else 'camera_capture'),
                  source_manifest=source_manifest, source_record_index=index)
    copied['capture_identity'] = capture_identity(row, history)
    return target, copied


def target_present(truth):
    return truth['state'] == 'complete_positive' or (
        truth['state'] == 'partial' and (
            truth.get('instances') or truth.get('candidate_instances') or truth.get('prior_observation')
            or 'artwork_reference' in truth.get('uses', [])))


def camera_assets(task, pilot, catalogs, history):
    for manifest, rows in catalogs.items():
        for index, row in enumerate(rows):
            if not row.get('image') or task not in row['tasks']:
                continue
            if pilot and (manifest != CAR or Path(row['image']).name != PILOT[task]):
                continue
            if target_present(row['tasks'][task]):
                yield camera_asset(row, task, manifest, index, history)


def site_metadata(row, batch, evidence, root, history):
    image = row['image']
    canonical = row.get('canonical_image', image)
    video = (evidence.get('parent_video') or evidence.get('source_video') or batch.get('source_video')
             or batch.get('video'))
    frame = next((evidence[key] for key in ('frame_index', 'frame_index_1based', 'source_frame_index_1based')
                  if key in evidence), None)
    if frame is None:
        # These are output ordinals from the site sequence, distinct from source PTS.
        match = re.fullmatch(r'frame_(\d+)', Path(canonical).stem)
        frame = int(match.group(1)) if match else None
    timestamp = next((evidence[key] for key in ('timestamp_seconds', 'source_time_seconds', 'parent_pts_time_seconds')
                      if key in evidence), None)
    group = evidence.get('source_group') or batch.get('source_group')
    if not group:
        group = 'site_capture_20261112_' + (Path(video).stem if video else 'photos')
    with Image.open(relative(image, root)) as pixels:
        size = list(pixels.size)
    return {'id': f'site_review:{historical_source(image, history)}', 'image': image,
            'capture_identity': historical_source(canonical, history),
            'source_kind': 'site_camera_video_frame' if video else 'site_camera_photo',
            'generation_method': evidence.get('generation_method'), 'source_group': group,
            'parent_video': video, 'frame_index': frame, 'timestamp_seconds': timestamp,
            'segment': evidence.get('segment_id'), 'image_size': size}


def site_truth(task, truth, evidence, manifest, index, classes):
    state = truth['state'].removeprefix('candidate_')
    instances = []
    for box in truth.get('boxes') or truth.get('boxes_xyxy') or []:
        if isinstance(box, dict):
            candidate = dict(box)
        elif len(box) == 2:
            candidate = {'class_id': box[0], 'xyxy': box[1]}
        else:
            candidate = {'class_id': 0, 'xyxy': box}
        cls = candidate.get('class_id')
        if type(cls) is int and 0 <= cls < len(classes):
            candidate.setdefault('class_name', classes[cls])
        instances.append(candidate)
    result = {'state': state, 'instances': instances, 'label': truth.get('label'),
              'label_level': 'auto_candidate', 'uses': ['review_reference'], 'split': evidence.get('split'),
              'evidence': {'manifest': manifest, 'record_index': index, 'review_evidence': truth.get('review_evidence')}}
    if task == 'building_name_paper' and state == 'complete_positive' and not all(b.get('text') for b in instances):
        result.update(state='partial', detection_state=state, missing_supervision='legible_ocr_text')
    return result


def person_layer_rows(root):
    evidence_batches = {}
    for index, layer in enumerate(load_json(root / PERSON_DIRECT)['records']):
        if layer['state'] != 'complete_positive' and not (layer['state'] == 'partial' and layer.get('candidate_boxes')):
            continue
        manifest = layer['source_manifest']
        if manifest not in evidence_batches:
            evidence_batches[manifest] = load_json(root / manifest)
        batch = evidence_batches[manifest]
        if batch.get('image'):
            evidence = batch
        else:
            evidence = next(row for row in batch.get('images', batch.get('samples', [])) if row['id'] == layer['id'])
        truth = {**evidence.get('tasks', {}).get('person', evidence), 'state': layer['state'], 'label': layer['label']}
        yield PERSON_DIRECT, index, layer, batch, evidence, truth


def site_review_assets(task, catalogs, root, history):
    if task not in {'person', 'e_bike', 'fire', 'building_name_paper', 'sign'}:
        return
    known = {key for rows in catalogs.values() for row in rows
             for key in (row.get('image'), row.get('original_path')) if key}
    targets = read_yaml(root / 'config/task_targets.yaml')['tasks'][task]['classes']
    selected_direct = ({row['image']: row for row in load_json(root / PAPER_DIRECT_SELECTION)['records']}
                       if task == 'building_name_paper' else {})

    def site_records():
        if task == 'person':
            yield from person_layer_rows(root)
        for manifest, collection, tasks in SITE_REVIEWS:
            if task not in tasks:
                continue
            batch = load_json(root / manifest)
            for index, row in enumerate(batch[collection]):
                truth = row['tasks'].get('fire_sticker' if task == 'fire' else task)
                if truth:
                    yield manifest, index, row, batch, row, truth

    for manifest, index, row, batch, evidence, truth in site_records():
        canonical = row.get('canonical_image', row['image'])
        if canonical in known or row['image'] in known:
            continue
        annotation = site_truth(task, truth, evidence, manifest, index, targets)
        direct = selected_direct.get(row['image']) or selected_direct.get(canonical)
        if not target_present(annotation) and direct is None:
            continue
        if direct is not None:
            annotation['uses'] = ['empty_negative' if annotation['state'] == 'complete_negative'
                                  else 'direct_target']
            annotation['split'] = direct['split']
            annotation['selection_evidence'] = {'manifest': PAPER_DIRECT_SELECTION,
                                                'record_index': list(selected_direct).index(direct['image'])}
        metadata = site_metadata(row, batch, evidence, root, history)
        metadata['tasks'] = {task: annotation}
        yield camera_asset(metadata, task, manifest, index, history)
        known.update([canonical, row['image']])


def asset(source, destination, group, kind, method, history, **metadata):
    source = Path(source).as_posix()
    return destination, {'id': f'{kind}:{historical_source(source, history)}',
                         'source_image': source, 'source_group': group,
                         'source_kind': kind, 'generation_method': method, 'tasks': {}, **metadata}


def digital_assets(task, root, history):
    source_config = f'datasets/scripts/{task}/config/source.yaml'
    if task == 'person':
        rows = load_json(root / PERSON_TEXTURES)['records']
        for index, row in enumerate(rows):
            original = Path(row['artwork'])
            group = f'person_team_artwork:{original.relative_to(PERSON_ARTWORK).as_posix()}'
            yield asset(original, Path('digital/team_artwork') / original.relative_to(PERSON_ARTWORK),
                        group, 'digital', 'digital_artwork', history, role='target_foreground', class_name=row['class'],
                        source_manifest=PERSON_TEXTURES, source_record_index=index)
            image = Path(row['image'])
            yield asset(image, Path('digital/whiteborder_textures') / image.name, group,
                        'digital', 'image_processing', history, role='target_foreground', class_name=row['class'],
                        source_manifest=PERSON_TEXTURES, source_record_index=index, parent_image=original.as_posix())
    elif task == 'fire':
        yield asset(FIRE_ARTWORK, Path('digital/team_artwork/image1.png'), 'fire_team_artwork',
                    'digital', 'digital_artwork', history, role='target_foreground', class_id=0,
                    source_manifest=source_config)
    elif task == 'building_name_paper':
        license_path = PAPER_FONT.with_name('OFL.txt')
        yield asset(PAPER_FONT, Path('digital/typeface') / PAPER_FONT.name, 'building_name_paper_typeface',
                    'digital', 'digital_typeface', history, role='text_generation_font', source_manifest=source_config,
                    license=license_path.as_posix())
        yield asset(license_path, Path('digital/typeface') / license_path.name, 'building_name_paper_typeface',
                    'digital', 'license_record', history, role='license', source_manifest=source_config)
    elif task == 'meter':
        yield asset(METER_FACE, Path('digital/station_face/face.png'), 'meter_parametric_station_face',
                    'digital', 'digital_texture', history, role='target_texture', source_manifest=source_config,
                    source_scene=Path(METER_FACE).with_name('station.blend').as_posix())
    elif task == 'waste_sticker':
        with (root / STICKER_ARTWORK).open(encoding='utf-8-sig') as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                for use in row['task_use_status']:
                    if (use['task'], use['role'], use['status']) != ('waste_sticker', 'target_foreground', 'ready'):
                        continue
                    image = Path(row['image'])
                    yield asset(image, Path('digital/official_item_artwork') / image.name, row['source_group'],
                                'digital', 'digital_artwork', history, role='target_foreground', class_id=use['class_id'],
                                class_name=use['class_name'], source_manifest=STICKER_ARTWORK, source_record_index=index)


def cutouts(task, root, history):
    selected = read_yaml(root / f'datasets/scripts/{task}/config/source.yaml')['current_selection']
    if task == 'e_bike':
        allowed = {row['image'] for row in selected['composite']['foregrounds']}
        manifests = ((BIKE_CUTOUTS, 'vehicles', 'path', ''),
                     (BIKE_FITTED_TEXTURES, 'records', 'cropped_image', '__cropped'))
        for manifest, key, image_key, suffix in manifests:
            for index, row in enumerate(load_json(root / manifest)[key]):
                if row['path'] not in allowed:
                    continue
                video = row['source_video']
                group = row.get('archive_original_for_source_video') or historical_source(video, history)
                image = Path(row[image_key])
                folder = group_folder(group, historical_source(video, history))
                yield asset(image, Path('camera', folder, 'cutouts', f'{image.stem}{suffix}{image.suffix}'),
                            group, 'cutout', 'camera_cutout', history, role='target_foreground', class_name=row['class'],
                            parent_video=video, parent_image=row['source_frame'],
                            source_manifest=manifest, source_record_index=index)
    elif task == 'sign':
        allowed = set(selected['foreground_masks'])
        for index, row in enumerate(load_json(root / SIGN_CUTOUTS)['records']):
            declared = row.get('v4_source_selection') or {}
            image = Path(declared.get('foreground_mask', ''))
            if image.name not in allowed:
                continue
            video = declared['parent_video']
            group = historical_source(video, history)
            folder = group_folder(group, group)
            yield asset(image, Path('camera', folder, 'cutouts', image.name), group,
                        'cutout', 'camera_cutout', history, role='target_foreground', class_name='sign_fixture',
                        parent_video=video, parent_image=row['source'], source_manifest=SIGN_CUTOUTS.as_posix(),
                        source_record_index=index)


def reusable_renders(task, root, history):
    path = Path('datasets/samples') / task / 'manifest.json'
    if not (root / path).is_file():
        return
    batch = load_json(root / path)
    if task == 'meter' and batch.get('status') == 'complete_synthetic_pose_candidates':
        with (root / path.parent / batch['samples']).open(encoding='utf-8-sig') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        for row in records:
            foreground = row.get('foreground_rgba')
            if foreground:
                image = path.parent / foreground
                annotation = path.parent / 'annotations' / f'{row["id"]}.json'
                if not (root / annotation).is_file():
                    raise ValueError(f'Meter foreground requires its projected points: {annotation}')
                yield asset(image, Path('rendered', 'arena_meter', image.name), batch['arena'],
                            'rendered', 'blender_render', history, role='target_foreground',
                            source_scene=batch['arena'], source_blend=batch['arena'], target_model_scene=batch['station'],
                            source_manifest=path.as_posix(), source_annotation=annotation.as_posix())
    elif task == 'waste_bin_state' and batch.get('status') == 'complete_candidates':
        for index, row in enumerate(batch['samples']):
            if not row.get('visible_bins'):
                continue
            image = path.parent / row['image']
            masks = {key: (path.parent / row[key]).as_posix() for key in ('mask_visible', 'mask_state_visible')}
            if any(not (root / name).is_file() for name in masks.values()):
                raise ValueError(f'Bin render requires visible instance and state masks: {image}')
            yield asset(image, Path('rendered', 'standalone_bins', image.name), batch['scene_file'],
                        'rendered', 'blender_render', history, role='target_foreground', source_scene=batch['scene_file'],
                        source_blend=batch['scene_file'],
                        source_manifest=path.as_posix(), source_record_index=index, related_masks=masks)


def all_assets(task, pilot, catalogs, root, history):
    yield from camera_assets(task, pilot, catalogs, history)
    if pilot:
        return
    yield from site_review_assets(task, catalogs, root, history)
    yield from digital_assets(task, root, history)
    if task in {'e_bike', 'sign'}:
        yield from cutouts(task, root, history)
    if task in {'meter', 'waste_bin_state'}:
        yield from reusable_renders(task, root, history)


def materialize(task, assets, root, *, pilot):
    task_root = root / 'datasets/raw' / task / 'targets'
    for kind in ('digital', 'camera', 'rendered'):
        relative((task_root / kind).relative_to(root), root).mkdir(parents=True, exist_ok=True)
    records = []
    paths = set()
    for name, record in assets:
        destination = relative((task_root / name).relative_to(root), root)
        source = relative(record['source_image'], root)
        if not destination.is_relative_to(task_root) or source == destination or not source.is_file():
            raise ValueError(f'Invalid {task} target source or destination: {record["source_image"]}')
        if destination.as_posix().casefold() in paths:
            raise ValueError(f'Duplicate target destination: {destination}')
        paths.add(destination.as_posix().casefold())
        copy_once(source, destination)
        records.append(dict(record, image=destination.relative_to(root).as_posix()))
    output = root / OUTPUT / task / 'target_sources_manifest.json'
    output = relative(output.relative_to(root), root)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = load_json(output) if output.exists() else None
    if pilot and previous and previous.get('scope') == 'full':
        return Counter(row['source_kind'] for row in records)
    payload = {'schema_version': 1, 'task': task, 'scope': 'pilot' if pilot else 'full', 'records': records}
    if not pilot and task in {'meter', 'waste_bin_state'} and not any(r['source_kind'] == 'rendered' for r in records):
        payload['pending_render'] = {'producer_batch': f'datasets/samples/{task}/manifest.json',
                                     'status': 'no_current_complete_reusable_target_render'}
    if payload != previous:
        with TemporaryDirectory(dir=output.parent) as temporary:
            stage = Path(temporary) / output.name
            stage.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            stage.replace(output)
    return Counter(row['source_kind'] for row in records)


def run(*, pilot=False, tasks=None, root=WORKSPACE):
    names = [CAR] if pilot else [CAR, ARCHIVE]
    catalogs = {name: load_json(root / name)['records'] for name in names}
    history = read_source_history(root)
    chosen = sorted(PILOT if pilot else TASKS) if tasks is None else sorted(set(tasks))
    if set(chosen) - (PILOT if pilot else TASKS):
        raise ValueError(f'Unsupported target source task: {chosen}')
    return {task: dict(materialize(task, all_assets(task, pilot, catalogs, root, history), root, pilot=pilot))
            for task in chosen}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Materialize the P02 and P09 task views')
    parser.add_argument('--task', action='append', choices=sorted(TASKS),
                        help='Only refresh the selected task; repeatable')
    args = parser.parse_args()
    print(json.dumps(run(pilot=args.pilot, tasks=args.task), ensure_ascii=False))


if __name__ == '__main__':
    main()
