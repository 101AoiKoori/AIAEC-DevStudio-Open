"""Validate and publish waste-bin-state splits and sync their training index."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.common import package_direct_real as direct
from datasets.scripts.common.config import load_task_config, read_yaml
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.paths import dataset_dir, split_dir
from datasets.scripts.common.source_history import RELOCATIONS, historical_source, read_source_history
from datasets.scripts.common.source_ledger.config import build_groups, normalize_path
from datasets.scripts.common.source_ledger.reservations import candidate_reservation
from datasets.scripts.waste_bin_state import build_current_sample as verified
from lab.training_common.production_guard import generation_guard

TASK = 'waste_bin_state'
SPLIT_ORDER = ('train', 'val', 'test')
CAMERA_KINDS = {'direct_real', 'direct_empty_negative', 'real_background_composite'}
ANNOTATED = ROOT / 'datasets/annotated/waste_bin_state'
HISTORY = ANNOTATED / 'full_export_history'
COMPOSITES = ANNOTATED / 'full_composites'


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def physical_group(record, relocations):
    group = record.get('parent_group') or record.get('parent_video')
    if not group:
        raise ValueError(f'Sample has no physical parent group: {record["source_image"]}')
    return historical_source(normalize_path(group), relocations)


def direct_rows(split, relocations):
    selected, _ = direct.prepare(TASK, split=split)
    rows = []
    for manifest, original, truth, image, parent, label, text in selected:
        record = {
            'source_kind': 'direct_real' if text else 'direct_empty_negative',
            'split': split,
            'source_image': relative(image),
            'source_manifest': relative(manifest),
            'source_record_id': original['id'],
            'parent_group': original.get('source_group') or original.get('parent_video') or relative(image),
            'parent_video': relative(parent) if parent else None,
            'capture_identity': original['capture_identity'],
            'frame_environment': original['frame_environment'],
            'annotation': truth,
            'bins': [
                {'color': item.get('color') or item.get('bin_color') or 'unknown',
                 'state': verified.CLASS_NAMES[item['class_id']], 'bbox_xyxy': item['xyxy']}
                for item in truth['instances'] if item.get('disposition', 'target') == 'target'
            ],
        }
        record['physical_parent_group'] = physical_group(record, relocations)
        media = {image, *([parent] if parent else [])}
        metadata = {manifest, *([label] if label else [])}
        if original.get('canonical_manifest'):
            media.add(direct.workspace_file(original['source_image']))
            metadata.add(direct.workspace_file(original['canonical_manifest']))
        rows.append((image, text, record, media, metadata))
    return rows


def blender_empty_rows(database, relocations, count=None):
    """Stratify admitted full-frame absences by scene and route, then space picks across each route."""
    groups = defaultdict(list)
    unique = set()
    candidates = query_frames(task=TASK, role='empty_negative', split='train', database=database)
    ordered = sorted(candidates, key=lambda row: (str(row.get('scene')), str(row.get('route')),
                                                   row.get('render_frame') or 0, row['frame_id']))
    for candidate in ordered:
        if candidate.get('source_kind') != 'blender_render':
            continue
        label = candidate['label']
        if (label.get('completeness') != 'complete' or label.get('presence') != 'absent'
                or label.get('scope', {}).get('kind') != 'frame' or label.get('instances')):
            raise ValueError('Blender empty frame lacks complete absence evidence: ' + candidate['image'])
        identity = candidate['content_sha256']
        if identity not in unique:
            groups[candidate['scene'], candidate['route']].append(candidate)
            unique.add(identity)
    count = len(unique) if count is None else count
    if not 0 <= count <= len(unique):
        raise ValueError(f'Plan needs {count} generated empty frames, with {len(unique)} eligible. Rebalance the batch.')
    budgets = dict.fromkeys(groups, 0)
    remaining = count
    while remaining:
        for key, frames in groups.items():
            if budgets[key] < len(frames):
                budgets[key] += 1
                remaining -= 1
                if not remaining:
                    break

    rows = []
    for key, frames in groups.items():
        count = budgets[key]
        for index in range(count):
            candidate = frames[((2 * index + 1) * len(frames)) // (2 * count)]
            image = direct.workspace_file(candidate['image'])
            with image.open('rb') as stream:
                if file_digest(stream, 'sha256').hexdigest() != candidate['content_sha256']:
                    raise ValueError('Blender empty image has changed: ' + candidate['image'])
            manifest = direct.workspace_file(candidate['label']['evidence']['manifest'])
            record = {
                'source_kind': 'blender_empty_negative', 'split': 'train',
                'source_image': relative(image), 'source_manifest': relative(manifest),
                'source_record_id': candidate['record_id'], 'parent_group': candidate['parent_group'],
                'mother_scene': candidate['mother_scene'], 'scene': key[0], 'route': key[1],
                'render_frame': candidate['render_frame'], 'capture_identity': candidate['capture_identity'],
                'frame_environment': {
                    **{field: candidate[field] for field in ('frame_id', 'record_id', 'split', 'role', 'content_sha256')},
                    'evidence_ids': [item['evidence_id'] for item in candidate['evidence']],
                },
                'annotation': candidate['label'], 'bins': [],
            }
            record['physical_parent_group'] = physical_group(record, relocations)
            rows.append((image, '', record, {image}, {manifest}))
    return rows


def composition_files(split):
    if split == 'train':
        base = ANNOTATED / 'full_plans'
        plan, audit = (base / f'train_composite_{kind}.json' for kind in ('plan', 'allocation'))
    else:
        base = ANNOTATED / 'full_evaluation' / split
        plan, audit = (base / f'composite_{kind}.json' for kind in ('plan', 'allocation'))
    return COMPOSITES / split / 'manifest.json', plan, audit


def verify_pair_allocation(split, states, audit_path):
    audit = verified.read_json(audit_path)
    if split == 'train':
        allocations = [allocation for branches in audit['allocations'].values()
                       for allocation in branches.values()]
    else:
        allocations = [audit['pair_allocation']]
    if any(allocation['target_split'] != split for allocation in allocations):
        raise ValueError(f'Allocation contains pairs from a different split: {split}')
    planned_pairs = [row['id'] for allocation in allocations for row in allocation['strata']
                     if row['planned_count'] == 1]
    if Counter(row['pair_id'] for row in states.values()) != {
            pair: 2 * count for pair, count in Counter(planned_pairs).items()}:
        raise ValueError(f'Execution plan does not match the Bayesian pair allocation: {split}')


def composition_rows(split, database, relocations):
    manifest_path, plan_path, audit = composition_files(split)
    if not all(path.is_file() for path in (manifest_path, plan_path, audit)):
        raise ValueError(f'Missing completed composition, final plan or allocation audit for {split}')
    manifest = verified.read_json(manifest_path)
    plan = verified.read_json(plan_path)
    if (manifest.get('status') != 'complete_candidates' or manifest.get('model_key') != TASK
            or manifest.get('split') != split or plan.get('target_split') != split):
        raise ValueError(f'Invalid {split} composition receipt or plan')
    states = {row['id']: row for row in plan['strata']}
    materialized = manifest['records']
    if (not materialized or len(states) != len(materialized) or len(materialized) != plan['budget']
            or {row['state_id'] for row in materialized} != states.keys()
            or any(row['min_count'] != 1 or row['max_count'] != 1 for row in states.values())):
        raise ValueError(f'Completed {split} composition does not match the allocated image states')
    verify_pair_allocation(split, states, audit)

    eligible = {row['record_id']: row for row in query_frames(
        task=TASK, role='composite_background', split=split, database=database)}
    render_splits = {}
    rows = []
    for output in materialized:
        actual = output['actual']
        background = actual['background']
        authority = eligible.get(background['record_id'])
        if (background['split'] != split or not authority or authority['frame_id'] != background['frame_id']
                or background['image'] not in authority['paths']):
            raise ValueError(f'Composition background eligibility changed: {output["state_id"]}')
        image, label = (direct.workspace_file(output[key]) for key in ('image', 'label'))
        if image.parent != manifest_path.parent / 'images' or label.parent != manifest_path.parent / 'labels':
            raise ValueError(f'Composition receipt points outside its {split} batch')
        if image.stem != label.stem:
            raise ValueError('Composition label does not pair with its image: ' + output['image'])
        render_manifest = direct.workspace_file(actual['render_manifest'])
        if render_manifest not in render_splits:
            render_splits[render_manifest] = verified.read_json(render_manifest)['split']
        if render_splits[render_manifest] != split:
            raise ValueError(f'Cross-split composition foreground: {output["state_id"]}')
        state = states[output['state_id']]
        kind = ('blender_route_composite' if authority.get('source_kind') == 'blender_render'
                else 'real_background_composite')
        expected_bins = state['target']['bins']
        observed_bins = {item['bin']: item['state'] for item in actual['bins']}
        if (output.get('status') != 'candidate' or state['axes']['source_kind'] != kind
                or expected_bins != observed_bins or len(expected_bins) != len(actual['bins'])):
            raise ValueError(f'Composition origin or inserted bin state differs from plan: {output["state_id"]}')
        record = {
            'source_kind': kind, 'split': split,
            'source_image': relative(image), 'source_manifest': relative(manifest_path),
            'source_record_id': output['state_id'],
            'parent_group': authority['parent_group'],
            'parent_video': authority.get('parent_video'),
            'frame_environment': {
                **background['frame_environment'], 'role': 'composite_background', 'split': split,
                'content_sha256': authority['content_sha256'],
            },
            'background_image': background['image'], 'render_image': actual['render_image'],
            'render_manifest': relative(render_manifest), 'allocation_axes': state.get('axes') or {},
            'pair_id': state.get('pair_id'), 'scale_basis': actual['scale_basis'],
            'augmentation': actual['augmentation'],
            'bins': [
                {'color': item.get('bin') or item.get('color') or item.get('bin_color') or 'unknown',
                 'state': item['state'], 'bbox_xyxy': item['bbox_xyxy']}
                for item in (*actual['bins'], *actual['retained_background_bins'])
            ],
        }
        record['physical_parent_group'] = physical_group(record, relocations)
        rows.append((image, label.read_text(encoding='utf-8'), record, {image},
                     {manifest_path, plan_path, audit, label, render_manifest}))
    return rows


def filter_direct(camera, compositions, pairing, history):
    """Apply declared parent reservations and keep actual camera frames disjoint."""
    groups = build_groups(TASK, pairing)
    frames = {}
    for split in SPLIT_ORDER:
        for _, _, row, *_ in compositions[split]:
            if row['source_kind'] not in CAMERA_KINDS:
                continue
            reservation = candidate_reservation({**row, 'workspace_image': row['background_image']},
                                                split, groups, history)
            if reservation['status'] == 'conflict':
                raise ValueError(f'Camera composition conflicts with source.yaml#pairing in {split}: '
                                 f'{row["source_record_id"]}: {reservation["conflicts"]}')
            frame = row['frame_environment']['content_sha256']
            if frames.setdefault(frame, split) != split:
                raise ValueError(f'Camera frame shared across production splits: {frame}')

    exclusions = []
    result = {}
    for split in SPLIT_ORDER:
        result[split] = []
        for entry in camera[split]:
            row = entry[2]
            frame = row['frame_environment']['content_sha256']
            reservation = candidate_reservation(row, split, groups, history)
            reason = None
            if reservation['status'] == 'conflict':
                reason = 'camera_parent_reserved_in_other_split'
            elif frames.get(frame, split) != split:
                reason = 'camera_frame_reserved_in_other_split'
            if reason:
                exclusions.append({'split': split, 'reason': reason, 'source_record_id': row['source_record_id'],
                                   'physical_parent_group': row['physical_parent_group'],
                                   'source_image': row['source_image'], 'reservation': reservation})
                continue
            frames[frame] = split
            result[split].append(entry)
    return result, exclusions


def camera_parent_reuse(rows):
    counts = defaultdict(Counter)
    for split, entries in rows.items():
        for _, _, row, *_ in entries:
            if row['source_kind'] in CAMERA_KINDS:
                counts[row['physical_parent_group']][split] += 1
    return [{'relation': 'source_group', 'source': group, 'splits': list(by_split),
             'counts': dict(by_split)}
            for group, by_split in sorted(counts.items()) if len(by_split) > 1]


def train_target():
    return read_yaml(ROOT / 'config/full_train_allocation.yaml')['tasks'][TASK]['total']


def prepare_dataset(database, source):
    history = read_source_history(ROOT)
    camera = {split: direct_rows(split, history) for split in SPLIT_ORDER}
    composites = {split: composition_rows(split, database, history) for split in SPLIT_ORDER}
    camera, exclusions = filter_direct(camera, composites, source['pairing'], history)
    target = train_target()
    empty = blender_empty_rows(database, history, target - len(camera['train']) - len(composites['train']))
    rows = {split: camera[split] + (empty if split == 'train' else []) + composites[split]
            for split in SPLIT_ORDER}
    if len(rows['train']) != target:
        raise ValueError(f'Train quota mismatch: {len(rows["train"])} of {target}')
    return rows, exclusions


def verify_and_summarize(rows):
    extras, seen = {}, {}
    for split, entries in rows.items():
        for image, text, record, *_ in entries:
            record['image_size'] = verified.verify_pair(image, text, record)
            source_kind = record['source_kind']
            if source_kind in {'direct_real', 'direct_empty_negative', 'blender_empty_negative'}:
                if record['image_size'] != record['annotation'].get('evidence', {}).get(
                        'coordinate_size', record['image_size']):
                    raise ValueError('Direct coordinate size changed: ' + record['source_image'])
            with image.open('rb') as stream:
                digest = file_digest(stream, 'sha256').hexdigest()
            if digest in seen:
                raise ValueError(f'Duplicate exported image content: {relative(image)} and {seen[digest]}')
            seen[digest] = relative(image)
            record['image_sha256'] = digest
            if source_kind in {'direct_real', 'direct_empty_negative', 'blender_empty_negative'}:
                if digest != record['frame_environment']['content_sha256']:
                    raise ValueError('Direct/empty image identity changed: ' + record['source_image'])
        records = [row[2] for row in entries]
        classes = set(item['state'] for row in records for item in row['bins'])
        if not {'open', 'closed'}.issubset(classes) or all(row['bins'] for row in records):
            raise ValueError(f'{split} requires open, closed and admitted empty scenes')
        manifest, plan, audit = composition_files(split)
        extras[split] = {
            **verified.summarize(records),
            'physical_parent_groups': dict(Counter(row['physical_parent_group'] for row in records)),
            'composition_manifest': relative(manifest), 'composition_plan': relative(plan),
            'allocation_audit': relative(audit),
        }
    return extras


def training_sources(base, published):
    """Translate verified export records into the existing V4 training schema."""
    dataset = verified.read_json(base / 'manifest.json')
    spec = read_yaml(base / 'data.yaml')
    names = list(verified.CLASS_NAMES)
    if (dataset.get('task') != TASK or dataset.get('status') != 'complete_dataset' or
            dataset.get('class_names') != names or spec.get('names') != names):
        raise ValueError('Expected the current complete waste-bin-state dataset')
    records, seen = [], set()
    for split in SPLIT_ORDER:
        batch = dataset['splits'][split]
        manifest = base / split / 'manifest.json'
        part = verified.read_json(manifest)
        if (batch['manifest'] != f'{split}/manifest.json' or spec.get(split) != f'{split}/images' or
                part.get('task') != TASK or part.get('split') != split or part.get('class_names') != names or
                part.get('status') != 'complete_dataset_split' or len(part['records']) != batch['images']):
            raise ValueError(f'Export data.yaml and manifest disagree for {split}')
        review_evidence = relative(published / split / manifest.name)
        for row in part['records']:
            image = (manifest.parent / row['image']).resolve()
            label = (manifest.parent / row['label']).resolve()
            if (row['split'] != split or image.parent != manifest.parent / 'images' or
                    label != manifest.parent / 'labels' / f'{image.stem}.txt' or image in seen):
                raise ValueError(f'Invalid or duplicate {split} image/label pair: {row["image"]}')
            seen.add(image)
            with image.open('rb') as stream:
                if file_digest(stream, 'sha256').hexdigest() != row['image_sha256']:
                    raise ValueError('Exported image differs from the batch manifest: ' + relative(image))
            if verified.verify_pair(image, label.read_text(encoding='utf-8'), row) != row['image_size']:
                raise ValueError('Exported image size differs from the batch manifest: ' + relative(image))
            source_manifest = direct.workspace_file(row['source_manifest'])
            if not row['physical_parent_group'] or not row['source_record_id']:
                raise ValueError('Exported pair lacks a parent or source record: ' + relative(image))
            records.append({
                'image': relative(published / image.relative_to(base)),
                'label': relative(published / label.relative_to(base)),
                'workspace_image': row['source_image'],
                'split': split,
                'source_id': row['physical_parent_group'],
                'source_kind': row['source_kind'],
                'source_record_id': row['source_record_id'],
                'label_status': 'reviewed',
                'review_state': 'complete_positive' if row['bins'] else 'complete_negative',
                'review_evidence': review_evidence,
                'source_review_evidence': relative(source_manifest),
                # Keep task diagnostics tied to the verified label order and geometry.
                'bins': row['bins'],
                'image_size': row['image_size'],
                'physical_parent_group': row['physical_parent_group'],
                **{key: row[key] for key in ('allocation_axes', 'augmentation') if key in row},
                **{key: row[key] for key in ('target_parent_group', 'capture_session', 'foreground_source',
                                           'foreground_mask', 'foreground_sources', 'placement',
                                           'view_observation', 'generation_method') if key in row},
                **{key: row[key] for key in ('parent_video', 'mother_scene') if row.get(key)},
            })
    if len(records) != dataset['images']:
        raise ValueError('Training source index count differs from the complete export')
    result = {'schema_version': 1, 'model_key': TASK, 'selection_mode': 'manifest_only', 'records': records}
    if 'source_reuse' in dataset:
        result['source_reuse'] = dataset['source_reuse']
    return result


def write_output(stage, rows, summary, exclusions):
    for split, entries in rows.items():
        folder = stage / split
        (folder / 'images').mkdir(parents=True)
        (folder / 'labels').mkdir()
        for index, (source, text, record, *_) in enumerate(entries, 1):
            stem = f'{record["source_kind"]}_{index:06d}'
            record['image'] = f'images/{stem}{source.suffix.lower()}'
            record['label'] = f'labels/{stem}.txt'
            exported = folder / record['image']
            shutil.copyfile(source, exported)
            (folder / record['label']).write_text(text, encoding='utf8', newline='\n')
            with exported.open('rb') as stream:
                if file_digest(stream, 'sha256').hexdigest() != record['image_sha256']:
                    raise ValueError('Published image differs from verified pixels: ' + record['source_image'])
        write_json(folder / 'manifest.json', {
            'task': TASK, 'split': split, 'status': 'complete_dataset_split',
            'class_names': verified.CLASS_NAMES, 'summary': summary[split],
            'records': [row[2] for row in entries],
        })
    write_json(stage / 'data.yaml', {
        'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
        'names': verified.CLASS_NAMES,
    })
    train = Counter(summary['train']['by_source'])
    reuse = camera_parent_reuse(rows)
    write_json(stage / 'manifest.json', {
        'task': TASK, 'status': 'complete_dataset', 'class_names': verified.CLASS_NAMES,
        'images': sum(entry['images'] for entry in summary.values()),
        'train_source_budget': {'direct_real': train['direct_real'] + train['direct_empty_negative'],
                                'real_background_composite': train['real_background_composite'],
                                'blender': train['blender_route_composite'] + train['blender_empty_negative']},
        'splits': {split: {'manifest': f'{split}/manifest.json', **entry} for split, entry in summary.items()},
        'camera_exclusions': exclusions, 'source_reuse': reuse, 'split_scope': {
            'camera_parent_rule': 'declared reservations apply; cross-split camera parents are reported',
            'blender_scope': 'different admitted split renders share the Waste Bins Photo asset and workspace project',
            'evaluation_scope': ('same-camera-parent diagnostic and same-asset composited evaluation'
                                 if reuse else 'labeled real and same-asset composited evaluation'),
        },
    })


def install(stage, outputs, backup, installed, moved):
    for path in outputs:
        if path.exists():
            saved = backup / path.name
            moved.append((saved, path))
            path.rename(saved)
        source = stage / path.name
        installed.append((path, source))
        source.rename(path)


def publish(rows, exclusions, database, archive_previous):
    base = dataset_dir(TASK).resolve()
    outputs = [*(split_dir(TASK, split) for split in SPLIT_ORDER),
               *(base / name for name in ('data.yaml', 'manifest.json', 'workspace_sources.json'))]
    for path in outputs:
        if path.resolve() != path or path.parent != base:
            raise ValueError('Publication output resolves outside its declared directory: ' + str(path))
    if HISTORY.resolve() != HISTORY or not HISTORY.is_relative_to(ROOT):
        raise ValueError('Archive resolves outside its declared directory')
    config = ROOT / 'datasets/scripts' / TASK / 'config'
    procedural = {database, direct.TARGETS, ROOT / 'config/full_train_allocation.yaml',
                  config / 'source.yaml', config / 'dataset.yaml'}
    if (ROOT / RELOCATIONS).is_file():
        procedural.add(ROOT / RELOCATIONS)
    media = set()
    for entries in rows.values():
        for _, _, _, images, metadata in entries:
            media.update(images)
            procedural.update(metadata)
    code = (Path(__file__), Path(direct.__file__), Path(verified.__file__),
            ROOT / 'datasets/scripts/common/frame_environment.py',
            ROOT / 'datasets/scripts/common/source_history.py',
            ROOT / 'datasets/scripts/common/source_ledger/config.py',
            ROOT / 'datasets/scripts/common/source_ledger/reservations.py')
    params = {'task': TASK, 'splits': SPLIT_ORDER,
              'train_source_budget': dict(Counter(row[2]['source_kind'] for row in rows['train'])),
              'camera_exclusions': exclusions, 'archive_previous': archive_previous}
    installed, moved = [], []
    backup = HISTORY / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history_existed = HISTORY.exists()
    backup_created = False
    with TemporaryDirectory(dir=ROOT / 'tmp', prefix='waste_bin_full_export_') as scratch:
        stage = Path(scratch)
        try:
            with generation_guard(f'{TASK}.build_full_dataset', sorted(media), [*outputs, HISTORY], params,
                                  code_paths=code, procedural_inputs=sorted(procedural), root=ROOT,
                                  adopt_outputs=[*outputs, HISTORY] if archive_previous else ()) as run:
                if run.reused:
                    return verified.read_json(base / 'manifest.json')
                if any(path.exists() for path in outputs) and not archive_previous:
                    raise ValueError('Use --archive-previous to preserve and replace existing published outputs')
                summary = verify_and_summarize(rows)
                write_output(stage, rows, summary, exclusions)
                write_json(stage / 'workspace_sources.json', training_sources(stage, base))
                HISTORY.mkdir(exist_ok=True)
                backup.mkdir()
                backup_created = True
                install(stage, outputs, backup, installed, moved)
                write_json(backup / 'publication_backup.json', {
                    'task': TASK, 'original_root': relative(base),
                    'paths': [relative(path) for _, path in moved],
                })
                manifest = verified.read_json(base / 'manifest.json')
                run.set_result({'images': manifest['images'], 'splits': summary,
                                'previous_export_archive': relative(backup) if moved else None})
                return manifest
        except BaseException:
            for path, source in reversed(installed):
                if path.exists() and not source.exists():
                    path.rename(source)
            for saved, path in reversed(moved):
                if saved.exists() and not path.exists():
                    saved.rename(path)
            if backup_created:
                evidence = backup / 'publication_backup.json'
                evidence.unlink(missing_ok=True)
                backup.rmdir()
            if not history_existed and HISTORY.is_dir() and not any(HISTORY.iterdir()):
                HISTORY.rmdir()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-direct', action='store_true', help='Show eligible camera and Blender-empty counts')
    mode.add_argument('--check', action='store_true', help='Verify assembled inputs without writing an export')
    mode.add_argument('--publish', action='store_true', help='Publish all three verified splits')
    mode.add_argument('--sync-training-sources', action='store_true',
                      help='Rebuild only the training index from the current export and split manifests')
    parser.add_argument('--archive-previous', action='store_true', help='Preserve the prior export when replacing it')
    args = parser.parse_args()
    if args.archive_previous and not args.publish:
        parser.error('--archive-previous requires --publish')
    if args.sync_training_sources:
        base = dataset_dir(TASK).resolve()
        sources = training_sources(base, base)
        output = base / 'workspace_sources.json'
        with TemporaryDirectory(dir=base, prefix='.training_sources_') as scratch:
            staged = Path(scratch) / output.name
            write_json(staged, sources)
            if not output.is_file() or output.read_bytes() != staged.read_bytes():
                staged.replace(output)
        print(json.dumps({'path': relative(output), 'splits': dict(Counter(
            row['split'] for row in sources['records']))}, ensure_ascii=False))
        return
    _, source = load_task_config(TASK)
    database = direct.workspace_file(source['databases']['frame_environment'])
    if args.check_direct:
        history = read_source_history(ROOT)
        counts = {split: dict(Counter(row[2]['source_kind'] for row in direct_rows(split, history)))
                  for split in SPLIT_ORDER}
        counts['train']['blender_empty_negative_available'] = len(blender_empty_rows(database, history))
        print(json.dumps(counts, ensure_ascii=False))
        return
    rows, exclusions = prepare_dataset(database, source)
    if args.check:
        result = {'splits': verify_and_summarize(rows), 'camera_exclusions': exclusions,
                  'source_reuse': camera_parent_reuse(rows)}
    else:
        result = publish(rows, exclusions, database, args.archive_previous)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
