"""Package complete direct person records and index the current person splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from datasets.scripts.common import package_direct_real as direct
from datasets.scripts.common.paths import SPLITS, WORKSPACE as ROOT, dataset_dir, split_dir
from datasets.scripts.common.source_history import RELOCATIONS, capture_identity, historical_source, read_source_history
from datasets.scripts.common.source_ledger.config import build_groups, load_source_config, source_config_path
from datasets.scripts.common.source_ledger.reservations import matching_groups as _matching_groups, reservation_conflicts


TASK = 'person'
CLASSES = ('community', 'noncommunity')
CAR_REAL = 'car_real_capture_20261112'
COMPLETE_LABEL = 'complete_task_yolo'
DIRECT_DIR = 'direct_real'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def prepare_direct(split, record_ids):
    """Apply the explicit split plan to task-complete records in the shared catalog."""
    if split not in SPLITS or not isinstance(record_ids, (tuple, list)) or any(
            not isinstance(value, str) or not value.strip() for value in record_ids):
        raise ValueError('Expected a person split and a list of direct capture IDs')
    if len(set(record_ids)) != len(record_ids):
        raise ValueError('Direct capture IDs must be unique within the split')
    if not record_ids:
        return []
    selected, _ = direct.prepare(TASK, record_ids=record_ids)
    groups, history = _source_policy()
    for _, row, truth, *_ in selected:
        if not row.get('source_group'):
            raise ValueError(f'Direct person capture lacks a source group: {row["id"]}')
        if truth.get('split') not in (None, 'unassigned', split):
            raise ValueError(f'Direct person split conflicts with the source: {row["id"]}')
        _check_reservations(row, split, groups, history)
    by_id = {item[1]['id']: item for item in selected}
    return [by_id[record_id] for record_id in record_ids]


def direct_dependencies(records):
    """Identify original and working paths consumed within the split guard."""
    media, procedural = set(), set()
    code = (Path(__file__),)
    if records:
        code += (Path(direct.__file__), ROOT / 'lab/scripts/data/export_site_adaptation.py',
                 ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/source_history.py',
                 ROOT / 'datasets/scripts/common/background_pool.py',
                 ROOT / 'datasets/scripts/common/source_ledger/config.py',
                 ROOT / 'datasets/scripts/common/source_ledger/reservations.py',
                 ROOT / 'datasets/scripts/common/source_ledger/schema.py')
        procedural.add(source_config_path(ROOT, TASK))
        if any('frame_environment' in item[1] for item in records):
            code += (ROOT / 'datasets/scripts/common/frame_environment.py',)
        procedural.add(direct.TARGETS)
        if (ROOT / RELOCATIONS).is_file():
            procedural.add(ROOT / RELOCATIONS)
    for source, row, _, image, parent, label, _ in records:
        media.add(image)
        procedural.add(source)
        if parent is not None:
            media.add(parent)
        if label is not None:
            procedural.add(label)
        if row.get('canonical_manifest'):
            media.add(direct.workspace_file(row['source_image']))
            procedural.add(direct.workspace_file(row['canonical_manifest']))
    return media, procedural, code


def write_direct_records(output, split, records):
    """Copy selected pixels and write the original-coordinate label under the active split guard."""
    if not records:
        return []
    # Verify both the selection and generated labels after the caller admits inputs to its guard.
    if prepare_direct(split, [item[1]['id'] for item in records]) != records:
        raise ValueError('Direct person evidence changed after selection')
    image_dir = output / 'images' / DIRECT_DIR
    label_dir = output / 'labels' / DIRECT_DIR
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    history = read_source_history(ROOT)
    entries = []
    for index, (source, row, truth, image, parent, label, body) in enumerate(records, 1):
        name = f'person_real_{index:06d}'
        output_image = image_dir / (name + image.suffix.lower())
        output_label = label_dir / (name + '.txt')
        shutil.copyfile(image, output_image)
        output_label.write_text(body, encoding='utf-8', newline='\n')
        source_name = source.relative_to(ROOT).as_posix()
        point_still = row['id'].endswith(':' + row['waypoint_label']) if row.get('waypoint_label') else False
        from_video = parent is not None and not point_still
        source_kind = row.get('source_kind') or ('video_frame' if from_video else 'photo')
        method = row.get('generation_method') or ('provided_video_frame' if from_video else 'camera_capture')
        entries.append({
            'type': 'direct_real', 'id': row['id'], 'split': split,
            'image': output_image.relative_to(ROOT).as_posix(),
            'label': output_label.relative_to(ROOT).as_posix(),
            'label_status': COMPLETE_LABEL, 'annotation': truth,
            'source_split': truth.get('split'),
            'split_basis': ('confirmed_source_split' if truth.get('split') == split else 'explicit_person_split_plan'),
            'source_group': row['source_group'], 'source_image': row['image'],
            'source_label': label.relative_to(ROOT).as_posix() if label else None,
            'source_manifest': source_name, 'original_source_image': row.get('source_image') or row['image'],
            'original_source_manifest': row.get('canonical_manifest') or source_name,
            'original_path': row.get('original_path'), 'capture_identity': capture_identity(row, history),
            'parent_video': parent.relative_to(ROOT).as_posix() if parent else None,
            'point': row.get('point'), 'segment_id': row.get('segment'),
            'frame_index': row.get('frame_index'), 'timestamp_seconds': row.get('timestamp_seconds'),
            'near_duplicate_group': row.get('near_duplicate_group'), 'image_size': row['image_size'],
            'source_kind': source_kind, 'generation_method': method,
            'image_processing': row.get('image_processing', {}),
            **({'frame_environment': row['frame_environment']} if 'frame_environment' in row else {}),
            'evaluation_scope': ('known_route_diagnostic' if _car_real_source(row['source_group'],
                                 row.get('parent_video')) else None),
        })
    return entries


def _label_rows(record):
    path = direct.workspace_file(record['label'])
    text = path.read_text(encoding='utf-8')
    direct.check_yolo(text, len(CLASSES), 0)
    return [line.split() for line in text.splitlines()]


def split_coverage(records):
    """Count actual exported labels, states and observed white backs."""
    by_type, counts, classes, white_backs, groups, failure_events = (Counter() for _ in range(6))
    class_by_type = {'synthetic': Counter(), 'direct_real': Counter()}
    visible_below_canvas, visible_below_effective = 0, 0
    for record in records:
        rows = _label_rows(record)
        kind = 'direct_real' if record.get('type') == 'direct_real' else 'synthetic'
        by_type[kind] += 1
        counts[len(rows)] += 1
        image_classes = [CLASSES[int(row[0])] for row in rows]
        classes.update(image_classes)
        class_by_type[kind].update(image_classes)
        canvas_height = (record['image_size'] if kind == 'direct_real' else record['parameters']['resolution'])[1]
        effective_height = record.get('effective_image_height', canvas_height)
        if not 0 < effective_height <= canvas_height:
            raise ValueError(f'Invalid effective height: {record["image"]}')
        visible_below_canvas += sum(float(row[4]) < .28 for row in rows)
        visible_below_effective += sum(float(row[4]) * canvas_height / effective_height < .28 for row in rows)
        if event := record.get('failure_event'):
            failure_events[event] += 1
        if group := record.get('source_group'):
            groups[group] += 1
        annotation = record.get('annotation', {})
        direct_observations = annotation.get('evidence', {}).get('background_observations', [])
        if (record.get('background_white_backs') or annotation.get('white_back_reference_boxes') or any(
                obj.get('kind') == 'person' and obj.get('view') == 'back' and obj.get('visible_bbox_xyxy')
                for obj in record.get('objects', [])) or any(
                    obj.get('observation') in ('white_back', 'back_silhouette') for obj in direct_observations)):
            white_backs['with_front_boxes' if rows else 'without_front_boxes'] += 1
    return {'images': sum(by_type.values()), 'by_type': dict(by_type), 'yolo_boxes': sum(classes.values()),
            'boxes_by_class': {cls: classes[cls] for cls in CLASSES}, 'images_by_box_count': dict(sorted(counts.items())),
            'boxes_by_type_and_class': {kind: {cls: values[cls] for cls in CLASSES}
                                        for kind, values in class_by_type.items()},
            'visible_boxes_below_0_28_canvas_height': visible_below_canvas,
            'visible_boxes_below_0_28_effective_height': visible_below_effective,
            'images_with_structured_white_back_evidence': dict(white_backs),
            'source_groups': dict(groups), 'failure_events': dict(failure_events)}


def _car_real_source(group, parent):
    return any(CAR_REAL in Path(value).parts for value in (group, parent) if value)


def _source_policy():
    source = load_source_config(ROOT, TASK)
    return build_groups(TASK, source.get('pairing') or {}), read_source_history(ROOT)


def _check_reservations(row, split, groups, history):
    matched = _matching_groups(row, groups, history)
    conflicts = reservation_conflicts(matched, split)
    if conflicts:
        group = conflicts[0]
        raise ValueError(f'Person source {group.group_id} is reserved for {group.split.get("primary")}, not {split}')
    return matched


def _reuse_declarations(left, right, groups, history):
    """Return matching groups explicitly allowing both splits; no inferred authorization."""
    a = {group.group_id: group for group in _matching_groups(left, groups, history)}
    b = {group.group_id: group for group in _matching_groups(right, groups, history)}
    accepted = []
    for key in sorted(set(a) & set(b)):
        group = a[key]
        if group.split.get('primary') == 'shared' and group.split.get('status') in ('reserved', 'confirmed'):
            accepted.append(key)
            continue
        allowed = {group.split.get('primary'), *(group.split.get('same_source_allowed_in') or [])}
        if {left['split'], right['split']}.issubset(allowed) and group.split.get('same_source_allowed_in'):
            accepted.append(key)
    return accepted


def _image_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _training_record(record, split, manifest_path):
    labels = _label_rows(record)
    state = 'complete_positive' if labels else 'complete_negative'
    if record.get('label_status') != COMPLETE_LABEL:
        raise ValueError(f'Incomplete person supervision in {split}: {record["image"]}')
    is_direct = record.get('type') == 'direct_real'
    background = record.get('background', {})
    if is_direct:
        if record.get('split') != split or state != record['annotation']['state']:
            raise ValueError(f'Direct person annotation or split mismatch: {record["image"]}')
        source_evidence = record['original_source_manifest']
        parent = record.get('parent_video')
        source_kind = record['source_kind']
        method = record['generation_method']
        processing = {**record.get('image_processing', {}), 'packaging_resampling': 'none',
                      'packaging_input_size': record['image_size']}
    else:
        source_evidence = record.get('background_evidence') or background.get('manifest') or background['review_manifest']
        parent = record.get('parent_video') or background.get('source_video')
        method = record['generation_method']
        source_kind, processing = 'derived_image', {}
        if method == 'real_background_composite':
            method = 'composite'
            processing = {'background_origin': 'direct_camera', 'origin_review': 'confirmed'}
    group = record['source_group']
    if not isinstance(group, str) or not group.strip():
        raise ValueError(f'Invalid person source group: {record["image"]}')
    direct.workspace_file(source_evidence)
    return {
        'image': record['image'], 'label': record['label'], 'split': split,
        'source_group': group, 'source_id': group, 'parent_video': parent,
        'source_kind': source_kind, 'generation_method': method, 'image_processing': processing,
        'workspace_image': record.get('source_image', record['image']),
        'workspace_label': record.get('source_label') or record['label'],
        'label_status': 'reviewed', 'review_state': state, 'source_review_evidence': source_evidence,
        'review_evidence': manifest_path.relative_to(ROOT).as_posix(),
        'source_assets': record.get('source_assets', []), 'mother_scene': record.get('mother_scene'),
        'capture_identity': record.get('capture_identity'),
        **({'frame_environment': record['frame_environment']} if is_direct and 'frame_environment' in record else {}),
        'near_duplicate_group': record.get('near_duplicate_group'), 'fragment': record.get('segment_id'),
        'evaluation_scope': record.get('evaluation_scope'),
    }


def _inspect_split(split):
    folder = split_dir(TASK, split)
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = manifest['records']
    if not records:
        raise ValueError(f'Person split has no records: {split}')
    image_dir, label_dir = folder / 'images', folder / 'labels'
    images = {path.relative_to(ROOT).as_posix() for path in image_dir.rglob('*')
              if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    labels = {path.relative_to(ROOT).as_posix() for path in label_dir.rglob('*.txt') if path.is_file()}
    declared_images = {row['image'] for row in records}
    declared_labels = {row['label'] for row in records}
    if (len(declared_images) != len(records) or len(declared_labels) != len(records) or
            not declared_images.issubset(images) or not declared_labels.issubset(labels)):
        raise ValueError(f'Person split manifest does not match the exported pairs: {split}')
    for row in records:
        path = Path(row['image']).relative_to(image_dir.relative_to(ROOT))
        expected = (label_dir.relative_to(ROOT) / path.with_suffix('.txt')).as_posix()
        if row['label'] != expected:
            raise ValueError(f'Exported label does not pair with its image: {row["image"]}')
    coverage = split_coverage(records)
    coverage['unowned_images_excluded'] = sorted(images - declared_images)
    coverage['unowned_labels_excluded'] = sorted(labels - declared_labels)
    return [_training_record(row, split, manifest_path) for row in records], coverage


def finalize(splits):
    """Produce the training data specification and evidence index from completed split manifests."""
    if 'train' not in splits or set(splits) - SPLITS or len(set(splits)) != len(splits):
        raise ValueError('Finalize requires train and can include the current val and test splits')
    records, coverage, captures = [], {}, set()
    history = read_source_history(ROOT)
    groups, _ = _source_policy()
    image_hashes, reuse, source_index = {}, {}, {}
    for split in splits:
        rows, coverage[split] = _inspect_split(split)
        for row in rows:
            _check_reservations(row, split, groups, history)
            if identity := row.get('capture_identity'):
                if identity in captures:
                    raise ValueError(f'Direct person capture is repeated in the export: {identity}')
                captures.add(identity)
            digest = _image_sha256(direct.workspace_file(row['image']))
            prior = image_hashes.setdefault(digest, row)
            if prior['split'] != split:
                raise ValueError(f'Identical person output images cross splits: {prior["image"]} and {row["image"]}')
            for kind in ('source_group', 'parent_video', 'mother_scene'):
                value = row.get(kind)
                if isinstance(value, str) and value:
                    key = historical_source(value.replace('\\', '/').rstrip('/'), history)
                    source_index.setdefault((kind, key), {}).setdefault(split, []).append(row)
        records.extend(rows)
    for (kind, value), by_split in source_index.items():
        splits_key = tuple(sorted(by_split))
        if len(splits_key) < 2:
            continue
        declared = set()
        for index, left_split in enumerate(splits_key):
            for right_split in splits_key[index + 1:]:
                declared.update(_reuse_declarations(by_split[left_split][0], by_split[right_split][0],
                                                    groups, history))
        reuse[(kind, value)] = {kind: value, 'splits': list(splits_key), 'groups': sorted(declared),
                                'evaluation_scope': 'same_source_diagnostic'}
        for split, involved_rows in by_split.items():
            if split != 'train':
                for row in involved_rows:
                    row['evaluation_scope'] = 'same_source_diagnostic'
    spec = {'names': list(CLASSES), **{split: f'{split}/images' for split in splits}}
    sources = {'model_key': TASK, 'selection_mode': 'manifest_only', 'records': records, 'coverage': coverage,
               'source_reuse': list(reuse.values())}
    output = dataset_dir(TASK)
    with TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        # JSON is accepted by the YAML loader and keeps the metadata writer dependency-free.
        for name, value in (('data.yaml', spec), ('workspace_sources.json', sources)):
            (stage / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for name in ('workspace_sources.json', 'data.yaml'):
            (stage / name).replace(output / name)
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', action='append', choices=sorted(SPLITS), required=True,
                        help='index this completed split (repeat for each current split)')
    args = parser.parse_args()
    print(json.dumps(finalize(args.split), ensure_ascii=False))


if __name__ == '__main__':
    main()
