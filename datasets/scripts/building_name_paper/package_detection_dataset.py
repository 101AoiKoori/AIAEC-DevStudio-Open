"""Publish the reviewed paper detection splits and their V4 source manifest."""
import json
import hashlib
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml

from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.paths import dataset_dir, split_dir
from datasets.scripts.common.source_history import RELOCATIONS, historical_source, read_source_history
from datasets.scripts.common.source_ledger.config import build_groups
from lab.training_common.production_guard import generation_guard

TASK = 'building_name_paper'


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def reviewed_record(image, label, split, count, group, manifest, **lineage):
    return {
        'image': relative(image), 'label': relative(label), 'split': split,
        'instances': count, 'source_id': group, 'label_status': 'reviewed',
        'review_state': 'complete_positive' if count else 'complete_negative',
        'review_evidence': relative(manifest), 'source_review_evidence': relative(manifest),
        **lineage,
    }


def train_entries():
    base = split_dir(TASK, 'train')
    manifests = [base / 'manifest.json']
    blender = base / 'blender/manifest.json'
    if blender.exists():
        manifests.append(blender)
    entries = []
    for manifest in manifests:
        batch = read_json(manifest)
        rendered = manifest.parent != base
        if (rendered and batch.get('label_scope') != 'paper_detection' or
                not rendered and batch.get('status') != 'complete_detection_records'):
            raise ValueError(f'Incomplete paper detection batch: {manifest}')
        for row in batch['records']:
            image, label = ROOT / row['image'], ROOT / row['label']
            images, labels = base / 'images', base / 'labels'
            if image.resolve() != image or label != (labels / image.relative_to(images)).with_suffix('.txt'):
                raise ValueError(f'Train paths do not pair: {image}, {label}')
            if rendered:
                expected = [yolo_row(p['bbox_xyxy_normalized'], 1, 1) for p in row['objects']]
                origin = {'mother_scene': row['source_scene'], 'source_kind': 'blender_scene',
                          'workspace_image': row['image']}
                group = row['source_scene']
            else:
                expected = [yolo_row(p['box_xyxy'], *row['image_size']) for p in row['papers']]
                origin = {'parent_video': row['parent_source'], 'workspace_image': row['source_image'],
                          'source_review': row['source_review']}
                group = row['parent_source']
            if label.read_text(encoding='utf8').splitlines() != expected:
                raise ValueError(f'Train label differs from completed batch: {label}')
            entries.append((image, label, reviewed_record(
                image, label, 'train', len(expected), group, manifest,
                generation_method=row['generation_method'], **origin)))
    return entries


def evaluation_entries(plan, database):
    entries = []
    for split in ('val', 'test'):
        for role in ('direct_target', 'empty_negative'):
            eligible = {
                row['record_id']: row for row in query_frames(task=TASK, role=role, database=database)
                if row['split'] in (split, 'unassigned') and not row.get('duplicate_of')
            }
            selection = plan[split].get(role)
            if selection is None:
                parent = plan[split][role + '_parent']
                selection = sorted(key for key, row in eligible.items()
                                   if row['split'] == split and row['parent_group'] == parent)
            if not selection or len(selection) != len(set(selection)) or set(selection) - eligible.keys():
                raise ValueError(f'Invalid or stale {split}/{role} selection')
            for key in selection:
                row = eligible[key]
                annotation = row['label']
                instances = annotation['instances'] or []
                if (annotation['completeness'] != 'complete' or row['scope']['kind'] != 'frame' or
                        bool(instances) != (role == 'direct_target') or
                        any(p['class_id'] != 0 for p in instances)):
                    raise ValueError(f'Incomplete full-frame detection: {key}')
                source = row.get('working_path') or next(
                    (path for path in row['paths'] if path.startswith(f'datasets/raw/{TASK}/')), row['image'])
                if image_digest(source) != row['content_sha256']:
                    raise ValueError(f'Frame identity changed: {source}')
                source = ROOT / source
                base = split_dir(TASK, split)
                stem = f'real_{row["frame_id"][:16]}'
                image, label = base / 'images' / (stem + source.suffix.lower()), base / 'labels' / (stem + '.txt')
                boxes = [yolo_row(p.get('xyxy') or p['box_xyxy'], *row['image_size']) for p in instances]
                # Archive photos in the same building folder share a conservative split group.
                group = row.get('parent_video') or Path(row['parent_group']).parent.as_posix()
                lineage = {
                    'workspace_image': relative(source), 'source_kind': row['source_kind'],
                    'generation_method': row['generation_method'], 'parent_video': row.get('parent_video'),
                    'source_parent': row['parent_group'], 'image_sha256': row['content_sha256'],
                    'source_review': {k: row[k] for k in ('frame_id', 'record_id', 'split', 'role', 'evidence')},
                }
                entries.append((source, image, label, boxes, reviewed_record(
                    image, label, split, len(boxes), group, base / 'manifest.json', **lineage)))
    return entries


def source_policy(records, images, source_config, history):
    """Check explicit reservations and record actual same-source reuse.

    ``images`` maps each output path to the file that supplies its exact bytes;
    evaluation files have not yet been copied into their final destinations.
    """
    groups = build_groups(TASK, source_config.get('pairing') or {})
    image_hashes, source_index = {}, {}
    for row in records:
        split = row['split']
        values = [row.get(key) for key in ('source_id', 'source_parent', 'parent_video',
                                           'mother_scene', 'workspace_image')]
        values = {historical_source(value.replace('\\', '/').rstrip('/'), history)
                  for value in values if isinstance(value, str) and value}
        for group in groups:
            paths = {historical_source(path.replace('\\', '/').rstrip('/'), history)
                     for path in group.paths}
            matched = group.group_id in values or any(value == path or value.startswith(path + '/')
                                                      for value in values for path in paths)
            rule = group.split
            primary = rule.get('primary')
            if (matched and (rule.get('status') == 'reserved' or rule.get('action') == 'reserved')
                    and primary in ('train', 'val', 'test') and primary != split
                    and split not in (rule.get('same_source_allowed_in') or [])):
                raise ValueError(f'Paper source {group.group_id} is reserved for {primary}, not {split}')

        digest = hashlib.sha256()
        with images[row['image']].open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        prior = image_hashes.setdefault(digest.hexdigest(), row)
        if prior['split'] != split:
            raise ValueError(f'Identical paper output images cross splits: {prior["image"]} and {row["image"]}')
        for kind in ('source_id', 'parent_video', 'mother_scene'):
            value = row.get(kind)
            if isinstance(value, str) and value:
                key = historical_source(value.replace('\\', '/').rstrip('/'), history)
                source_index.setdefault((kind, key), {}).setdefault(split, []).append(row)

    reuse = []
    for (kind, value), by_split in source_index.items():
        if len(by_split) < 2:
            continue
        relation = 'source_group' if kind == 'source_id' else kind
        reuse.append({kind: value, 'relation': relation, 'source': value,
                      'splits': sorted(by_split), 'evaluation_scope': 'same_source_diagnostic'})
        for split, rows in by_split.items():
            if split != 'train':
                for row in rows:
                    row['evaluation_scope'] = 'same_source_diagnostic'
    return reuse


def main():
    _, source_config = load_task_config(TASK)
    base = dataset_dir(TASK)
    plan_path = source_path(TASK, 'detection_evaluation')
    database = ROOT / source_config['databases']['frame_environment']
    plan = yaml.safe_load(plan_path.read_text(encoding='utf8'))
    historical = train_entries()
    evaluation = evaluation_entries(plan, database)
    records = [row[-1] for row in historical + evaluation]
    images = {row['image']: image for image, _, row in historical}
    images.update({row['image']: source for source, _, _, _, row in evaluation})
    if len(images) != len(records):
        raise ValueError('Repeated output path in paper detection manifests')
    reuse = source_policy(records, images, source_config, read_source_history(ROOT))
    data, sources = base / 'data.yaml', base / 'workspace_sources.json'
    folders = [split_dir(TASK, split) for split in ('val', 'test')]
    if any(path.resolve() != path for path in (*folders, data, sources)):
        raise ValueError('Detection output resolves outside its declared path')
    cfg = plan_path.parent
    inputs = {plan_path, cfg / 'dataset.yaml', cfg / 'source.yaml',
              *(label for _, label, _ in historical),
              *(ROOT / row['review_evidence'] for _, _, row in historical)}
    if (ROOT / RELOCATIONS).is_file():
        inputs.add(ROOT / RELOCATIONS)
    # The immutable revisions read from SQLite are fingerprinted in the receipt.
    selection = [row[-1]['source_review'] for row in evaluation]
    media = [image for image, _, _ in historical] + [source for source, *_ in evaluation]
    created = []
    created_folders = []
    try:
        with generation_guard('building_name.package_detection_dataset', media, [*folders, data, sources],
                              {'selection': selection, 'plan': relative(plan_path)}, root=ROOT,
                              procedural_inputs=sorted(inputs),
                              code_paths=[Path(__file__), ROOT / 'datasets/scripts/common/annotations.py',
                                          ROOT / 'datasets/scripts/common/frame_environment.py',
                                          ROOT / 'datasets/scripts/common/config.py',
                                          ROOT / 'datasets/scripts/common/source_history.py',
                                          ROOT / 'datasets/scripts/common/background_pool.py',
                                          ROOT / 'datasets/scripts/common/source_ledger/config.py',
                                          ROOT / 'datasets/scripts/common/source_ledger/schema.py']) as run:
            if not run.reused:
                if data.exists() or sources.exists() or any(
                        folder.exists() and any(folder.iterdir()) for folder in folders):
                    raise ValueError('Detection publication requires empty evaluation destinations')
                for folder in folders:
                    for path in (folder, folder / 'images', folder / 'labels'):
                        if not path.is_dir():
                            path.mkdir()
                            created_folders.append(path)
                for source, image, label, boxes, row in evaluation:
                    created.extend((image, label))
                    shutil.copyfile(source, image)
                    if image_digest(image) != row['image_sha256']:
                        raise ValueError(f'Copied image differs from source: {source}')
                    label.write_text(''.join(box + '\n' for box in boxes), encoding='utf8')
                for folder in folders:
                    manifest = folder / 'manifest.json'
                    created.append(manifest)
                    write_json(manifest, {'task': TASK, 'split': folder.name, 'label_scope': 'paper_detection',
                                          'records': [r for r in records if r['split'] == folder.name]})
                created.extend((sources, data))
                write_json(sources, {'model_key': TASK, 'selection_mode': 'manifest_only',
                                     'records': records, 'source_reuse': reuse,
                                     'exposure_warnings': ['Archive sessions are unverified; the same physical '
                                                           'building designs recur in development.']})
                write_json(data, {'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
                                  'names': ['paper']})
                run.set_result({'by_split': dict(Counter(row['split'] for row in records))})
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for path in reversed(created_folders):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        raise
    for split in ('train', 'val', 'test'):
        counts = Counter(row['instances'] for row in records if row['split'] == split)
        print(f'{split}: {sum(counts.values())} images, {counts[0]} negatives, {sum(k * n for k, n in counts.items())} boxes')


if __name__ == '__main__':
    main()
