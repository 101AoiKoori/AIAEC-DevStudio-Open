"""Inspect or package explicitly complete direct camera labels for a V4 task."""

from __future__ import annotations

import argparse
from collections import Counter
import filecmp
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.common.config import load_task_config, read_yaml
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.paths import SPLITS, TASKS, sample_dir, split_dir
from datasets.scripts.common.source_history import RELOCATIONS, capture_identity, read_source_history
from lab.scripts.data.export_site_adaptation import label_lines
from lab.training_common.production_guard import generation_guard


TARGETS = ROOT / 'config/task_targets.yaml'


def workspace_file(value):
    relative = Path(value)
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Expected an existing workspace-relative file: {value}')
    return path


def catalog(name):
    manifest = workspace_file(name)
    data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if data.get('schema_version') != 1 or not isinstance(data.get('records'), list):
        raise ValueError(f'Unsupported source catalog: {name}')
    return manifest, data['records']


def read_records(task, *, history=None):
    if history is None:
        history = read_source_history(ROOT)
    _, source = load_task_config(task)
    aliases = {}
    if source.get('target_working_manifest'):
        working_manifest, rows = catalog(source['target_working_manifest'])
        for row in rows:
            if (task not in row['tasks'] or not row.get('source_image') or
                    row.get('source_manifest') not in source['direct_capture_manifests']):
                continue
            if not Path(row['image']).is_relative_to(Path('datasets/raw') / task / 'targets/camera'):
                raise ValueError(f'Target working capture is outside its task camera directory: {row["image"]}')
            key = (row['id'], row['source_image'])
            if key in aliases:
                raise ValueError(f'Repeated target working record: {key!r}')
            aliases[key] = row
    for name in source['direct_capture_manifests']:
        manifest, rows = catalog(name)
        for row in rows:
            if task in row['tasks']:
                truth = row['tasks'][task]
                alias = aliases.get((row['id'], row['image']))
                if alias is not None:
                    identity = capture_identity(row, history)
                    if (alias['tasks'][task] != truth or alias.get('capture_identity') != identity
                            or any(alias.get(key) != row.get(key) for key in (
                                'duplicate_of', 'source_group', 'parent_video', 'image_size'))):
                        raise ValueError(f'Target working record differs from its source: {row["id"]}')
                    yield working_manifest, {**row, 'image': alias['image'], 'source_image': row['image'],
                                             'capture_identity': identity,
                                             'canonical_manifest': manifest.relative_to(ROOT).as_posix()}, truth
                else:
                    yield manifest, row, truth


def check_yolo(text, class_count, keypoints):
    """Validate stored YOLO labels against the task's detection or pose shape."""
    for line in text.splitlines():
        cls, *values = line.split()
        numbers = [float(value) for value in values]
        if (not 0 <= int(cls) < class_count or len(numbers) != 4 + keypoints * 3
                or not all(math.isfinite(x) and 0 <= x <= (2 if i >= 4 and i % 3 == 0 else 1)
                           for i, x in enumerate(numbers))):
            raise ValueError('Invalid YOLO class, coordinate or label shape')
        x, y, w, h = numbers[:4]
        if (min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-6
                or max(x+w/2, y+h/2) > 1 + 1e-6):
            raise ValueError('YOLO box outside image')
        for px, py, visible in zip(numbers[4::3], numbers[5::3], numbers[6::3]):
            if visible not in (0, 1, 2) or (visible == 0 and (px or py)):
                raise ValueError('Invalid YOLO keypoint visibility')


def reviewed_direct_records(task, source):
    """Index current, eligible direct supervision by every registered image path."""
    database = workspace_file(source['databases']['frame_environment'])
    by_path = {}
    for role in ('direct_target', 'empty_negative'):
        for record in query_frames(task=task, role=role, database=database):
            label = record.get('label') or {}
            if (label.get('scope', {}).get('kind') != 'frame' or
                    label.get('completeness') != 'complete' or
                    label.get('presence') != ('present' if role == 'direct_target' else 'absent')):
                continue
            for path in record['paths']:
                by_path.setdefault(path, []).append(record)
    return by_path


def database_only_records(task, source, reviewed):
    """Expose complete camera frames that have no row in an older source catalog.

    The environment record is the qualification authority.  A historical
    manifest, when present, remains provenance rather than an enumeration
    gate for newly reviewed frames.
    """
    source_config = ROOT / 'datasets/scripts' / task / 'config/source.yaml'
    seen = set()
    camera_kinds = {'camera', 'camera_photo', 'video_frame',
                    'archive_camera_photo', 'archive_camera_video_frame'}
    for records in reviewed.values():
        for record in records:
            if record['record_id'] in seen or record.get('source_kind') not in camera_kinds:
                continue
            seen.add(record['record_id'])
            paths = record['paths']
            task_camera = f'datasets/raw/{task}/targets/camera/'
            shared_camera = 'datasets/raw/shared/backgrounds/camera/'
            image_path = next((path for prefix in (task_camera, shared_camera)
                               for path in paths if path.startswith(prefix) and (ROOT / path).is_file()),
                              record['image'])
            if not (ROOT / image_path).is_file():
                continue
            provenance = record.get('provenance') or {}
            manifest_name = provenance.get('manifest')
            manifest = (ROOT / manifest_name) if manifest_name else source_config
            if not manifest.is_file():
                manifest = source_config
            row = {
                'id': f"db_{record['frame_id'][:16]}_{record['role']}_{record['split']}",
                'image': image_path,
                'image_size': record['image_size'],
                'capture_identity': record.get('capture_identity') or record['frame_id'],
                'parent_video': record.get('parent_video'),
                'source_group': record.get('parent_group'),
                'duplicate_of': record.get('duplicate_of'),
                'source_kind': record.get('source_kind'),
                'generation_method': record.get('generation_method'),
            }
            label = record['label']
            state = 'complete_positive' if label['presence'] == 'present' else 'complete_negative'
            yield manifest, row, {**label, 'state': state, 'split': record['split']}


def prepare(task, *, split=None, record_ids=(), frame_record_ids=()):
    """Validate eligible images and labels in the source's original coordinates."""
    targets = read_yaml(TARGETS)['tasks'][task]
    class_count = len(targets['classes'])
    keypoints = targets.get('keypoints', 0)
    candidates = {}
    states = Counter()
    history = read_source_history(ROOT)
    _, source = load_task_config(task)
    reviewed = reviewed_direct_records(task, source)
    seen_frames = set()
    rows = list(read_records(task, history=history))
    rows.extend(database_only_records(task, source, reviewed))
    for manifest, row, truth in rows:
        if record_ids and row['id'] not in record_ids:
            continue
        decisions = reviewed.get(row['image']) or reviewed.get(row.get('source_image'), [])
        decisions = [item for item in decisions if split is None or item['split'] == split]
        if frame_record_ids:
            decisions = [item for item in decisions if item['record_id'] in frame_record_ids]
        if not decisions:
            continue
        if len(decisions) > 1:
            declared = truth.get('split')
            assigned = [item for item in decisions if item['split'] in SPLITS]
            preferred = [item for item in assigned if item['split'] == declared] if declared in SPLITS else assigned
            if len(preferred) == 1:
                decisions = preferred
        if len(decisions) != 1:
            raise ValueError(f'A direct frame has conflicting current role decisions: {row["image"]}')
        decision = decisions[0]
        if decision['frame_id'] in seen_frames:
            continue
        seen_frames.add(decision['frame_id'])
        label = decision['label']
        state = 'complete_positive' if label['presence'] == 'present' else 'complete_negative'
        states[state] += 1
        truth = {**label, 'state': state, 'split': decision['split']}
        row = {**row, 'capture_identity': decision.get('capture_identity') or capture_identity(row, history),
               'frame_environment': {
                   'frame_id': decision['frame_id'], 'record_id': decision['record_id'],
                   'role': decision['role'], 'split': decision['split'],
                   'evidence_ids': [item['evidence_id'] for item in decision['evidence']],
                   'content_sha256': decision['content_sha256'],
               }}
        image = workspace_file(row['image'])
        identity = row['capture_identity']
        prior = candidates.get(identity)
        if prior is not None and (record_ids or frame_record_ids or not (prior[1].get('duplicate_of') or row.get('duplicate_of'))):
            raise ValueError(f'Duplicate selected camera capture: {row["image"]}')
        if prior is None or not row.get('duplicate_of'):
            candidates[identity] = (manifest, row, truth, image)
    selected = []
    for manifest, row, truth, image in candidates.values():
        state = truth['state']
        if image_digest(row['image']) != row['frame_environment']['content_sha256']:
            raise ValueError(f'Direct frame pixels changed after review: {row["image"]}')
        if row.get('canonical_manifest') and not filecmp.cmp(image, workspace_file(row['source_image']), shallow=False):
            raise ValueError(f'Target working pixels differ from source: {row["image"]}')
        with Image.open(image) as pixels:
            width, height = pixels.size
            pixels.load()
        if [width, height] != row['image_size']:
            raise ValueError(f'Annotation dimensions differ from source: {row["image"]}')
        label = workspace_file(truth['label']) if truth.get('label') else None
        try:
            text = label.read_text(encoding='utf-8-sig') if label else label_lines(
                truth['instances'], width, height, class_count, keypoints)
            text = ''.join(line.strip() + '\n' for line in text.splitlines() if line.strip())
            check_yolo(text, class_count, keypoints)
            if bool(text) != (state == 'complete_positive'):
                raise ValueError('Label content differs from its positive/negative state')
        except (KeyError, ValueError) as error:
            raise ValueError(f'{row["image"]}: {error}') from error
        parent = workspace_file(row['parent_video']) if row.get('parent_video') else None
        selected.append((manifest, row, truth, image, parent, label, text))
    if set(record_ids) - {item[1]['id'] for item in selected}:
        raise ValueError('Requested record has no complete direct label for this task and split')
    if set(frame_record_ids) - {item[1]['frame_environment']['record_id'] for item in selected}:
        raise ValueError('Requested frame record has no complete direct label for this task and split')
    return selected, states


def package(task, split, selected, output):
    if not selected:
        raise ValueError('No eligible direct capture labels for this task and split')
    output = output.resolve()
    allowed = {sample_dir(task).resolve()}
    if split is not None:
        allowed.add(split_dir(task, split).resolve())
    if output not in allowed:
        raise ValueError('Use the current task sample directory or its explicitly selected split')
    configs = [ROOT / 'datasets/scripts' / task / 'config' / name
               for name in ('source.yaml', 'dataset.yaml')]
    if (ROOT / RELOCATIONS).is_file():
        configs.append(ROOT / RELOCATIONS)
    inputs = sorted({path for item in selected for path in item[3:5] if path is not None})
    metadata = sorted({TARGETS, *configs, *(item[0] for item in selected),
                       *(item[5] for item in selected if item[5] is not None),
                       *(workspace_file(item[1]['canonical_manifest']) for item in selected
                         if item[1].get('canonical_manifest'))})
    inputs += sorted({workspace_file(item[1]['source_image']) for item in selected
                      if item[1].get('canonical_manifest')})
    code = [Path(__file__), ROOT / 'datasets/scripts/common/frame_environment.py',
            ROOT / 'lab/scripts/data/export_site_adaptation.py',
            ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py',
            ROOT / 'datasets/scripts/common/source_history.py']
    params = {'task': task, 'split': split, 'record_ids': [item[1]['id'] for item in selected],
              'frame_environment_record_ids': [item[1]['frame_environment']['record_id'] for item in selected]}
    current, _ = prepare(task, split=split, record_ids=params['record_ids'],
                         frame_record_ids=params['frame_environment_record_ids'])
    if current != selected:
        raise ValueError('Direct capture evidence changed after selection')
    with generation_guard(f'{task}.package_direct_real:{output.relative_to(ROOT).as_posix()}',
                          inputs, [output], params, code_paths=code,
                          procedural_inputs=metadata, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output already contains results')
        current, _ = prepare(task, split=split, record_ids=params['record_ids'],
                             frame_record_ids=params['frame_environment_record_ids'])
        if current != selected:
            raise ValueError('Direct capture evidence changed after selection')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as workdir:
            stage = Path(workdir) / 'batch'
            (stage / 'images').mkdir(parents=True)
            (stage / 'labels').mkdir()
            records = []
            for index, (source, row, truth, image, _, _, text) in enumerate(selected, 1):
                image_name = f'real_{index:06d}{image.suffix.lower()}'
                label_name = f'real_{index:06d}.txt'
                shutil.copyfile(image, stage / 'images' / image_name)
                (stage / 'labels' / label_name).write_text(text, encoding='utf-8')
                records.append({'id': row['id'], 'image': f'images/{image_name}', 'label': f'labels/{label_name}',
                                'source_image': row['image'], 'parent_video': row.get('parent_video'),
                                'original_source_image': row.get('source_image') if row.get('canonical_manifest') else None,
                                'original_source_manifest': row.get('canonical_manifest'),
                                'source_group': row.get('source_group'), 'frame_index': row.get('frame_index'),
                                'capture_identity': row.get('capture_identity'),
                                'near_duplicate_group': row.get('near_duplicate_group'),
                                'point': row.get('point'), 'segment': row.get('segment'),
                                'timestamp_seconds': row.get('timestamp_seconds'),
                                'source_manifest': source.relative_to(ROOT).as_posix(), 'annotation': truth})
                records[-1]['frame_environment'] = row['frame_environment']
            manifest = {'task': task, 'split': split, 'status': 'complete_direct_candidates', 'records': records}
            (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
            if output.exists():
                output.rmdir()
            stage.rename(output)
        run.set_result({'images': len(records), 'manifest': (output / 'manifest.json').relative_to(ROOT).as_posix()})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=sorted(TASKS), required=True)
    parser.add_argument('--split', choices=sorted(SPLITS))
    parser.add_argument('--id', action='append', default=[], help='Package or inspect selected record IDs')
    parser.add_argument('--frame-record-id', action='append', default=[],
                        help='Select current frame environment decision IDs, e.g. assigned_record_id from assign')
    parser.add_argument('--output', type=Path, help='Write to the current task sample or explicitly selected split')
    args = parser.parse_args()
    selected, states = prepare(args.task, split=args.split, record_ids=args.id, frame_record_ids=args.frame_record_id)
    summary = {'task': args.task, 'split': args.split, 'eligible': len(selected), 'states': dict(states),
               'eligible_by_split': dict(Counter(item[2].get('split') or 'unassigned' for item in selected))}
    if args.id or args.frame_record_id:
        summary['sources'] = [{**{key: item[1].get(key) for key in ('id', 'image', 'capture_identity', 'parent_video')},
                               'frame_record_id': item[1]['frame_environment']['record_id']}
                              for item in selected]
    if args.output is not None:
        manifest = package(args.task, args.split, selected, ROOT / args.output)
        summary['packaged'] = len(manifest['records'])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
