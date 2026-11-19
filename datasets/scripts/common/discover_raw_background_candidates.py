"""Inventory registered raw frames for cross-task background review.

This is a read-only discovery tool. A task directory is provenance, never proof
that another task's target is absent or that a frame is fit for compositing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .frame_environment import DEFAULT_DATABASE
from .paths import TASKS, WORKSPACE


RAW_ROOT = WORKSPACE / 'datasets/raw'
POOL_MANIFEST = WORKSPACE / 'datasets/annotated/common/sources/background_pool_manifest.json'
IMAGE_SUFFIXES = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.ppm'})


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE).as_posix()


def _payload(value: str) -> dict:
    result = json.loads(value)
    return result if isinstance(result, dict) else {}


def _label_summary(observation: sqlite3.Row) -> dict:
    payload = _payload(observation['payload_json'])
    scope = payload.get('scope')
    scope_kind = scope.get('kind') if isinstance(scope, dict) else observation['scope']
    return {
        'evidence_id': observation['evidence_id'],
        'scope': scope_kind,
        'completeness': payload.get('completeness', 'unknown'),
        'presence': payload.get('presence', 'unknown'),
        'state': payload.get('label_state', payload.get('state', 'unknown')),
        'instance_count': len(payload.get('instances') or []),
        'origin': observation['origin'],
    }


def _task_labels(items: list[dict]) -> dict:
    full = [item for item in items if item['scope'] == 'frame'
            and item['completeness'] == 'complete'
            and item['presence'] in {'present', 'absent'}]
    presences = {item['presence'] for item in full}
    if len(presences) > 1:
        state = 'conflicting_complete'
    elif full:
        state = 'complete_positive' if 'present' in presences else 'complete_negative'
    elif items:
        state = 'incomplete_or_local'
    else:
        state = 'not_recorded'
    return {'state': state, 'observations': items}


def _space_summary(observation: sqlite3.Row) -> dict:
    payload = _payload(observation['payload_json'])
    placement = payload.get('placement_record')
    if not isinstance(placement, dict):
        placement = {}
    return {
        'evidence_id': observation['evidence_id'],
        'kind': observation['kind'],
        'scope': observation['scope'],
        'algorithm_version': observation['algorithm_version'],
        'status': payload.get('status'),
        'geometry_basis': payload.get('geometry_basis'),
        'primary_subject_task': payload.get('primary_subject_task') or (
            payload.get('task') if payload.get('status') == 'primary_task_target' else None),
        'ground_reason_codes': payload.get('ground_reason_codes', []),
        'ground_polygon_vertices': len(payload.get('ground_polygon') or []),
        'placement_status': placement.get('status', payload.get('placement_status')),
    }


def discover(*, database=DEFAULT_DATABASE, raw_root=RAW_ROOT, pool_manifest=POOL_MANIFEST):
    """Return (summary, frames) without decoding media or changing the database."""
    raw_root = Path(raw_root)
    pool_manifest = Path(pool_manifest)
    raw_root_relative = Path(_relative(raw_root))
    raw_paths = sorted(_relative(path) for path in raw_root.rglob('*')
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    conn = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute('PRAGMA user_version').fetchone()[0] != 1:
            raise ValueError('Unsupported frame environment schema')
        registered = {row['path']: row['frame_id'] for row in conn.execute(
            'SELECT path, frame_id FROM frame_paths')}
        by_frame = defaultdict(list)
        unregistered = []
        for path in raw_paths:
            frame_id = registered.get(path)
            if frame_id is None:
                unregistered.append(path)
            else:
                by_frame[frame_id].append(path)

        manifest = json.loads(pool_manifest.read_text(encoding='utf-8'))
        pool_paths = [row['image'] for row in manifest['records']]
        for extension in manifest.get('extensions', []):
            extension_path = WORKSPACE / extension
            batch = json.loads(extension_path.read_text(encoding='utf-8'))
            pool_paths.extend(row['image'] for row in batch['records'])
        if len(pool_paths) != len(set(pool_paths)):
            raise ValueError('Shared background pool has duplicate image paths')
        pool_by_frame = defaultdict(list)
        pool_unregistered = []
        for path in pool_paths:
            frame_id = registered.get(path)
            if frame_id is None:
                pool_unregistered.append(path)
            else:
                pool_by_frame[frame_id].append(path)
        pool_frame_ids = set(pool_by_frame)

        frames = {}
        for row in conn.execute('SELECT frame_id, content_sha256, width, height, parent_group '
                                'FROM frames'):
            frame_id = row['frame_id']
            if frame_id not in by_frame:
                continue
            frames[frame_id] = {
                'frame_id': frame_id,
                'content_sha256': row['content_sha256'],
                'image_size': [row['width'], row['height']],
                'parent_group': row['parent_group'],
                'raw_paths': by_frame[frame_id],
                'raw_top_level_directories': sorted({
                    Path(path).relative_to(raw_root_relative).parts[0]
                    for path in by_frame[frame_id]
                }),
                'in_shared_pool': frame_id in pool_frame_ids,
                'shared_pool_paths': sorted(pool_by_frame.get(frame_id, [])),
                'task_labels': {task: _task_labels([]) for task in sorted(TASKS)},
                'observed_objects': [],
                'spatial_observations': [],
                'composite_background': {task: [] for task in sorted(TASKS)},
            }

        labels = defaultdict(list)
        for row in conn.execute("SELECT * FROM observations WHERE current=1 AND kind='label'"):
            if row['frame_id'] in frames and row['task'] in TASKS:
                labels[row['frame_id'], row['task']].append(_label_summary(row))
        for (frame_id, task), items in labels.items():
            frames[frame_id]['task_labels'][task] = _task_labels(items)

        for row in conn.execute("SELECT frame_id, evidence_id, payload_json FROM observations "
                                "WHERE current=1 AND kind='objects' AND task=''"):
            if row['frame_id'] not in frames:
                continue
            payload = _payload(row['payload_json'])
            frames[row['frame_id']]['observed_objects'].append({
                'evidence_id': row['evidence_id'],
                'objects': [{key: obj[key] for key in ('object_id', 'kind', 'visibility', 'xyxy')
                             if key in obj} for obj in payload.get('objects', []) if isinstance(obj, dict)],
            })

        for row in conn.execute("SELECT * FROM observations WHERE current=1 AND task='' "
                                "AND kind IN ('geometry','environment')"):
            if row['frame_id'] in frames:
                frames[row['frame_id']]['spatial_observations'].append(_space_summary(row))

        decision_sql = '''
            SELECT d.record_id, d.frame_id, d.task, d.split, d.status, d.origin,
                   d.details_json, d.recorded_at,
                   CASE WHEN d.status='eligible' AND EXISTS (
                     SELECT 1 FROM decision_evidence de JOIN observations o USING(evidence_id)
                     WHERE de.record_id=d.record_id AND o.current=0
                   ) THEN 'unknown' ELSE d.status END AS effective_status
            FROM decisions d WHERE d.current=1 AND d.role='composite_background'
        '''
        for row in conn.execute(decision_sql):
            if row['frame_id'] not in frames or row['task'] not in TASKS:
                continue
            details = _payload(row['details_json'])
            reasons = list(details.get('reason_codes') or [])
            if row['effective_status'] != row['status']:
                reasons.append('evidence_superseded')
            frames[row['frame_id']]['composite_background'][row['task']].append({
                'record_id': row['record_id'], 'split': row['split'],
                'status': row['effective_status'], 'recorded_status': row['status'],
                'origin': row['origin'], 'reason_codes': reasons,
                'recorded_at': row['recorded_at'],
            })
        result = sorted(frames.values(), key=lambda item: item['frame_id'])
        for frame in result:
            frame['spatial_observations'].sort(key=lambda item: (item['kind'], item['evidence_id']))
            for decisions in frame['composite_background'].values():
                decisions.sort(key=lambda item: (item['split'], item['record_id']))

        eligibility = {}
        label_states = {}
        for task in sorted(TASKS):
            eligibility[task] = dict(Counter(
                'eligible' if any(d['status'] == 'eligible' for d in f['composite_background'][task])
                else 'decided_without_eligible' if f['composite_background'][task] else 'not_recorded'
                for f in result))
            label_states[task] = dict(Counter(f['task_labels'][task]['state'] for f in result))
        summary = {
            'raw_image_paths': len(raw_paths),
            'registered_raw_image_paths': sum(map(len, by_frame.values())),
            'unregistered_raw_image_paths': unregistered,
            'raw_frame_identities': len(result),
            'shared_pool_records': len(pool_paths),
            'shared_pool_registered_frame_identities': len(pool_frame_ids),
            'shared_pool_frame_identities_in_raw': sum(f['in_shared_pool'] for f in result),
            'shared_pool_unregistered_paths': pool_unregistered,
            'raw_frame_identities_outside_pool': sum(not f['in_shared_pool'] for f in result),
            'raw_frames_with_shared_spatial_observation': sum(
                any(o['scope'] == 'shared_background' for o in f['spatial_observations'])
                for f in result),
            'composite_background_by_task': eligibility,
            'label_state_by_task': label_states,
            'interpretation': 'Discovery only; directory and pool membership do not prove target absence or placement eligibility.',
        }
        return summary, result
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--pool-manifest', type=Path, default=POOL_MANIFEST)
    parser.add_argument('--output-json', type=Path, help='Write summary and all frames as JSON')
    parser.add_argument('--output-jsonl', type=Path, help='Write one frame per JSONL line')
    args = parser.parse_args(argv)
    summary, frames = discover(database=args.database, raw_root=args.raw_root,
                               pool_manifest=args.pool_manifest)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({'summary': summary, 'frames': frames},
                                              ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open('w', encoding='utf-8') as stream:
            for frame in frames:
                stream.write(json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
