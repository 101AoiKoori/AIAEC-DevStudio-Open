"""Link immutable machine review rows back to the rebuilt shared background pool.

The frame environment database is the eligibility authority. This sidecar is a
rebuildable index from a legacy pool row to a versioned visual observation and
its current decision; it never edits or upgrades the historical candidate row.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from .background_pool import MACHINE_REVIEW_INDEX, POOL_MANIFEST
from .frame_environment import DEFAULT_DATABASE, workspace_path
from .paths import WORKSPACE


def _relative(path):
    return workspace_path(path)


def _observation(conn, frame_id, task, kind, scope):
    return conn.execute(
        '''SELECT evidence_id, payload_json FROM observations
           WHERE frame_id=? AND task=? AND kind=? AND scope=?
             AND evidence_key='intake' AND origin='machine' AND current=1''',
        (frame_id, task, kind, scope),
    ).fetchone()


def build_machine_review_index(review_jsonl, *, pool_manifest=POOL_MANIFEST,
                               database=DEFAULT_DATABASE):
    """Select only reviews whose bytes, shared objects and task evidence are current."""
    inputs = [review_jsonl] if isinstance(review_jsonl, (str, Path)) else list(review_jsonl)
    review_paths = [_relative(path) for path in inputs]
    if not review_paths or len(set(review_paths)) != len(review_paths):
        raise ValueError('Select distinct nonempty machine review JSONL inputs')
    pool_path = _relative(pool_manifest)
    pool = json.loads((WORKSPACE / pool_path).read_text(encoding='utf-8-sig'))
    if pool.get('role') != 'shared_background_pool' or pool.get('schema_version') != 1:
        raise ValueError('Expected the version 1 shared background pool')
    by_image = {row['image']: index for index, row in enumerate(pool['records'])}
    if len(by_image) != len(pool['records']):
        raise ValueError('Shared background pool has duplicate image paths')
    reviews, rows = [], []
    for review_path in review_paths:
        raw = (WORKSPACE / review_path).read_bytes()
        reviews.append({'path': review_path, 'sha256': sha256(raw).hexdigest()})
        rows.extend((review_path, number, json.loads(line))
                    for number, line in enumerate(raw.decode('utf-8-sig').splitlines()) if line.strip())
    latest = {(row.get('frame_id'), task): (review_path, number)
              for review_path, number, row in rows for task in row.get('task_annotations', {})}
    conn = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    records, excluded = [], []
    try:
        conn.execute('BEGIN')
        for review_path, number, row in rows:
            source = {'review_manifest': review_path, 'review_record_index': number}
            image = _relative(row['image'])
            frame_id, digest = row.get('frame_id'), row.get('content_sha256')
            if not frame_id or not digest:
                raise ValueError(f'Machine review row {number} lacks frame identity or digest')
            frame = conn.execute('SELECT content_sha256 FROM frames WHERE frame_id=?', (frame_id,)).fetchone()
            if frame is None or frame['content_sha256'] != digest or sha256((WORKSPACE / image).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Machine review row {number} has stale image bytes or frame identity')
            linked = conn.execute('SELECT 1 FROM frame_paths WHERE frame_id=? AND path=?',
                                  (frame_id, image)).fetchone()
            if linked is None:
                raise ValueError(f'Machine review row {number} is not a path of the declared frame')
            pool_index = by_image.get(image)
            if pool_index is None:
                excluded.append({**source, 'frame_id': frame_id,
                                 'image': image, 'reason': 'image_not_in_shared_pool'})
                continue
            objects = _observation(conn, frame_id, '', 'objects', 'frame')
            if objects is None or json.loads(objects['payload_json']).get('objects') != row.get('objects'):
                excluded.append({**source, 'frame_id': frame_id,
                                 'image': image, 'reason': 'shared_objects_not_current'})
                continue
            for task, annotation in row.get('task_annotations', {}).items():
                key = frame_id, task
                if latest[key] != (review_path, number):
                    excluded.append({**source, 'frame_id': frame_id, 'image': image,
                                     'task': task, 'reason': 'superseded_review_row'})
                    continue
                scope = annotation.get('scope', {'kind': 'frame'})['kind']
                label = _observation(conn, frame_id, task, 'label', scope)
                geometry = _observation(conn, frame_id, task, 'geometry', 'placement')
                if label is None or geometry is None:
                    excluded.append({**source, 'frame_id': frame_id,
                                     'image': image, 'task': task, 'reason': 'task_evidence_not_current'})
                    continue
                label_payload = json.loads(label['payload_json'])
                geometry_payload = json.loads(geometry['payload_json'])
                if (label_payload.get('evidence') != annotation.get('evidence') or
                        label_payload.get('split') != annotation.get('split') or
                        geometry_payload.get('evidence') != row['task_geometry'][task].get('evidence')):
                    excluded.append({**source, 'frame_id': frame_id,
                                     'image': image, 'task': task, 'reason': 'task_evidence_differs_from_review'})
                    continue
                current = conn.execute(
                    '''SELECT record_id, role, split, status, origin FROM decisions
                       WHERE frame_id=? AND task=? AND split=? AND current=1 ORDER BY role''',
                    (frame_id, task, annotation['split']),
                ).fetchall()
                decisions = []
                for decision in current:
                    evidence = {item[0] for item in conn.execute(
                        'SELECT evidence_id FROM decision_evidence WHERE record_id=?',
                        (decision['record_id'],))}
                    if decision['origin'] == 'machine' and evidence & {label['evidence_id'], geometry['evidence_id']}:
                        decisions.append({k: decision[k] for k in ('record_id', 'role', 'split', 'status')})
                if not decisions:
                    excluded.append({**source, 'frame_id': frame_id,
                                     'image': image, 'task': task, 'reason': 'machine_decision_not_current'})
                    continue
                records.append({
                    'pool_record_index': pool_index, 'image': image, 'frame_id': frame_id, 'task': task,
                    **source, 'review_row_sha256': sha256(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
                    ).hexdigest(),
                    'objects_evidence_id': objects['evidence_id'],
                    'label_evidence_id': label['evidence_id'],
                    'geometry_evidence_id': geometry['evidence_id'], 'decisions': decisions,
                })
        conn.rollback()
    finally:
        conn.close()
    return {'schema_version': 1, 'role': 'background_pool_machine_review_index',
            'source_manifest': pool_path, 'review_manifests': reviews, 'database': _relative(database),
            'records': records, 'excluded': excluded}


def publish_machine_review_index(review_jsonl, *, pool_manifest=POOL_MANIFEST,
                                 database=DEFAULT_DATABASE, output=MACHINE_REVIEW_INDEX):
    """Atomically publish a checked sidecar; stale rows remain explicitly excluded."""
    payload = build_machine_review_index(review_jsonl, pool_manifest=pool_manifest, database=database)
    output_path = WORKSPACE / _relative(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as temporary:
        staged = Path(temporary) / output_path.name
        staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        staged.replace(output_path)
    return payload


def check_machine_review_index(index=MACHINE_REVIEW_INDEX, *, database=DEFAULT_DATABASE):
    path = WORKSPACE / _relative(index)
    stored = json.loads(path.read_text(encoding='utf8'))
    current = build_machine_review_index([item['path'] for item in stored['review_manifests']],
                                         pool_manifest=stored['source_manifest'], database=database)
    if stored != current:
        raise ValueError('Machine review sidecar is stale relative to its JSONL, pool or database')
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'check'))
    parser.add_argument('--review-jsonl', type=Path, action='append')
    parser.add_argument('--pool-manifest', default=POOL_MANIFEST)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', default=MACHINE_REVIEW_INDEX)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        if args.review_jsonl is None:
            parser.error('build requires --review-jsonl')
        result = (publish_machine_review_index if args.write else build_machine_review_index)(
            args.review_jsonl, pool_manifest=args.pool_manifest, database=args.database,
            **({'output': args.output} if args.write else {}),
        )
    else:
        result = check_machine_review_index(args.output, database=args.database)
    print(json.dumps({'records': len(result['records']), 'excluded': len(result['excluded']),
                      'reviews': result['review_manifests']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
