"""Give every shared-pool frame/task an explicit composite status without inventing labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .frame_environment import DEFAULT_DATABASE, edit_database
from .paths import TASKS, WORKSPACE


POOL = WORKSPACE / 'datasets/annotated/common/sources/background_pool_manifest.json'


def _best(rows, *, kind):
    def rank(row):
        payload = json.loads(row['payload_json'])
        return (
            kind == 'label' and payload.get('completeness') == 'complete'
            and payload.get('scope', {}).get('kind') == 'frame',
            kind == 'label' and payload.get('completeness') == 'complete',
            row['origin'] == 'human', row['origin'] == 'machine',
            row['evidence_key'] == 'intake', row['evidence_id'],
        )

    return max(rows, key=rank) if rows else None


def reconcile(*, pool=POOL, database=DEFAULT_DATABASE, write=False):
    """Add only missing composite decisions; existing human and machine facts stay current."""
    manifest = json.loads(Path(pool).read_text(encoding='utf-8-sig'))
    if manifest.get('role') != 'shared_background_pool':
        raise ValueError('Expected the shared background pool manifest')
    planned, identities = [], {}
    with edit_database(database) as writer:
        conn = writer.conn
        for index, item in enumerate(manifest['records']):
            image = item['image']
            match = conn.execute('''SELECT f.frame_id, f.content_sha256, f.parent_group
                                    FROM frame_paths p JOIN frames f USING(frame_id) WHERE p.path = ?''',
                                 (image,)).fetchone()
            if not match:
                raise ValueError(f'Pool path has no registered frame: {image}')
            frame_id = match['frame_id']
            identities.setdefault(frame_id, (index, item, match))
        for frame_id, (index, item, frame) in sorted(identities.items()):
            shared = conn.execute('''SELECT evidence_id FROM observations WHERE frame_id = ?
                                     AND task = '' AND kind = 'geometry' AND scope = 'shared_background'
                                     AND current = 1''', (frame_id,)).fetchone()
            if not shared:
                raise ValueError(f'Pool frame lacks current shared spatial evidence: {frame_id}')
            for task in sorted(TASKS):
                if conn.execute('''SELECT 1 FROM decisions WHERE frame_id = ? AND task = ?
                                   AND role = 'composite_background' AND current = 1 LIMIT 1''',
                                (frame_id, task)).fetchone():
                    continue
                labels = list(conn.execute('''SELECT * FROM observations WHERE frame_id = ? AND task = ?
                                              AND kind = 'label' AND current = 1''', (frame_id, task)))
                geometry = list(conn.execute('''SELECT * FROM observations WHERE frame_id = ? AND task = ?
                                                AND kind = 'geometry' AND current = 1''', (frame_id, task)))
                label, placement = _best(labels, kind='label'), _best(geometry, kind='geometry')
                reasons = []
                if label is None:
                    reasons.append('task_label_not_reviewed')
                else:
                    payload = json.loads(label['payload_json'])
                    if payload.get('completeness') != 'complete':
                        reasons.append('task_label_incomplete')
                    elif payload.get('scope', {}).get('kind') != 'frame':
                        reasons.append('task_label_scope_requires_use_review')
                if placement is None:
                    reasons.append('task_specific_placement_not_reviewed')
                else:
                    payload = json.loads(placement['payload_json'])
                    if payload.get('status') != 'candidate':
                        reasons.append('task_specific_placement_not_qualified')
                if not reasons:
                    reasons.append('task_composite_qualification_not_recomputed')
                evidence = [shared['evidence_id']]
                details = {
                    'image': item['image'], 'parent_group': frame['parent_group'],
                    'scope': {'kind': 'frame'}, 'reason_codes': reasons,
                    'audit_basis': 'existing_current_evidence_only',
                    'pool_manifest': str(Path(pool).resolve().relative_to(WORKSPACE)).replace('\\', '/'),
                    'pool_index': index,
                }
                if label:
                    evidence.append(label['evidence_id'])
                    details['label_evidence_id'] = label['evidence_id']
                if placement:
                    evidence.append(placement['evidence_id'])
                    details['geometry_evidence_id'] = placement['evidence_id']
                row = {'frame_id': frame_id, 'task': task, 'image': item['image'],
                       'parent_group': frame['parent_group'], 'pool_index': index,
                       'reason_codes': reasons, 'evidence_ids': evidence}
                if write:
                    row['record_id'] = writer.decide(
                        frame_id, task=task, role='composite_background', split='unassigned',
                        status='unknown', evidence_ids=evidence, details=details, origin='machine',
                    )
                planned.append(row)
    return {
        'schema_version': 1, 'pool_records': len(manifest['records']),
        'frame_identities': len(identities), 'write': write, 'added_or_planned': len(planned),
        'by_task': dict(sorted(Counter(row['task'] for row in planned).items())),
        'records': planned,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=POOL)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = reconcile(pool=args.pool, database=args.database, write=args.write)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
