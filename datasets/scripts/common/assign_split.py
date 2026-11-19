"""Assign explicitly selected frame decisions; never infer a split or source identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .frame_environment import DEFAULT_DATABASE, SPLITS, FrameWriter, _connect, _CURRENT_DECISIONS, _record
from .paths import WORKSPACE, TASKS
from .source_history import RELOCATIONS, read_source_history
from .source_ledger.config import build_groups, source_config_path
from .source_ledger.reservations import candidate_reservation


def assign_selected(task, role, split, record_ids, *, database=DEFAULT_DATABASE, root=WORKSPACE):
    """Assign selected current inventory records, returning per-record outcomes.

    Explicit selection permits unknown or ambiguous parent relations and records
    them; known reservations still apply. A matching parent or group in another
    split is reported as contact, never as independent evidence. The writer's
    same-frame reuse rule remains in force.
    """
    if task not in TASKS or not isinstance(role, str) or not role.strip() or split not in SPLITS:
        raise ValueError('Expected a known task, role and train/val/test split')
    if not isinstance(record_ids, (list, tuple)) or any(not isinstance(x, str) or not x for x in record_ids):
        raise ValueError('record_ids must be a list of nonempty IDs')
    root = Path(root)
    source_path = source_config_path(root, task)
    history_path = root / RELOCATIONS
    source_bytes = source_path.read_bytes()
    history_bytes = history_path.read_bytes() if history_path.exists() else None
    source = yaml.safe_load(source_bytes)
    if not isinstance(source, dict) or source.get('task') != task:
        raise ValueError('Invalid task source configuration')
    groups = build_groups(task, source.get('pairing') or {})
    history = read_source_history(root)
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    outcomes = []
    if not Path(database).is_file():
        raise FileNotFoundError(f'Frame environment database not found: {database}')
    conn = _connect(database, writable=True)
    try:
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            writer = FrameWriter(conn)
            for record_id in record_ids:
                outcome = {'record_id': record_id, 'requested_split': split}
                row = conn.execute(f'SELECT * FROM ({_CURRENT_DECISIONS}) WHERE record_id = ?',
                                   (record_id,)).fetchone()
                if row is None or row['task'] != task or row['role'] != role or row['split'] != 'unassigned':
                    outcomes.append({**outcome, 'status': 'unknown_record'})
                    continue
                # A repeated request can return the already committed result, but
                # never treat an obsolete assigned decision as successful.
                if not row['current']:
                    existing = conn.execute(f'''SELECT * FROM ({_CURRENT_DECISIONS})
                        WHERE frame_id = ? AND task = ? AND role = ? AND split = ? AND current = 1
                          AND effective_status = 'eligible' ''', (row['frame_id'], task, role, split)).fetchall()
                    match = next((item for item in existing
                                  if json.loads(item['details_json']).get('assignment', {}).get('inventory_record_id')
                                  == record_id), None)
                    outcomes.append({**outcome, 'status': 'unchanged' if match else 'stale',
                                     **({'assigned_record_id': match['record_id']} if match else {})})
                    continue
                if row['effective_status'] != 'eligible':
                    outcomes.append({**outcome, 'status': 'stale'})
                    continue
                record = _record(conn, row)
                reservation = candidate_reservation(record, split, groups, history)
                if reservation['status'] == 'conflict':
                    outcomes.append({**outcome, 'status': reservation['status'], 'reservation': reservation})
                    continue
                others = conn.execute(f'''SELECT * FROM ({_CURRENT_DECISIONS})
                    WHERE task = ? AND role = ? AND current = 1 AND effective_status = 'eligible'
                      AND split NOT IN (?, 'unassigned')''', (task, role, split)).fetchall()
                related = []
                for item in others:
                    other = _record(conn, item)
                    matched = candidate_reservation(other, split, groups, history)['matched_groups']
                    same_parent = bool(record.get('parent_group') and
                                       record['parent_group'] == other.get('parent_group'))
                    same_group = bool(set(reservation['matched_groups']) & set(matched))
                    if same_parent or same_group or record['frame_id'] == other['frame_id']:
                        related.append({'record_id': other['record_id'], 'frame_id': other['frame_id'],
                                        'split': other['split'], 'same_frame': record['frame_id'] == other['frame_id'],
                                        'same_parent': same_parent, 'shared_declaration': same_group})
                details = json.loads(row['details_json'])
                details['assignment'] = {'inventory_record_id': record_id,
                                         'reservation': reservation, 'related_records': related,
                                         'contact_scope': 'current_eligible_same_task_and_role',
                                         'source_config_sha256': fingerprint}
                evidence = [item['evidence_id'] for item in conn.execute(
                    'SELECT evidence_id FROM decision_evidence WHERE record_id = ? ORDER BY evidence_id', (record_id,))]
                conn.execute('SAVEPOINT assign_one')
                try:
                    new_id = writer.decide(row['frame_id'], task=task, role=role, split=split, status='eligible',
                                           evidence_ids=evidence, details=details, origin='machine')
                    assigned = conn.execute(f'SELECT * FROM ({_CURRENT_DECISIONS}) WHERE record_id = ?',
                                            (new_id,)).fetchone()
                    if not assigned['current'] or assigned['effective_status'] != 'eligible':
                        raise ValueError('Assigned record is not current and eligible')
                    conn.execute('UPDATE decisions SET current = 0 WHERE record_id = ?', (record_id,))
                    conn.execute('RELEASE assign_one')
                    outcomes.append({**outcome, 'status': 'assigned', 'assigned_record_id': new_id,
                                     'reservation': reservation, 'related_records': related,
                                     'contact_scope': 'current_eligible_same_task_and_role'})
                except ValueError as exc:
                    conn.execute('ROLLBACK TO assign_one')
                    conn.execute('RELEASE assign_one')
                    outcomes.append({**outcome, 'status': 'conflict', 'reason': str(exc),
                                     'reservation': reservation, 'related_records': related})
            if source_path.read_bytes() != source_bytes or (
                history_path.read_bytes() if history_path.exists() else None
            ) != history_bytes:
                raise ValueError('Source declarations changed during assignment; transaction rolled back')
    finally:
        conn.close()
    return outcomes
