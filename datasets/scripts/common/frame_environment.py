"""Durable V4 frame observations and current task-specific role decisions."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from .paths import TASKS, WORKSPACE


DEFAULT_DATABASE = WORKSPACE / 'datasets/annotated/common/sources/frame_environment.sqlite3'
SCHEMA_VERSION = 1
SPLITS = ('train', 'val', 'test')
STATUSES = ('eligible', 'unknown', 'blocked')
ORIGIN_PRIORITY = {'legacy': 0, 'machine': 1, 'human': 2}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id TEXT PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    parent_group TEXT,
    identity_scope TEXT NOT NULL,
    UNIQUE (content_sha256, identity_scope)
);
CREATE TABLE IF NOT EXISTS frame_paths (
    frame_id TEXT NOT NULL REFERENCES frames,
    path TEXT NOT NULL,
    PRIMARY KEY (frame_id, path)
);
CREATE INDEX IF NOT EXISTS frame_paths_by_path ON frame_paths(path, frame_id);
CREATE TABLE IF NOT EXISTS observations (
    evidence_id TEXT PRIMARY KEY,
    frame_id TEXT NOT NULL REFERENCES frames,
    task TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    evidence_key TEXT NOT NULL,
    origin TEXT NOT NULL,
    algorithm_version TEXT,
    reference_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    current INTEGER NOT NULL CHECK (current IN (0, 1))
);
CREATE UNIQUE INDEX IF NOT EXISTS observation_current
    ON observations(frame_id, task, kind, scope, evidence_key) WHERE current = 1;
CREATE INDEX IF NOT EXISTS lineage_capture_identity
    ON observations(json_extract(payload_json, '$.capture_identity')) WHERE kind = 'lineage';
CREATE TABLE IF NOT EXISTS decisions (
    record_id TEXT PRIMARY KEY,
    frame_id TEXT NOT NULL REFERENCES frames,
    task TEXT NOT NULL,
    role TEXT NOT NULL,
    split TEXT NOT NULL,
    status TEXT NOT NULL,
    origin TEXT NOT NULL,
    details_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    current INTEGER NOT NULL CHECK (current IN (0, 1))
);
CREATE UNIQUE INDEX IF NOT EXISTS decision_current
    ON decisions(frame_id, task, role, split) WHERE current = 1;
CREATE INDEX IF NOT EXISTS decision_query ON decisions(task, role, split, current);
CREATE TABLE IF NOT EXISTS decision_evidence (
    record_id TEXT NOT NULL REFERENCES decisions,
    evidence_id TEXT NOT NULL REFERENCES observations,
    PRIMARY KEY (record_id, evidence_id)
);
"""


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _identity(*parts):
    return sha256(_json(parts).encode('utf-8')).hexdigest()


def workspace_path(path):
    return (WORKSPACE / path).resolve().relative_to(WORKSPACE).as_posix()


def image_digest(path):
    with (WORKSPACE / workspace_path(path)).open('rb') as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def _connect(database, *, writable=False):
    path = Path(database).resolve()
    if writable:
        conn = sqlite3.connect(path, timeout=30)
    else:
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    version = conn.execute('PRAGMA user_version').fetchone()[0]
    if version != SCHEMA_VERSION:
        has_tables = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1").fetchone()
        if not writable or version != 0 or has_tables:
            conn.close()
            raise ValueError(f'Unsupported frame environment schema {version}: {path}')
        conn.executescript(f'BEGIN IMMEDIATE; {_SCHEMA} PRAGMA user_version = {SCHEMA_VERSION}; COMMIT;')
    return conn


@contextmanager
def edit_database(database=DEFAULT_DATABASE):
    """Commit frame, observations and decisions together, including on incremental ingest."""
    conn = _connect(database, writable=True)
    try:
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute(
            """CREATE INDEX IF NOT EXISTS lineage_capture_identity
               ON observations(json_extract(payload_json, '$.capture_identity')) WHERE kind = 'lineage'"""
        )
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            yield FrameWriter(conn)
            conn.execute(f'''
                UPDATE decisions AS inventory SET current = 0
                WHERE inventory.current = 1 AND inventory.split = 'unassigned' AND EXISTS (
                    SELECT 1 FROM ({_CURRENT_DECISIONS}) AS reserved
                    WHERE reserved.current = 1 AND reserved.effective_status = 'eligible'
                      AND reserved.split <> 'unassigned' AND reserved.frame_id = inventory.frame_id
                      AND reserved.task = inventory.task AND reserved.role = inventory.role
                )
            ''')
    finally:
        conn.close()


class FrameWriter:
    def __init__(self, conn):
        self.conn = conn

    def register(self, image, *, image_size, parent_group=None, verified_copy_paths=(), content_sha256=None):
        """A verified exact copy may share geometry. A changed image gets a new frame ID."""
        image = workspace_path(image)
        paths = sorted({image, *(workspace_path(path) for path in verified_copy_paths)})
        digest = content_sha256 or image_digest(image)
        for alias in paths:
            if alias != image and image_digest(alias) != digest:
                raise ValueError(f'Frame copy content differs: {image} and {alias}')
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f'Invalid frame size: {image_size}')
        # Unknown parents fall back to path scope, preventing unrelated captures
        # with identical bytes from silently becoming a single capture.
        scope = parent_group or image
        rows = self.conn.execute(
            f'''SELECT DISTINCT f.frame_id, f.content_sha256 FROM frames f JOIN frame_paths p USING(frame_id)
                WHERE p.path IN ({','.join('?' for _ in paths)})''',
            paths,
        ).fetchall()
        if any(row['content_sha256'] != digest for row in rows):
            raise ValueError(f'Registered image path now contains different bytes: {paths}')
        if len(rows) > 1:
            raise ValueError(f'Conflicting frame identities for copy paths: {paths}')
        frame_id = rows[0]['frame_id'] if rows else _identity('frame', scope, digest)
        self.conn.execute('INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?, ?, ?)',
                          (frame_id, digest, width, height, parent_group, scope))
        frame = self.conn.execute('SELECT * FROM frames WHERE frame_id = ?', (frame_id,)).fetchone()
        if (frame['width'], frame['height']) != (width, height):
            raise ValueError(f'Frame size differs across evidence: {image}')
        self.conn.executemany('INSERT OR IGNORE INTO frame_paths VALUES (?, ?)',
                              [(frame_id, path) for path in paths])
        return frame_id

    def observe(self, frame_id, *, kind, scope, evidence_key, reference, payload,
                task=None, origin='machine', algorithm_version=None):
        """Append a revision for the same observation identity. Human and machine evidence can coexist."""
        if task is not None and task not in TASKS:
            raise ValueError(f'Unknown V4 task: {task}')
        task = task or ''
        origin_priority = ORIGIN_PRIORITY[origin]
        reference_json, payload_json = _json(reference), _json(payload)
        fields = (frame_id, task, kind, scope, evidence_key)
        evidence_id = _identity('observation', fields, origin, algorithm_version, reference_json, payload_json)
        previous = self.conn.execute(
            '''SELECT origin FROM observations WHERE current = 1
               AND frame_id = ? AND task = ? AND kind = ? AND scope = ? AND evidence_key = ?''', fields,
        ).fetchone()
        if self.conn.execute('SELECT 1 FROM observations WHERE evidence_id = ?', (evidence_id,)).fetchone():
            return evidence_id
        current = previous is None or origin_priority >= ORIGIN_PRIORITY[previous['origin']]
        if current and previous:
            self.conn.execute(
                '''UPDATE observations SET current = 0 WHERE current = 1
                   AND frame_id = ? AND task = ? AND kind = ? AND scope = ? AND evidence_key = ?''', fields,
            )
        self.conn.execute(
            '''INSERT INTO observations(evidence_id, frame_id, task, kind, scope, evidence_key,
                                        origin, algorithm_version, reference_json, payload_json, current)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (evidence_id, *fields, origin, algorithm_version, reference_json, payload_json, current),
        )
        return evidence_id

    def current_observation(self, frame_id, *, task, kind, scope, evidence_key):
        row = self.conn.execute(
            '''SELECT evidence_id, payload_json FROM observations WHERE current = 1
               AND frame_id = ? AND task = ? AND kind = ? AND scope = ? AND evidence_key = ?''',
            (frame_id, task or '', kind, scope, evidence_key),
        ).fetchone()
        return (row['evidence_id'], json.loads(row['payload_json'])) if row else None

    def decide(self, frame_id, *, task, role, split, status, evidence_ids, details, origin='machine'):
        """Set a role decision with immutable references to this frame's evidence."""
        if task not in TASKS or status not in STATUSES:
            raise ValueError(f'Invalid task or role status: {task}, {status}')
        split = split or 'unassigned'
        if split not in (*SPLITS, 'unassigned'):
            raise ValueError(f'Invalid role split: {split}')
        origin_priority = ORIGIN_PRIORITY[origin]
        evidence_ids = sorted(set(evidence_ids))
        if not evidence_ids:
            raise ValueError('Role decisions require frame evidence')
        fields = (frame_id, task, role, split)
        details_json = _json(details)
        record_id = _identity('decision', fields, status, origin, details_json, evidence_ids)
        if self.conn.execute('SELECT 1 FROM decisions WHERE record_id = ?', (record_id,)).fetchone():
            return record_id
        previous = self.conn.execute(
            'SELECT origin FROM decisions WHERE current = 1 AND frame_id = ? AND task = ? AND role = ? AND split = ?',
            fields,
        ).fetchone()
        current = previous is None or origin_priority >= ORIGIN_PRIORITY[previous['origin']]
        if current and split != 'unassigned' and status == 'eligible' and not details.get('split_reuse_evidence'):
            reused = self.conn.execute(f'''
                SELECT split FROM ({_CURRENT_DECISIONS}) WHERE frame_id = ? AND task = ? AND role = ?
                  AND split NOT IN (?, 'unassigned') AND current = 1 AND effective_status = 'eligible'
            ''', fields).fetchone()
            if reused:
                raise ValueError(f'Explicit split reuse needs evidence: {frame_id} / {task} / {role}')
        linked = {details[f'{kind}_evidence_id']: kind for kind in ('label', 'geometry')
                  if f'{kind}_evidence_id' in details}
        if set(linked) - set(evidence_ids):
            raise ValueError('Decision payload links must also be referenced as evidence')
        for evidence_id in evidence_ids:
            evidence = self.conn.execute(
                'SELECT frame_id, task, current, kind FROM observations WHERE evidence_id = ?', (evidence_id,),
            ).fetchone()
            if not evidence or evidence['frame_id'] != frame_id or evidence['task'] not in ('', task):
                raise ValueError(f'Evidence does not apply to {frame_id} / {task}: {evidence_id}')
            if evidence_id in linked and evidence['kind'] != linked[evidence_id]:
                raise ValueError(f'Unexpected decision payload kind: {evidence_id}')
            if current and status == 'eligible' and not evidence['current']:
                raise ValueError(f'Eligible decisions require current evidence: {evidence_id}')
        if details.get('image') and not self.conn.execute(
            'SELECT 1 FROM frame_paths WHERE frame_id = ? AND path = ?', (frame_id, details['image']),
        ).fetchone():
            raise ValueError(f'Decision image is not registered for {frame_id}: {details["image"]}')
        if current and previous:
            self.conn.execute(
                'UPDATE decisions SET current = 0 WHERE current = 1 AND frame_id = ? AND task = ? AND role = ? AND split = ?',
                fields,
            )
        self.conn.execute(
            '''INSERT INTO decisions(record_id, frame_id, task, role, split, status, origin, details_json, current)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (record_id, *fields, status, origin, details_json, current),
        )
        self.conn.executemany('INSERT INTO decision_evidence VALUES (?, ?)',
                              [(record_id, evidence_id) for evidence_id in evidence_ids])
        return record_id


_CURRENT_DECISIONS = """
SELECT d.*, f.content_sha256, f.parent_group, f.width, f.height,
       CASE WHEN d.status = 'eligible' AND EXISTS (
           SELECT 1 FROM decision_evidence de JOIN observations o USING(evidence_id)
           WHERE de.record_id = d.record_id AND o.current = 0
       ) THEN 'unknown' ELSE d.status END AS effective_status
FROM decisions d JOIN frames f USING(frame_id)
"""


def _record(conn, row):
    evidence = []
    linked_payloads = {}
    for item in conn.execute(
        '''SELECT o.* FROM decision_evidence de JOIN observations o USING(evidence_id)
           WHERE de.record_id = ? ORDER BY o.kind, o.evidence_id''', (row['record_id'],),
    ):
        evidence.append({
            'evidence_id': item['evidence_id'], 'kind': item['kind'], 'scope': item['scope'],
            'origin': item['origin'], 'algorithm_version': item['algorithm_version'],
            'reference': json.loads(item['reference_json']), 'current': bool(item['current']),
        })
        if item['kind'] in {'label', 'geometry'}:
            linked_payloads[item['evidence_id']] = item['payload_json']
    paths = [item['path'] for item in conn.execute(
        'SELECT path FROM frame_paths WHERE frame_id = ? ORDER BY path', (row['frame_id'],),
    )]
    details = json.loads(row['details_json'])
    for kind in ('label', 'geometry'):
        if (evidence_id := details.pop(f'{kind}_evidence_id', None)) is not None:
            details[kind] = json.loads(linked_payloads[evidence_id])
    reasons = details.get('reason_codes', [])
    if row['effective_status'] != row['status']:
        reasons = [*reasons, 'evidence_superseded']
    return {
        **details, 'frame_id': row['frame_id'], 'record_id': row['record_id'], 'task': row['task'],
        'role': row['role'], 'split': row['split'], 'status': row['effective_status'],
        'recorded_status': row['status'], 'recorded_at': row['recorded_at'],
        'current': bool(row['current']), 'origin': row['origin'], 'reason_codes': reasons,
        'image_size': [row['width'], row['height']], 'content_sha256': row['content_sha256'],
        'parent_group': details.get('parent_group') or row['parent_group'], 'paths': paths, 'evidence': evidence,
    }


def query_frames(*, task, role, split=None, frame_id=None, record_id=None, status='eligible',
                 database=DEFAULT_DATABASE):
    """Query one role. Use status=None for all decisions, or record_id to replay a receipt."""
    sql = f'SELECT * FROM ({_CURRENT_DECISIONS}) WHERE task = ? AND role = ?'
    args = [task, role]
    if record_id is None:
        sql += ' AND current = 1'
    for column, value in (('split', split), ('frame_id', frame_id), ('record_id', record_id),
                          ('effective_status', status if record_id is None else None)):
        if value is not None:
            sql += f' AND {column} = ?'
            args.append(value)
    conn = _connect(database)
    try:
        with conn:
            conn.execute('BEGIN')
            return [_record(conn, row) for row in conn.execute(sql + ' ORDER BY frame_id, split', args)]
    finally:
        conn.close()


def query_split_candidates(*, task, role, split, database=DEFAULT_DATABASE):
    """List eligible records for planning, retaining the actual assignment.

    A pending record is only inventory: a caller must select and assign it
    through the decision writer before an execution query can use that split.
    This query does not resolve parent reservations or cross-split reuse.
    """
    if split not in SPLITS:
        raise ValueError('Candidate split must be train, val or test')
    return [
        {**row, 'candidate_for_split': split,
         'assignment_state': 'pending' if row['split'] == 'unassigned' else 'assigned'}
        for row in query_frames(task=task, role=role, database=database)
        if row['split'] in (split, 'unassigned')
    ]


def frame_history(frame_id, *, task=None, database=DEFAULT_DATABASE):
    """Read shared and task-specific observations, including superseded versions."""
    conn = _connect(database)
    try:
        sql = 'SELECT * FROM observations WHERE frame_id = ?'
        args = [frame_id]
        if task is not None:
            sql += " AND task IN ('', ?)"
            args.append(task)
        return [
            {**dict(row), 'reference': json.loads(row['reference_json']), 'payload': json.loads(row['payload_json'])}
            for row in conn.execute(sql + ' ORDER BY recorded_at, rowid', args)
        ]
    finally:
        conn.close()


def find_frames(*, path=None, capture_identity=None, database=DEFAULT_DATABASE):
    """Find exact frame versions and their declared capture/parent relations."""
    if path is None and capture_identity is None:
        raise ValueError('Supply a path or capture identity')
    clauses, args = [], []
    if path is not None:
        clauses.append('EXISTS (SELECT 1 FROM frame_paths p WHERE p.frame_id = f.frame_id AND p.path = ?)')
        args.append(workspace_path(path))
    if capture_identity is not None:
        clauses.append("""EXISTS (SELECT 1 FROM observations o WHERE o.frame_id = f.frame_id
                          AND o.kind = 'lineage' AND o.current = 1
                          AND json_extract(o.payload_json, '$.capture_identity') = ?)""")
        args.append(capture_identity)
    conn = _connect(database)
    try:
        with conn:
            conn.execute('BEGIN')
            frames = []
            for frame in conn.execute(
                f'SELECT f.* FROM frames f WHERE {" AND ".join(clauses)} ORDER BY frame_id', args,
            ):
                lineage = [
                    json.loads(row['payload_json']) for row in conn.execute(
                        """SELECT payload_json FROM observations WHERE frame_id = ? AND kind = 'lineage'
                           AND current = 1 ORDER BY rowid""", (frame['frame_id'],),
                    )
                ]
                frames.append({**dict(frame), 'lineage': lineage})
            return frames
    finally:
        conn.close()


def quota_summary(*, task=None, database=DEFAULT_DATABASE):
    """Count eligible stock once per frame/role, and count explicit splits separately."""
    sql = f'SELECT * FROM ({_CURRENT_DECISIONS}) WHERE current = 1'
    args = []
    if task is not None:
        sql += ' AND task = ?'
        args.append(task)
    conn = _connect(database)
    try:
        with conn:
            conn.execute('BEGIN')
            breakdown = [dict(row) for row in conn.execute(
                f'''SELECT task, role, split, effective_status AS status,
                           COALESCE(json_extract(details_json, '$.parent_group'), parent_group) AS parent_group,
                           json_extract(details_json, '$.source_kind') AS source_kind,
                           COUNT(*) AS frames
                    FROM ({sql}) GROUP BY 1, 2, 3, 4, 5, 6 ORDER BY 1, 2, 3, 4, 5, 6''', args,
            )]
            stock = [dict(row) for row in conn.execute(f'''
                SELECT task, role, COUNT(DISTINCT frame_id) AS frames,
                       COUNT(DISTINCT CASE WHEN split = 'unassigned' THEN frame_id END) AS unassigned,
                       COUNT(DISTINCT CASE WHEN split = 'train' THEN frame_id END) AS train,
                       COUNT(DISTINCT CASE WHEN split = 'val' THEN frame_id END) AS val,
                       COUNT(DISTINCT CASE WHEN split = 'test' THEN frame_id END) AS test,
                       COUNT(DISTINCT COALESCE(json_extract(details_json, '$.capture_identity'), frame_id))
                           AS captures,
                       COUNT(DISTINCT COALESCE(json_extract(details_json, '$.parent_group'), parent_group))
                           AS parents
                  FROM ({sql}) WHERE effective_status = 'eligible' GROUP BY task, role ORDER BY task, role
                ''', args)]
            by_split = [dict(row) for row in conn.execute(f'''
                SELECT task, role, split, COUNT(DISTINCT frame_id) AS frames,
                       COUNT(DISTINCT COALESCE(json_extract(details_json, '$.capture_identity'), frame_id))
                           AS captures,
                       COUNT(DISTINCT COALESCE(json_extract(details_json, '$.parent_group'), parent_group))
                           AS parents
                  FROM ({sql}) WHERE effective_status = 'eligible' GROUP BY task, role, split
                  ORDER BY task, role, split
                ''', args)]
            scene_review = [dict(row) for row in conn.execute('''
                WITH versions AS (
                    SELECT payload_json, ROW_NUMBER() OVER (
                        PARTITION BY frame_id
                        ORDER BY CASE origin WHEN 'human' THEN 2 WHEN 'machine' THEN 1 ELSE 0 END DESC, rowid DESC
                    ) AS revision_order
                    FROM observations WHERE current = 1 AND kind = 'environment'
                )
                SELECT json_extract(payload_json, '$.source_domain') AS source_domain,
                       json_extract(payload_json, '$.status') AS status, COUNT(*) AS frames
                  FROM versions WHERE revision_order = 1 GROUP BY 1, 2 ORDER BY 1, 2
                ''')]
            return {
                'registered_frames': conn.execute('SELECT COUNT(*) FROM frames').fetchone()[0],
                'registered_paths': conn.execute('SELECT COUNT(*) FROM frame_paths').fetchone()[0],
                'observation_revisions': conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
                'decision_revisions': conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0],
                'scene_review': scene_review, 'eligible_stock': stock, 'eligible_by_split': by_split,
                'decisions_by_split_parent': breakdown,
            }
    finally:
        conn.close()


def verified_image(record):
    """Check bytes at composition time; all scene and label evidence was recorded at ingest."""
    image = record['image']
    if image_digest(image) != record['content_sha256']:
        raise ValueError(f'Frame content has changed since review: {image}')
    return WORKSPACE / image
