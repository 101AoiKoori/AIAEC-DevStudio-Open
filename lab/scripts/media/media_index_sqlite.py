"""Rebuildable SQLite query cache for the media JSONL catalog.

The JSON catalog and its versioned shards remain authoritative. A cache is
usable only while its catalog and lifecycle snapshot exactly match the source.
An explicitly partial cache is only queryable with ``--allow-partial`` and
does not claim to include evidence from omitted partitions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import media_index as core


SCHEMA_VERSION = '3'
DATABASE = core.INDEX / 'query_cache.sqlite3'
FILTER_FIELDS = {
    'generation_method': ('generation_method',),
    'source_group': ('source_group', 'source_id', 'episode_group', 'near_duplicate_group'),
    'observed_content': ('observed_content',),
    'review_flag': ('review_flag',),
    'background_view': ('background_view',),
    'background_placement_plane': ('background_placement_plane',),
    'background_target_presence': ('background_target_presence',),
    'background_use': ('background_use',),
    'background_review_state': ('background_review_state',),
    'observed_task': ('observed_tasks',),
    'region_role': ('regions',),
    'region_task': ('regions',),
    'ground_patch_use': ('visible_ground_patches',),
    'split': ('split',),
    'evaluation_scope': ('evaluation_scope', 'conclusion_scope', 'test_kind'),
}
EXACT_FILTERS = {'background_view', 'background_placement_plane',
                 'background_target_presence', 'background_use',
                 'background_review_state'}
NUMERIC_FILTERS = {
    'min_luma': ('mean_luma_0_255', '>='),
    'max_luma': ('mean_luma_0_255', '<='),
    'min_saturation': ('mean_saturation_0_255', '>='),
    'max_saturation': ('mean_saturation_0_255', '<='),
}


@contextmanager
def cache_guard(timeout=60):
    """Use an OS-owned lock separate from the JSON builder's catalog lease."""
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with (DATABASE.parent / '.sqlite-cache.lock').open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if not lock.tell():
            lock.write(b'0')
            lock.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('SQLite cache writer is busy')
                time.sleep(.05)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def path_key(path):
    return path.casefold() if os.name == 'nt' else path


def normalize_policy(value):
    """Normalize a cached policy document to the path-independent identity."""
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    if isinstance(data, dict) and isinstance(data.get('metadata'), dict):
        data['metadata'] = core.policy_identity(data['metadata'])
    if isinstance(data, dict) and isinstance(data.get('records'), list):
        data['records'] = [
            {key: item for key, item in record.items() if key != 'policy'}
            if isinstance(record, dict) else record
            for record in data['records']
        ]
    return canonical(data)


def policy_payload():
    """Return the semantic lifecycle payload without filesystem locations."""
    records = [
        {key: item for key, item in record.items() if key != 'policy'}
        for record in core._LIFECYCLE_RECORDS
    ]
    return {'records': records, 'patterns': core._LIFECYCLE_PATTERNS,
            'metadata': core.policy_identity(core._POLICY_METADATA)}


def snapshot():
    """Read catalog and effective lifecycle policy while publication is locked."""
    with core.catalog_guard():
        core.load_lifecycle_rules()
        if core._LIFECYCLE_WARNINGS:
            raise ValueError('lifecycle evidence unavailable: ' + canonical(core._LIFECYCLE_WARNINGS))
        catalog = json.loads((core.INDEX / 'catalog.json').read_text(encoding='utf8'))
        latest_build = catalog.get('latest_build_scope', {})
        last_full = catalog.get('last_full_build', {})
        latest_revision = core.policy_identity(latest_build.get('lifecycle_policy_revision'))
        full_revision = core.policy_identity(last_full.get('lifecycle_policy_revision'))
        if latest_revision != core.policy_identity(core._POLICY_METADATA) or full_revision != core.policy_identity(core._POLICY_METADATA):
            raise ValueError('JSON catalog lifecycle revision is stale; complete a JSON rebuild first')
        if last_full.get('state') != 'complete' or last_full.get('failures'):
            raise ValueError('last full JSON build is incomplete or failed')
        parts = []
        incomplete = []
        for ordinal, (name, part) in enumerate(catalog['partitions'].items()):
            if core.lifecycle_reason(name):
                continue
            warnings = part.get('warnings', [])
            if any(w.get('error') != 'manifest exceeds 64 MiB; not parsed' for w in warnings):
                incomplete.append(name)
            shard = core.inside(part['shard'])
            evidence = core.inside(part['evidence_shard']) if part.get('evidence_shard') else None
            shard_stat = shard.stat()
            evidence_stat = evidence.stat() if evidence else None
            parts.append({'name': name, 'shard': part['shard'],
                          'evidence_shard': part.get('evidence_shard'),
                          'shard_stat': [shard_stat.st_size, shard_stat.st_mtime_ns],
                          'evidence_stat': [evidence_stat.st_size, evidence_stat.st_mtime_ns] if evidence_stat else None,
                          'warnings': warnings,
                          'source_present': part.get('source_present', True),
                          'updated_utc': part.get('updated_utc', ''), 'ordinal': ordinal})
        policy = canonical(policy_payload())
        state = latest_build.get('state')
        if state != 'complete' or latest_build.get('failures') or incomplete:
            state = 'incomplete'
        return parts, policy, state


def selected_snapshot(parts, names):
    by_name = {p['name']: p for p in parts}
    if names:
        missing = set(names) - set(by_name)
        if missing:
            raise ValueError('requested partitions absent or inactive: ' + ', '.join(sorted(missing)))
        return [p for p in parts if p['name'] in set(names)]
    return parts


def connect(readonly=False):
    if readonly:
        # Resolving this path never creates a missing cache.
        if not DATABASE.is_file():
            raise ValueError('SQLite cache missing; run sync first')
        db = sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True, timeout=10)
    else:
        DATABASE.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(DATABASE), timeout=30)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA foreign_keys=ON')
    return db


def init_schema(db):
    has_meta = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
    if has_meta:
        version = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if version and version[0] != SCHEMA_VERSION:
            raise ValueError('unsupported SQLite cache schema; rebuild in a new cache location')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS partitions (name TEXT PRIMARY KEY, descriptor TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS media (
            partition_name TEXT NOT NULL REFERENCES partitions(name) ON DELETE CASCADE,
            position INTEGER NOT NULL, path TEXT NOT NULL, path_key TEXT NOT NULL,
            item_json TEXT NOT NULL, PRIMARY KEY(partition_name, position));
        CREATE INDEX IF NOT EXISTS media_path ON media(path);
        CREATE INDEX IF NOT EXISTS media_path_key ON media(path_key);
        CREATE TABLE IF NOT EXISTS manifest_rows (
            id INTEGER PRIMARY KEY,
            partition_name TEXT NOT NULL REFERENCES partitions(name) ON DELETE CASCADE,
            manifest TEXT NOT NULL, record_index INTEGER NOT NULL, shared_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS manifest_rows_partition ON manifest_rows(partition_name);
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY, partition_name TEXT NOT NULL REFERENCES partitions(name) ON DELETE CASCADE,
            position INTEGER NOT NULL, path TEXT NOT NULL, path_key TEXT NOT NULL, entry_json TEXT NOT NULL,
            manifest_row_id INTEGER REFERENCES manifest_rows(id) ON DELETE CASCADE,
            UNIQUE(partition_name, position));
        CREATE INDEX IF NOT EXISTS evidence_path ON evidence(path);
        CREATE INDEX IF NOT EXISTS evidence_path_key ON evidence(path_key);
        CREATE INDEX IF NOT EXISTS evidence_manifest_row ON evidence(manifest_row_id);
        CREATE TABLE IF NOT EXISTS terms (
            entry_id INTEGER NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
            field TEXT NOT NULL, value TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS terms_field ON terms(field);
        CREATE INDEX IF NOT EXISTS terms_entry ON terms(entry_id);
        CREATE TABLE IF NOT EXISTS batches (
            entry_id INTEGER NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
            record_json TEXT NOT NULL, text_cf TEXT NOT NULL,
            scopes_cf TEXT NOT NULL, statuses_cf TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS batches_entry ON batches(entry_id);
    ''')
    version = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if version and version[0] != SCHEMA_VERSION:
        raise ValueError('unsupported SQLite cache schema')
    db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)", (SCHEMA_VERSION,))
    db.commit()


def put_meta(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)', (key, value))


def read_shards(part, policy):
    """Copy one immutable shard pair into memory under a short catalog lease."""
    with core.catalog_guard():
        core.load_lifecycle_rules()
        if core._LIFECYCLE_WARNINGS:
            raise ValueError('lifecycle evidence unavailable')
        current_policy = canonical(policy_payload())
        catalog = json.loads((core.INDEX / 'catalog.json').read_text(encoding='utf8'))
        current = catalog['partitions'].get(part['name'])
        if (current is None or current_policy != policy or current.get('shard') != part['shard'] or
                current.get('evidence_shard') != part['evidence_shard'] or
                current.get('warnings', []) != part['warnings'] or
                core.policy_identity(catalog.get('latest_build_scope', {}).get('lifecycle_policy_revision')) != core.policy_identity(core._POLICY_METADATA) or
                core.policy_identity(catalog.get('last_full_build', {}).get('lifecycle_policy_revision')) != core.policy_identity(core._POLICY_METADATA) or
                current.get('source_present', True) != part['source_present'] or
                current.get('updated_utc', '') != part['updated_utc']):
            raise ValueError('catalog or lifecycle changed during sync; retry')
        media_path = core.inside(part['shard'])
        evidence_path = core.inside(part['evidence_shard']) if part['evidence_shard'] else None
        for path, expected in ((media_path, part['shard_stat']), (evidence_path, part['evidence_stat'])):
            if path:
                stat = path.stat()
                if [stat.st_size, stat.st_mtime_ns] != expected:
                    raise ValueError('JSON shard changed during sync; retry')
        media = media_path.read_bytes()
        evidence = evidence_path.read_bytes() if evidence_path else b''
    return media, evidence


def import_partition(db, part, media_bytes, evidence_bytes):
    """Replace all rows of one partition in a single SQLite transaction."""
    name = part['name']
    with db:
        db.execute('DELETE FROM partitions WHERE name=?', (name,))
        db.execute('INSERT INTO partitions(name,descriptor) VALUES(?,?)', (name, canonical(part)))
        for position, line in enumerate(media_bytes.splitlines()):
            item = json.loads(line)
            path = item['path']
            db.execute('INSERT INTO media VALUES(?,?,?,?,?)',
                       (name, position, path, path_key(path),
                        line.decode('utf8')))
        fields = {f for group in FILTER_FIELDS.values() for f in group}
        fields.update(field for field, _ in NUMERIC_FILTERS.values())
        manifest_rows = {}
        for position, line in enumerate(evidence_bytes.splitlines()):
            entry = json.loads(line)
            row_id = None
            entry_json = line.decode('utf8')
            if len(entry['evidence']) == 1:
                record = entry['evidence'][0]
                if record.get('manifest') is not None and record.get('record_index') is not None:
                    shared = {k: record[k] for k in ('claims', 'manifest_context') if k in record}
                    shared_json = json.dumps(shared, ensure_ascii=False, separators=(',', ':'))
                    key = (record['manifest'], record['record_index'], shared_json)
                    row_id = manifest_rows.get(key)
                    if row_id is None:
                        row_id = db.execute('INSERT INTO manifest_rows(partition_name,manifest,record_index,shared_json) '
                                            'VALUES(?,?,?,?)', (name, *key)).lastrowid
                        manifest_rows[key] = row_id
                    compact = dict(record)
                    for field in shared:
                        compact[field] = None
                    compact_entry = dict(entry, evidence=[compact])
                    entry_json = json.dumps(compact_entry, ensure_ascii=False, separators=(',', ':'))
            cur = db.execute('INSERT INTO evidence(partition_name,position,path,path_key,entry_json,manifest_row_id) '
                             'VALUES(?,?,?,?,?,?)',
                             (name, position, entry['claimed_path'], path_key(entry['claimed_path']), entry_json, row_id))
            values = set()
            for record in entry['evidence']:
                if record.get('relation') == 'batch_context':
                    context = record.get('batch_context', {})
                    evaluation = context.get('evaluation')
                    evaluation = evaluation if isinstance(evaluation, dict) else {}
                    scopes = [context.get('evaluation_scope'), context.get('conclusion_scope'), evaluation.get('data_scope')]
                    statuses = [context.get('status'), evaluation.get('status')]
                    db.execute('INSERT INTO batches VALUES(?,?,?,?,?)',
                               (cur.lastrowid, json.dumps(record, ensure_ascii=False),
                                json.dumps(record, ensure_ascii=False).casefold(),
                                json.dumps(scopes, ensure_ascii=False).casefold(),
                                json.dumps(statuses, ensure_ascii=False).casefold()))
                    continue
                claims = record['claims']
                for field in fields:
                    if claims.get(field) is not None:
                        values.add((field, json.dumps(claims[field], ensure_ascii=False).casefold()))
            db.executemany('INSERT INTO terms(entry_id,field,value) VALUES(?,?,?)',
                           ((cur.lastrowid, field, value) for field, value in values))


def sync(args):
    with cache_guard():
        parts, policy, state = snapshot()
        selected = selected_snapshot(parts, args.partition)
        # A partial cache has an explicit, restricted query scope.
        full_requested = not args.partition
        if full_requested and state != 'complete':
            raise ValueError('JSON catalog build is not complete: ' + str(state))
        # A complete, ready cache with the same catalog needs no write connection,
        # transaction, timestamp update, or partition import.
        if state == 'complete' and DATABASE.is_file():
            read_db = connect(readonly=True)
            try:
                tables = {name for name, in read_db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if {'meta', 'partitions'} <= tables:
                    meta = dict(read_db.execute('SELECT key,value FROM meta'))
                    indexes = {name for name, in read_db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
                    current = {p['name']: canonical(p) for p in parts}
                    cached = dict(read_db.execute('SELECT name,descriptor FROM partitions'))
                    unchanged = (meta.get('schema_version') == SCHEMA_VERSION and
                                 {'manifest_rows_partition', 'evidence_manifest_row'} <= indexes and
                                 meta.get('state') == 'ready' and meta.get('complete_catalog') == '1' and
                                 meta.get('policy') == policy and
                                 meta.get('scope') == canonical([p['name'] for p in parts]) and cached == current)
                    if unchanged:
                        latest, latest_policy, latest_state = snapshot()
                        if canonical(latest) != canonical(parts) or latest_policy != policy or latest_state != state:
                            raise ValueError('catalog or lifecycle changed during sync; retry')
                        print(json.dumps({'state': 'ready', 'partitions': len(parts),
                                          'complete_catalog': True, 'unchanged': True}), flush=True)
                        return 0
            finally:
                read_db.close()
        db = connect()
        try:
            init_schema(db)
            old_policy = db.execute("SELECT value FROM meta WHERE key='policy'").fetchone()
            with db:
                put_meta(db, 'state', 'building')
                # An altered lifecycle invalidates all previous partitions.
                if old_policy and old_policy[0] != policy:
                    db.execute('DELETE FROM partitions')
                current = {p['name']: canonical(p) for p in parts}
                for name, descriptor in list(db.execute('SELECT name,descriptor FROM partitions')):
                    if name not in current or descriptor != current[name]:
                        db.execute('DELETE FROM partitions WHERE name=?', (name,))
            for part in selected:
                old = db.execute('SELECT descriptor FROM partitions WHERE name=?', (part['name'],)).fetchone()
                if old and old[0] == canonical(part):
                    continue
                media, evidence = read_shards(part, policy)
                import_partition(db, part, media, evidence)
                print(json.dumps({'partition': part['name'], 'media_bytes': len(media),
                                  'evidence_bytes': len(evidence)}, ensure_ascii=False), flush=True)
            latest, latest_policy, latest_state = snapshot()
            if (canonical(selected_snapshot(latest, [p['name'] for p in selected])) != canonical(selected)
                    or latest_policy != policy or (full_requested and (len(latest) != len(parts) or latest_state != 'complete'))):
                raise ValueError('catalog or lifecycle changed during sync; retry')
            with db:
                current = {p['name']: canonical(p) for p in latest}
                for name, descriptor in list(db.execute('SELECT name,descriptor FROM partitions')):
                    if name not in current or descriptor != current[name]:
                        db.execute('DELETE FROM partitions WHERE name=?', (name,))
                cached = {name for name, in db.execute('SELECT name FROM partitions')}
                names = [p['name'] for p in latest if p['name'] in cached]
                full = len(names) == len(latest) and latest_state == 'complete'
                put_meta(db, 'policy', policy)
                put_meta(db, 'scope', canonical(names))
                put_meta(db, 'complete_catalog', '1' if full else '0')
                put_meta(db, 'snapshot_utc', datetime.now(timezone.utc).isoformat())
                put_meta(db, 'state', 'ready')
            print(json.dumps({'state': 'ready', 'partitions': len(names),
                              'complete_catalog': full}, ensure_ascii=False), flush=True)
            return 0
        finally:
            db.close()


def matching_entries(db, args, only_path=None):
    filters = [(key, getattr(args, key)) for key in FILTER_FIELDS
               if key not in ('region_role', 'region_task') and getattr(args, key)]
    clauses, params = [], []
    for key, value in filters:
        fields = FILTER_FIELDS[key]
        if key == 'observed_task':
            predicate = "EXISTS (SELECT 1 FROM json_each(t.value) j WHERE lower(j.value)=?)"
        elif key == 'ground_patch_use':
            predicate = "EXISTS (SELECT 1 FROM json_each(t.value) j WHERE CASE WHEN j.type='object' THEN lower(json_extract(j.value,'$.use_status'))=? ELSE 0 END)"
        else:
            predicate = ('lower(json_extract(value,\'$\'))=?' if key in EXACT_FILTERS
                         else 'instr(value,?)>0')
        clauses.append('SELECT entry_id FROM terms t WHERE field IN (' + ','.join('?' for _ in fields) + ') AND ' + predicate)
        params.extend(fields)
        params.append(value.casefold())
    role, task = getattr(args, 'region_role', None), getattr(args, 'region_task', None)
    if role or task:
        tests, region_params = [], ['regions']
        if role:
            tests.append("lower(json_extract(j.value,'$.role'))=?")
            region_params.append(role.casefold())
        if task:
            tests.append("CASE WHEN json_type(j.value,'$.task_candidates')='array' THEN EXISTS (SELECT 1 FROM json_each(json_extract(j.value,'$.task_candidates')) k WHERE lower(k.value)=?) ELSE 0 END")
            region_params.append(task.casefold())
        clauses.append("SELECT entry_id FROM terms t WHERE field=? AND EXISTS (SELECT 1 FROM json_each(t.value) j WHERE CASE WHEN j.type='object' THEN " + ' AND '.join(tests) + " ELSE 0 END)")
        params.extend(region_params)
    for key, (field, operator) in NUMERIC_FILTERS.items():
        value = getattr(args, key, None)
        if value is not None:
            clauses.append('SELECT entry_id FROM terms WHERE field=? AND CAST(json_extract(value,\'$\') AS REAL) ' + operator + ' ?')
            params.extend((field, value))
    where = 'e.id IN (' + ' INTERSECT '.join(clauses) + ')' if clauses else '1=1'
    if only_path is not None:
        where += ' AND e.path_key=?'
        params.append(path_key(only_path))
    rows = db.execute('SELECT e.path,e.entry_json,p.descriptor,r.shared_json FROM evidence e '
                      'JOIN partitions p ON p.name=e.partition_name '
                      'LEFT JOIN manifest_rows r ON r.id=e.manifest_row_id WHERE ' + where +
                      " ORDER BY json_extract(p.descriptor,'$.ordinal'),e.position", params)
    for path, raw, descriptor, shared_json in rows:
        if shared_json is not None:
            entry = json.loads(raw)
            entry['evidence'][0].update(json.loads(shared_json))
            raw = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
        yield path, raw, descriptor


def validated(db, allow_partial):
    meta = dict(db.execute('SELECT key,value FROM meta'))
    if meta.get('schema_version') != SCHEMA_VERSION or meta.get('state') != 'ready':
        raise ValueError('SQLite cache incomplete; run sync again')
    parts, policy, state = snapshot()
    names = json.loads(meta['scope'])
    selected = selected_snapshot(parts, names)
    cache = dict(db.execute('SELECT name,descriptor FROM partitions'))
    if (policy != normalize_policy(meta.get('policy')) or set(cache) != set(names) or
            any(cache[p['name']] != canonical(p) for p in selected)):
        raise ValueError('SQLite cache is stale; run sync again')
    full = meta.get('complete_catalog') == '1'
    if full and (len(parts) != len(names) or state != 'complete'):
        raise ValueError('SQLite cache is stale or JSON build is incomplete')
    if not full and not allow_partial:
        raise ValueError('partial cache: pass --allow-partial for explicitly scoped results')
    return meta, selected, full


def snapshot_notice(meta, selected, full):
    return {'type': 'snapshot', 'state': 'ready', 'snapshot_utc': meta.get('snapshot_utc'),
            'partition_count': len(selected), 'complete_catalog': full,
            'missing_sources': [p['name'] for p in selected if not p['source_present']],
            'unparsed_manifests': [w for p in selected for w in p['warnings']
                                   if w.get('error') == 'manifest exceeds 64 MiB; not parsed'],
            'freshness': 'catalog_snapshot; refresh JSON after external file changes'}


def query(args):
    db = connect(readonly=True)
    try:
        db.execute('BEGIN')
        meta, selected, full = validated(db, args.allow_partial)
        names = [p['name'] for p in selected]
        if not full:
            print('partial cache: results and evidence cover only synced partitions', file=sys.stderr)
        print(json.dumps(snapshot_notice(meta, selected, full), ensure_ascii=False), file=sys.stderr)
        if args.path:
            only_path = core.relative(core.inside(args.path))
        else:
            only_path = None
        joins = bool(args.with_evidence or any(getattr(args, k) for k in FILTER_FIELDS) or
                     any(getattr(args, k, None) is not None for k in NUMERIC_FILTERS))
        numeric_requested = any(getattr(args, k, None) is not None for k in NUMERIC_FILTERS)
        matching = {}
        matching_keys = {}
        if joins:
            for path, raw, descriptor in matching_entries(db, args, only_path):
                partition = json.loads(descriptor)
                if core.lifecycle_reason(partition['name']) or (not args.include_missing and not partition['source_present']):
                    continue
                entry = json.loads(raw)
                entry['evidence'] = [r for r in entry['evidence'] if r.get('relation') != 'batch_context']
                if not entry['evidence']:
                    continue
                if not core.claim_matches(entry['evidence'], args):
                    continue
                for record in entry['evidence']:
                    sig = (record['manifest'], record['record_index'], record.get('relation'), record.get('reference_field'))
                    key = path_key(path)
                    if sig not in matching_keys.setdefault(key, set()):
                        matching.setdefault(key, []).append(record)
                        matching_keys[key].add(sig)
        conditions, values = [], []
        if only_path:
            conditions.append('m.path_key=?')
            values.append(path_key(only_path))
        if args.partition:
            # Python casefold matches the existing CLI even for non-ASCII names.
            allowed = [p['name'] for p in selected if args.partition.casefold() in p['name'].casefold()]
            if not allowed:
                return 0
            conditions.append('m.partition_name IN (' + ','.join('?' for _ in allowed) + ')')
            values.extend(allowed)
        sql = ('SELECT m.path,m.path_key,m.item_json,p.descriptor FROM media m '
               'JOIN partitions p ON p.name=m.partition_name')
        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)
        # Catalog order breaks timestamp ties just as Python's stable sort does.
        sql += " ORDER BY json_extract(p.descriptor,'$.updated_utc') DESC, json_extract(p.descriptor,'$.ordinal'),m.position"
        seen, count = set(), 0
        for path, key_path, raw, desc in db.execute(sql, values):
            part = json.loads(desc)
            if core.lifecycle_reason(part['name']) or (not args.include_missing and not part['source_present']):
                continue
            if core.lifecycle_reason(path) or key_path in seen:
                continue
            seen.add(key_path)
            if args.kind and json.loads(raw)['media_type'] != args.kind:
                continue
            if numeric_requested and key_path not in matching:
                continue
            item = json.loads(raw)
            item['evidence'] = [r for r in item.get('evidence', []) if r.get('relation') != 'batch_context']
            if joins:
                joined = list(item.get('evidence', []))
                signatures = {(x['manifest'], x['record_index'], x.get('relation'), x.get('reference_field')) for x in joined}
                for record in matching.get(key_path, []):
                    sig = (record['manifest'], record['record_index'], record.get('relation'), record.get('reference_field'))
                    if sig not in signatures:
                        joined.append(record)
                        signatures.add(sig)
                item['evidence'] = joined
                item['recorded_claims'] = core.structured_claims(joined)
            else:
                item['recorded_claims'] = core.structured_claims(item.get('evidence', []))
            if args.with_evidence and not item['evidence']:
                continue
            if not core.claim_matches(item['evidence'], args):
                continue
            if args.contains and args.contains.casefold() not in json.dumps(item, ensure_ascii=False).casefold():
                continue
            if not args.include_missing and not (core.ROOT / item['path']).is_file():
                continue
            print(json.dumps(item, ensure_ascii=False))
            count += 1
            if args.limit and count >= args.limit:
                break
        return 0
    finally:
        db.close()


def evidence_query(args):
    db = connect(readonly=True)
    try:
        db.execute('BEGIN')
        meta, selected, full = validated(db, args.allow_partial)
        print(json.dumps(snapshot_notice(meta, selected, full), ensure_ascii=False), file=sys.stderr)
        path = core.relative(core.inside(args.path)) if args.path else None
        count = 0
        for claimed_path, raw, descriptor in matching_entries(db, args, path):
            part = json.loads(descriptor)
            if (core.lifecycle_reason(part['name']) or core.lifecycle_reason(claimed_path) or
                    (not args.include_missing and not part['source_present'])):
                continue
            entry = json.loads(raw)
            entry['evidence'] = [r for r in entry['evidence'] if r.get('relation') != 'batch_context']
            if not entry['evidence']:
                continue
            if not core.claim_matches(entry['evidence'], args):
                continue
            entry['recorded_claims'] = core.structured_claims(entry['evidence'])
            print(json.dumps(entry, ensure_ascii=False))
            count += 1
            if args.limit and count >= args.limit:
                break
        return 0
    finally:
        db.close()


def batches_query(args):
    db = connect(readonly=True)
    try:
        db.execute('BEGIN')
        meta, selected, full = validated(db, args.allow_partial)
        print(json.dumps(snapshot_notice(meta, selected, full), ensure_ascii=False), file=sys.stderr)
        sql = ('SELECT b.record_json,p.descriptor FROM batches b '
               'JOIN evidence e ON e.id=b.entry_id JOIN partitions p ON p.name=e.partition_name '
               'WHERE 1=1')
        values = []
        # Restrict on the full JSON text only. Batch scope and status can be
        # nested or non-string values; Python's final comparison preserves the
        # JSON CLI semantics for escaping and substring matches.
        if args.contains:
            sql += ' AND instr(b.text_cf,?)>0'
            values.append(args.contains.casefold())
        sql += " ORDER BY json_extract(p.descriptor,'$.ordinal'),e.position"
        count = 0
        for raw, descriptor in db.execute(sql, values):
            part = json.loads(descriptor)
            if core.lifecycle_reason(part['name']) or (not args.include_missing and not part['source_present']):
                continue
            record = json.loads(raw)
            # Match the JSON CLI's individual-value semantics, not cross-field
            # substring matches spanning serialized list punctuation.
            context = record['batch_context']
            evaluation = context.get('evaluation')
            evaluation = evaluation if isinstance(evaluation, dict) else {}
            if args.evaluation_scope:
                scopes = [context.get('evaluation_scope'), context.get('conclusion_scope'), evaluation.get('data_scope')]
                if not any(args.evaluation_scope.casefold() in json.dumps(x, ensure_ascii=False).casefold()
                           for x in scopes if x is not None):
                    continue
            if args.status:
                statuses = [context.get('status'), evaluation.get('status')]
                if not any(args.status.casefold() in str(x).casefold() for x in statuses if x is not None):
                    continue
            print(json.dumps(record, ensure_ascii=False))
            count += 1
            if args.limit and count >= args.limit:
                break
        return 0
    finally:
        db.close()


def task_input_records(db, task, *, kind='plan', split=None, role=None, plan_key=None, path=None):
    """Read current task declarations from the cache, never execute their plans.

    Source YAML and plan JSON remain authoritative. File stamps reject stale
    declarations without hashing files or scanning original media partitions.
    """
    if task == 'fire_sticker':
        task = 'fire'
    if task not in {'person', 'e_bike', 'fire', 'building_name_paper', 'sign',
                    'waste_bin_state', 'waste_sticker', 'meter'}:
        raise ValueError('unknown dataset task: ' + task)
    partition = 'datasets/scripts/' + task
    rows = [json.loads(raw).get('claims', {}) for (raw,) in db.execute(
        'SELECT DISTINCT shared_json FROM manifest_rows WHERE partition_name=?', (partition,))]
    configs = [row['task_config'] for row in rows if 'task_config' in row]
    if len(configs) != 1:
        raise ValueError(f'{partition}: task configuration is not indexed; build and sync this partition')
    config = configs[0]
    if config.get('schema_version') != 1 or config.get('task') != task:
        raise ValueError(f'{partition}: unsupported or mismatched task declaration')
    for stamp in config['declared_files']:
        file = core.inside(stamp['path'])
        present = file.is_file()
        if present != stamp['present'] or (present and (
                file.stat().st_size != stamp['size'] or file.stat().st_mtime_ns != stamp['mtime_ns'])):
            raise ValueError(f'{partition}: declaration changed: {stamp["path"]}; refresh this partition')
    states = [row['plan_state'] for row in rows if 'plan_state' in row]
    states = [state for state in states if
              (not split or state['target_split'] == split) and
              (not plan_key or state['plan_key'] == plan_key)]
    result = []
    if kind in ('config', 'all'):
        result.append({'kind': 'config', 'task': task, 'partition': partition,
                       'declaration': config})
    if kind in ('plan', 'all'):
        for plan in config['plans']:
            if plan_key and plan['plan_key'] != plan_key:
                continue
            if split and plan.get('target_split') not in (None, split):
                continue
            result.append({'kind': 'plan', 'task': task, **plan,
                           'status': 'declared' if plan['present'] else 'pending_plan',
                           'state_count': sum(s['plan_key'] == plan['plan_key'] for s in states),
                           'source_config': partition + '/config/source.yaml',
                           'dataset_config': partition + '/config/dataset.yaml'})
    if kind == 'state':
        result.extend({'kind': 'state', **state} for state in states)
    if kind in ('input', 'all'):
        inputs = {}
        for state in states:
            for pointer, value in core.task_declared_paths(state['stratum']):
                input_role = _declared_input_role(pointer)
                if role and input_role != role:
                    continue
                if path and path_key(value) != path_key(path):
                    continue
                key = (state['plan_key'], value, input_role)
                item = inputs.setdefault(key, {
                    'kind': 'input', 'task': task, 'split': state['target_split'],
                    'plan_key': state['plan_key'], 'plan_path': state['plan_path'],
                    'path': value, 'role': input_role,
                    'status': 'planned_input', 'present': core.inside(value).is_file(),
                    'reference_fields': [], 'state_ids': []})
                if pointer not in item['reference_fields']:
                    item['reference_fields'].append(pointer)
                if state['stratum']['id'] not in item['state_ids']:
                    item['state_ids'].append(state['stratum']['id'])
        result.extend(inputs[key] for key in sorted(inputs))
    return result


def _declared_input_role(pointer):
    # Roles describe declaration locations, not an inferred visual suitability.
    parts = pointer.split('/')
    if any(word in parts for word in ('parent', 'parent_source', 'parent_video', 'source_video')):
        return 'parent_source'
    if any(word in parts for word in ('manifest', 'geometry_manifest', 'annotation_manifest')):
        return 'annotation_reference'
    if pointer.startswith('/background/') or pointer == '/parameters/background_image':
        return 'background'
    if pointer.startswith('/target/') or pointer.startswith('/parameters/actors/'):
        return 'target'
    if pointer.startswith('/labels/'):
        return 'annotation_reference'
    return 'declared_input'


def task_inputs_query(args):
    db = connect(readonly=True)
    try:
        db.execute('BEGIN')
        meta, selected, full = validated(db, args.allow_partial)
        tasks = [args.task] if args.task else ['person', 'e_bike', 'fire', 'building_name_paper',
                                              'sign', 'waste_bin_state', 'waste_sticker', 'meter']
        path = core.relative(core.inside(args.path)) if args.path else None
        # Validate every requested task before printing any records.
        records = [record for task in tasks for record in task_input_records(
            db, task, kind=args.kind, split=args.split, role=args.role,
            plan_key=args.plan_key, path=path)]
        print(json.dumps(snapshot_notice(meta, selected, full), ensure_ascii=False), file=sys.stderr)
        for record in records[:args.limit or None]:
            print(json.dumps(record, ensure_ascii=False))
        return 0
    finally:
        db.close()


def status():
    db = connect(readonly=True)
    try:
        db.execute('BEGIN')
        meta, selected, full = validated(db, True)
        result = snapshot_notice(meta, selected, full)
        result['partitions'] = [p['name'] for p in selected]
        print(json.dumps(result, ensure_ascii=False))
        return 0
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('sync', help='synchronize versioned JSON partitions into the cache')
    build.add_argument('--partition', action='append', default=[], help='exact partition name; repeat for scoped sync')
    find = sub.add_parser('query', help='query the synchronized cache')
    find.add_argument('--path', help='exact workspace-relative media path')
    find.add_argument('--partition', help='casefold substring of partition name')
    find.add_argument('--contains')
    find.add_argument('--kind', choices=('image', 'video', 'array_asset'))
    find.add_argument('--with-evidence', action='store_true')
    find.add_argument('--include-missing', action='store_true')
    find.add_argument('--allow-partial', action='store_true')
    find.add_argument('--limit', type=int, default=20)
    evidence = sub.add_parser('evidence', help='query reverse source and parent relations')
    evidence.add_argument('path', nargs='?', help='workspace-relative media or source path')
    evidence.add_argument('--include-missing', action='store_true')
    evidence.add_argument('--allow-partial', action='store_true')
    evidence.add_argument('--limit', type=int, default=20)
    batches = sub.add_parser('batches', help='query explicit batch-level context and evaluation scope')
    batches.add_argument('--contains')
    batches.add_argument('--evaluation-scope')
    batches.add_argument('--status')
    batches.add_argument('--include-missing', action='store_true')
    batches.add_argument('--allow-partial', action='store_true')
    batches.add_argument('--limit', type=int, default=20)
    sub.add_parser('status', help='report whether the cache matches the JSON snapshot')
    task_inputs = sub.add_parser('task-inputs', help='query current task source declarations and plans')
    task_inputs.add_argument('--task', help='task directory name; omitted queries the eight current tasks')
    task_inputs.add_argument('--kind', choices=('config', 'plan', 'input', 'state', 'all'), default='plan')
    task_inputs.add_argument('--split', choices=('train', 'val', 'test', 'unassigned'))
    task_inputs.add_argument('--role', help='background, target, parent_source, annotation_reference or declared_input')
    task_inputs.add_argument('--plan-key', choices=('production_plan', 'render_plan', 'direct_real_plan', 'blender_plan'))
    task_inputs.add_argument('--path', help='exact input path for reverse task lookup; use --kind input')
    task_inputs.add_argument('--allow-partial', action='store_true')
    task_inputs.add_argument('--limit', type=int, default=20, help='maximum records; 0 returns all')
    for action in (find, evidence):
        for option in FILTER_FIELDS:
            action.add_argument('--' + option.replace('_', '-'))
        for option in NUMERIC_FILTERS:
            action.add_argument('--' + option.replace('_', '-'), type=float)
    args = parser.parse_args(argv)
    if args.command in ('query', 'evidence', 'batches', 'task-inputs') and args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.command == 'task-inputs' and (args.role or args.path) and args.kind != 'input':
        parser.error('--role and --path require --kind input')
    try:
        if args.command == 'sync':
            return sync(args)
        if args.command == 'query':
            return query(args)
        if args.command == 'evidence':
            return evidence_query(args)
        if args.command == 'batches':
            return batches_query(args)
        if args.command == 'task-inputs':
            return task_inputs_query(args)
        return status()
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        if args.command == 'status':
            print(json.dumps({'state': 'stale_or_unavailable', 'reason': str(exc)}, ensure_ascii=False))
            return 1
        print('SQLite media cache:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
