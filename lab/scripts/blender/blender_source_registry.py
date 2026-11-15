"""Authoritative SQLite registry for Blender source assets and their lifecycle.

Unlike the media query cache, this database is maintained through explicit
registry commands and is never replaced by media-index synchronization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / 'datasets' / 'blender_source_registry.sqlite3'
SCHEMA_VERSION = 1


def connect(readonly=False):
    if readonly:
        if not DATABASE.is_file():
            raise ValueError('registry database is missing; run init first')
        return sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True)
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE)
    db.execute('PRAGMA foreign_keys=ON')
    return db


def init(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            current_path TEXT NOT NULL UNIQUE,
            asset_type TEXT NOT NULL,
            task TEXT,
            classification TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL,
            truth_level TEXT NOT NULL,
            evidence_ref TEXT NOT NULL,
            note TEXT NOT NULL,
            updated_utc TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS asset_paths (
            path TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL REFERENCES assets(asset_id),
            path_role TEXT NOT NULL,
            evidence_ref TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS asset_paths_asset ON asset_paths(asset_id);
        CREATE TABLE IF NOT EXISTS asset_relations (
            relation_id INTEGER PRIMARY KEY,
            subject_asset_id TEXT REFERENCES assets(asset_id),
            subject_path TEXT,
            relation TEXT NOT NULL,
            object_asset_id TEXT REFERENCES assets(asset_id),
            object_path TEXT,
            evidence_ref TEXT NOT NULL,
            record_index TEXT,
            note TEXT NOT NULL,
            UNIQUE(subject_asset_id,subject_path,relation,object_asset_id,object_path,evidence_ref,record_index));
        CREATE INDEX IF NOT EXISTS asset_relations_subject ON asset_relations(subject_asset_id);
        CREATE INDEX IF NOT EXISTS asset_relations_object ON asset_relations(object_asset_id);
        CREATE TABLE IF NOT EXISTS asset_events (
            event_id INTEGER PRIMARY KEY,
            asset_id TEXT NOT NULL REFERENCES assets(asset_id),
            event_type TEXT NOT NULL,
            before_json TEXT,
            after_json TEXT NOT NULL,
            evidence_ref TEXT NOT NULL,
            recorded_utc TEXT NOT NULL);
    ''')
    row = db.execute("SELECT value FROM registry_meta WHERE key='schema_version'").fetchone()
    if row and row[0] != str(SCHEMA_VERSION):
        raise ValueError('unsupported registry schema version')
    db.execute("INSERT OR IGNORE INTO registry_meta VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
    db.execute("INSERT OR REPLACE INTO registry_meta VALUES('authority','blender_source_assets')")
    db.execute('''UPDATE asset_relations SET subject_asset_id=(SELECT asset_id FROM asset_paths
                WHERE path=asset_relations.subject_path) WHERE subject_asset_id IS NULL AND subject_path IS NOT NULL''')
    db.execute('''UPDATE asset_relations SET object_asset_id=(SELECT asset_id FROM asset_paths
                WHERE path=asset_relations.object_path) WHERE object_asset_id IS NULL AND object_path IS NOT NULL''')
    log_missing_relation_events(db)
    db.commit()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def relative(path):
    value = Path(path).as_posix()
    if Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('use a workspace-relative path without parent traversal')
    return value


def asset_id(path):
    return 'blender:' + relative(path)


def inventory_defaults(path):
    value = relative(path)
    lower = value.casefold()
    task = None
    parts = Path(value).parts
    if len(parts) > 2 and parts[0] == 'datasets' and parts[1] == 'assets' and parts[2] == 'blender':
        if len(parts) > 3 and parts[3] == 'batch_scenes':
            task = parts[4] if len(parts) > 4 else None
        else:
            task = parts[3] if len(parts) > 3 and parts[3] != 'shared' else None
    if lower.startswith('archive/technical/datasets/'):
        classification, state = 'historical_batch_scene', 'archived'
    elif lower.startswith('archive/technical/revisions/'):
        classification, state = 'historical_model_revision', 'archived'
    elif lower.startswith('datasets/assets/blender/batch_scenes/'):
        classification, state = 'batch_scene_snapshot', 'retained_snapshot'
    elif lower.startswith('tmp/'):
        classification, state = 'temporary_fixture', 'unresolved'
    elif lower.endswith('.blend1'):
        classification, state = 'automatic_backup_candidate', 'review_pending'
    else:
        classification, state = 'source_or_experiment_model', 'review_pending'
    return task, classification, state


def record_event(db, key, event_type, before, after, evidence):
    db.execute('INSERT INTO asset_events(asset_id,event_type,before_json,after_json,evidence_ref,recorded_utc) '
               'VALUES(?,?,?,?,?,?)', (key, event_type,
               json.dumps(before, ensure_ascii=False, sort_keys=True) if before else None,
               json.dumps(after, ensure_ascii=False, sort_keys=True), evidence, now()))


def log_missing_relation_events(db):
    known = set()
    for raw, in db.execute("SELECT after_json FROM asset_events WHERE event_type='relation_registered'"):
        try:
            relation_id = json.loads(raw).get('relation_id')
            if relation_id is not None:
                known.add(int(relation_id))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    rows = db.execute('''SELECT relation_id,subject_asset_id,subject_path,relation,object_asset_id,
                         object_path,evidence_ref,record_index,note FROM asset_relations ORDER BY relation_id''')
    for row in rows:
        relation_id = row[0]
        if relation_id in known:
            continue
        fields = ('relation_id','subject_asset_id','subject_path','relation','object_asset_id',
                  'object_path','evidence_ref','record_index','note')
        relation = dict(zip(fields, row))
        owner = row[1] or row[4]
        if owner:
            record_event(db, owner, 'relation_registered', None, relation, row[6])


def insert_relation(db, subject_id, subject_path, relation_name, object_id, object_path,
                    evidence, record_index, note):
    db.execute('''INSERT OR IGNORE INTO asset_relations(subject_asset_id,subject_path,relation,
                object_asset_id,object_path,evidence_ref,record_index,note)
                VALUES(?,?,?,?,?,?,?,?)''',
               (subject_id, subject_path, relation_name, object_id, object_path,
                evidence, record_index, note))
    inserted = bool(db.execute('SELECT changes()').fetchone()[0])
    row = db.execute('''SELECT relation_id,subject_asset_id,subject_path,relation,object_asset_id,
                        object_path,evidence_ref,record_index,note FROM asset_relations
                        WHERE subject_asset_id IS ? AND subject_path IS ? AND relation=?
                        AND object_asset_id IS ? AND object_path IS ? AND evidence_ref=?
                        AND record_index IS ?''',
                     (subject_id, subject_path, relation_name, object_id, object_path,
                      evidence, record_index)).fetchone()
    if inserted and row:
        names = ('relation_id','subject_asset_id','subject_path','relation','object_asset_id',
                 'object_path','evidence_ref','record_index','note')
        record_event(db, subject_id or object_id, 'relation_registered', None,
                     dict(zip(names, row)), evidence)
    return inserted


def register(args):
    path = relative(args.path)
    if not (ROOT / path).is_file() and not args.allow_missing:
        raise ValueError('asset file is not present; use --allow-missing only for a documented historical path')
    key = args.asset_id or asset_id(path)
    task, default_class, default_state = inventory_defaults(path)
    with connect() as db:
        init(db)
        old = db.execute('SELECT current_path,asset_type,task,classification,lifecycle_state,truth_level,evidence_ref,note '
                         'FROM assets WHERE asset_id=?', (key,)).fetchone()
        path_owner = db.execute('SELECT asset_id FROM asset_paths WHERE path=?', (path,)).fetchone()
        if path_owner and path_owner[0] != key:
            raise ValueError('path is already registered to another asset: ' + path)
        current = (path,
                   args.asset_type or (old[1] if old else
                       ('blender_backup' if Path(path).suffix.casefold() == '.blend1' else 'blender_scene')),
                   args.task if args.task is not None else (old[2] if old else task),
                   args.classification or (old[3] if old else default_class),
                   args.state or (old[4] if old else default_state),
                   args.truth_level or (old[5] if old else 'unknown'), args.evidence, args.note)
        if old and old[0] != path:
            db.execute("UPDATE asset_paths SET path_role='historical_location' WHERE path=? AND asset_id=?",
                       (old[0], key))
        db.execute('''INSERT INTO assets(asset_id,current_path,asset_type,task,classification,lifecycle_state,
                    truth_level,evidence_ref,note,updated_utc) VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(asset_id) DO UPDATE SET current_path=excluded.current_path,
                    asset_type=excluded.asset_type,task=excluded.task,classification=excluded.classification,
                    lifecycle_state=excluded.lifecycle_state,truth_level=excluded.truth_level,
                    evidence_ref=excluded.evidence_ref,note=excluded.note,updated_utc=excluded.updated_utc''',
                   (key, *current, now()))
        db.execute('''INSERT INTO asset_paths(path,asset_id,path_role,evidence_ref) VALUES(?,?,?,?)
                      ON CONFLICT(path) DO UPDATE SET path_role='current',evidence_ref=excluded.evidence_ref''',
                   (path, key, 'current', args.evidence))
        if old != current:
            record_event(db, key, 'register' if old is None else 'update', old, current, args.evidence)
    print(json.dumps({'asset_id': key, 'path': path, 'classification': current[3],
                      'state': current[4], 'recorded': old != current}, ensure_ascii=False))


def relate(args):
    subject = args.subject_id
    object_id = args.object_id
    subject_path = relative(args.subject_path) if args.subject_path else None
    object_path = relative(args.object_path) if args.object_path else None
    if not subject and not subject_path:
        raise ValueError('provide --subject-id or --subject-path')
    if not object_id and not object_path:
        raise ValueError('provide --object-id or --object-path')
    with connect() as db:
        init(db)
        if subject is None and subject_path:
            row = db.execute('SELECT asset_id FROM asset_paths WHERE path=?', (subject_path,)).fetchone()
            subject = row[0] if row else None
        if object_id is None and object_path:
            row = db.execute('SELECT asset_id FROM asset_paths WHERE path=?', (object_path,)).fetchone()
            object_id = row[0] if row else None
        inserted = insert_relation(db, subject, subject_path, args.relation, object_id,
                                   object_path, args.evidence, args.record_index, args.note)
    print(json.dumps({'relation': args.relation, 'inserted': inserted}, ensure_ascii=False))


def import_batch_manifests(args):
    with connect() as db:
        init(db)
        scene_rows = list(db.execute("SELECT asset_id,current_path FROM assets "
                                     "WHERE classification='batch_scene_snapshot' ORDER BY current_path"))
        path_assets = dict(db.execute('SELECT path,asset_id FROM asset_paths'))
        path_assets.update(dict(db.execute('SELECT current_path,asset_id FROM assets')))
        matched = 0
        scene_links = 0
        source_links = 0
        unresolved = []
        for scene_id, scene_path in scene_rows:
            parts = Path(scene_path).parts
            task, batch = parts[4], parts[5]
            folders = [ROOT / 'datasets/exports' / task / batch]
            if task == 'waste_bin':
                folders.append(ROOT / 'datasets/exports/waste_bin_state' / batch)
            matches = []
            for folder in folders:
                if not folder.is_dir():
                    continue
                for manifest in folder.rglob('*.json'):
                    try:
                        data = json.loads(manifest.read_text(encoding='utf8'))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    for field in ('editable_preview_scene', 'scene_copy'):
                        value = data.get(field)
                        if not isinstance(value, str):
                            continue
                        normalized = value.replace('\\', '/')
                        value_path = Path(normalized)
                        if normalized.startswith('datasets/'):
                            resolved = (ROOT / value_path).resolve()
                        else:
                            resolved = (manifest.parent / value_path).resolve()
                        if normalized == scene_path or resolved == (ROOT / scene_path).resolve():
                            matches.append((manifest, data, field))
            if len(matches) != 1:
                unresolved.append({'scene': scene_path, 'manifest_matches': len(matches)})
                continue
            matched += 1
            manifest, data, scene_field = matches[0]
            manifest_path = manifest.relative_to(ROOT).as_posix()
            batch_path = manifest.parent.relative_to(ROOT).as_posix()
            if insert_relation(db, scene_id, scene_path, 'scene_snapshot_for_batch', None,
                               batch_path, manifest_path, '$.' + scene_field,
                               'Manifest maps this editable scene snapshot to the export batch; this does not approve the scene or dataset.'):
                scene_links += 1
            for field in ('source_scene', 'source_blend', 'scene_file', 'bin_source', 'mother_blend', 'mother_scene'):
                value = data.get(field)
                if not isinstance(value, str) or not value.replace('\\', '/').casefold().endswith('.blend'):
                    continue
                source_path = value.replace('\\', '/')
                if not source_path.startswith('datasets/') and not source_path.startswith('Archive/'):
                    source_path = (manifest.parent / source_path).resolve().relative_to(ROOT).as_posix()
                source_id = path_assets.get(source_path)
                if source_id is None:
                    unresolved.append({'scene': scene_path, 'manifest': manifest_path,
                                       'field': field, 'source_path': source_path,
                                       'reason': 'source path is not registered'})
                    continue
                relation_name = 'manifest_declares_source_scene' if field in {
                    'source_scene','scene_file','mother_blend','mother_scene'} else 'manifest_declares_source_model'
                note = ('Manifest records this source field; the registry preserves the claim and does not treat it as an independent verification of model quality or scene fit: '
                        + field)
                if insert_relation(db, source_id, source_path, relation_name, None,
                                   batch_path, manifest_path, '$.' + field, note):
                    source_links += 1
    print(json.dumps({'snapshot_assets': len(scene_rows), 'matched_manifests': matched,
                      'scene_batch_links_added': scene_links, 'source_claim_links_added': source_links,
                      'unresolved': unresolved}, ensure_ascii=False))


def alias(args):
    path = relative(args.path)
    with connect() as db:
        init(db)
        if db.execute('SELECT 1 FROM assets WHERE asset_id=?', (args.asset_id,)).fetchone() is None:
            raise ValueError('asset_id is not registered')
        owner = db.execute('SELECT asset_id FROM asset_paths WHERE path=?', (path,)).fetchone()
        if owner and owner[0] != args.asset_id:
            raise ValueError('path is already assigned to another asset')
        db.execute('INSERT OR IGNORE INTO asset_paths VALUES(?,?,?,?)',
                   (path, args.asset_id, args.role, args.evidence))
        if db.execute('SELECT changes()').fetchone()[0]:
            record_event(db, args.asset_id, 'add_path_alias', None,
                         {'path': path, 'role': args.role}, args.evidence)
    print(json.dumps({'asset_id': args.asset_id, 'path': path, 'role': args.role}, ensure_ascii=False))


def correct_relation(args):
    with connect() as db:
        init(db)
        row = db.execute('''SELECT subject_asset_id,subject_path,relation,object_asset_id,object_path,
                            evidence_ref,record_index,note FROM asset_relations WHERE relation_id=?''',
                         (args.relation_id,)).fetchone()
        if row is None:
            raise ValueError('relation_id not found')
        after = (row[0], row[1], args.relation or row[2], row[3], row[4],
                 args.evidence or row[5], args.record_index if args.record_index is not None else row[6],
                 args.note if args.note is not None else row[7])
        db.execute('''UPDATE asset_relations SET relation=?,evidence_ref=?,record_index=?,note=?
                      WHERE relation_id=?''', (after[2], after[5], after[6], after[7], args.relation_id))
        actor = row[0] or row[3]
        if actor:
            record_event(db, actor, 'correct_relation', row, after, args.evidence or row[5])
    print(json.dumps({'relation_id': args.relation_id, 'before': row, 'after': after}, ensure_ascii=False))


def listing(args):
    db = connect(readonly=True)
    try:
        query = ('SELECT asset_id,current_path,asset_type,task,classification,lifecycle_state,truth_level,evidence_ref,note '
                 'FROM assets')
        params = ()
        if args.state:
            query += ' WHERE lifecycle_state=?'
            params = (args.state,)
        query += ' ORDER BY current_path'
        rows = [dict(zip(('asset_id','path','asset_type','task','classification','state','truth_level','evidence_ref','note'), row))
                for row in db.execute(query, params)]
        print(json.dumps({'database': DATABASE.relative_to(ROOT).as_posix(), 'count': len(rows), 'assets': rows},
                         ensure_ascii=False, indent=2))
    finally:
        db.close()


def show(args):
    db = connect(readonly=True)
    try:
        row = db.execute('SELECT * FROM assets WHERE asset_id=? OR current_path=?',
                         (args.asset, args.asset)).fetchone()
        if row is None:
            raise ValueError('asset not found')
        names = [x[1] for x in db.execute('PRAGMA table_info(assets)')]
        result = dict(zip(names, row))
        result['paths'] = [dict(zip(('path','role','evidence_ref'), r)) for r in
                           db.execute('SELECT path,path_role,evidence_ref FROM asset_paths WHERE asset_id=?', (row[0],))]
        result['relations'] = [dict(zip(('subject_asset_id','subject_path','relation','object_asset_id','object_path',
                                         'evidence_ref','record_index','note'), r)) for r in
                               db.execute('SELECT subject_asset_id,subject_path,relation,object_asset_id,object_path,'
                                          'evidence_ref,record_index,note FROM asset_relations WHERE subject_asset_id=? '
                                          'OR object_asset_id=?', (row[0], row[0]))]
        result['events'] = [dict(zip(('event_type','before_json','after_json','evidence_ref','recorded_utc'), r)) for r in
                            db.execute('SELECT event_type,before_json,after_json,evidence_ref,recorded_utc '
                                       'FROM asset_events WHERE asset_id=? ORDER BY event_id', (row[0],))]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        db.close()


def bootstrap(args):
    roots = [ROOT / 'datasets/assets/blender', ROOT / 'Archive/technical/revisions',
             ROOT / 'Archive/technical/datasets', ROOT / 'tmp']
    files = sorted({p for root in roots if root.exists() for p in root.rglob('*')
                    if p.is_file() and p.suffix.casefold() in {'.blend', '.blend1'}},
                   key=lambda p: p.as_posix().casefold())
    inserted = 0
    reconciled = 0
    with connect() as db:
        init(db)
        for file in files:
            path = file.relative_to(ROOT).as_posix()
            key = asset_id(path)
            task, classification, state = inventory_defaults(path)
            if path.startswith('Archive/technical/revisions/'):
                evidence = 'Archive/technical/revisions/README.md'
                note = 'Archive inventory baseline; version-family mapping is in the revision README, exact per-file lineage still needs review.'
            elif path.startswith('Archive/technical/datasets/'):
                evidence = 'Archive/technical/dataset-versions.tsv'
                note = 'Archived batch scene inventory; consult the version table for mapped batches, per-file generation relation still needs review.'
            elif path.startswith('tmp/'):
                evidence = 'datasets/README.md#blender-源资产与派生数据'
                note = 'Temporary Blender fixture; exact creation source and retention period are unknown.'
            else:
                evidence = 'datasets/README.md#blender-源资产与派生数据'
                note = 'Inventory baseline only; detailed per-file lineage/review remains pending.'
            before = db.execute('SELECT current_path,asset_type,task,classification,lifecycle_state,truth_level,evidence_ref,note '
                                'FROM assets WHERE asset_id=?', (key,)).fetchone()
            current = (path, 'blender_scene' if file.suffix.casefold() == '.blend' else 'blender_backup',
                       task, classification, state, 'unknown', evidence, note)
            db.execute('''INSERT OR IGNORE INTO assets(asset_id,current_path,asset_type,task,classification,
                        lifecycle_state,truth_level,evidence_ref,note,updated_utc) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                       (key, *current, now()))
            db.execute('INSERT OR IGNORE INTO asset_paths VALUES(?,?,?,?)', (path, key, 'current', evidence))
            if before is None:
                record_event(db, key, 'inventory_baseline', None, current, evidence)
                inserted += 1
            elif ((before[5] == 'unknown' and before[7].startswith('Inventory baseline only;')) or
                  (before[7].startswith('Archive inventory baseline;')) or
                  (before[7].startswith('Archived batch scene inventory;')) or
                  (before[7].startswith('Inventory location and lifecycle recorded;')) or
                  (before[7].startswith('Inventory baseline only;')) or
                  (before[7].startswith('Temporary Blender fixture;')) or
                  (before[3] == 'batch_scene_snapshot' and before[2] == 'batch_scenes')):
                refreshed = (before[0], before[1], task, classification, state,
                             before[5], evidence, note)
                if before != refreshed:
                    db.execute('UPDATE assets SET task=?,classification=?,lifecycle_state=?,evidence_ref=?,note=?,updated_utc=? '
                               'WHERE asset_id=?', (task, classification, state, evidence, note, now(), key))
                    record_event(db, key, 'inventory_reconcile', before, refreshed, evidence)
                    reconciled += 1
    print(json.dumps({'database': DATABASE.relative_to(ROOT).as_posix(), 'discovered': len(files),
                      'inserted': inserted, 'reconciled': reconciled,
                      'unchanged_or_reviewed': len(files) - inserted - reconciled}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help='create or validate the durable registry schema')
    scan = sub.add_parser('bootstrap', help='register the current Blender inventory as explicitly unreviewed baseline')
    scan.add_argument('--confirm-inventory', action='store_true', required=True,
                      help='confirm this explicit, bounded Blender inventory import')
    reg = sub.add_parser('register', help='register or update one asset with an evidence reference')
    reg.add_argument('path')
    reg.add_argument('--asset-id')
    reg.add_argument('--asset-type')
    reg.add_argument('--task')
    reg.add_argument('--classification')
    reg.add_argument('--state')
    reg.add_argument('--truth-level', choices=('measured','fit_from_photo','approximate','unknown'))
    reg.add_argument('--evidence', required=True)
    reg.add_argument('--note', required=True)
    reg.add_argument('--allow-missing', action='store_true')
    edge = sub.add_parser('relate', help='record one evidence-backed source or lifecycle relation')
    edge.add_argument('--subject-id')
    edge.add_argument('--subject-path')
    edge.add_argument('--relation', required=True)
    edge.add_argument('--object-id')
    edge.add_argument('--object-path')
    edge.add_argument('--evidence', required=True)
    edge.add_argument('--record-index')
    edge.add_argument('--note', default='')
    path_alias = sub.add_parser('alias', help='map a historical or manifest path to a registered asset')
    path_alias.add_argument('asset_id')
    path_alias.add_argument('path')
    path_alias.add_argument('--role', required=True)
    path_alias.add_argument('--evidence', required=True)
    correction = sub.add_parser('correct-relation', help='correct a relation while retaining an asset event record')
    correction.add_argument('relation_id', type=int)
    correction.add_argument('--relation')
    correction.add_argument('--evidence')
    correction.add_argument('--record-index')
    correction.add_argument('--note')
    batch_import = sub.add_parser('import-batch-manifests',
                                  help='import exact batch snapshot/source claims from manifests matched to registered scene snapshots')
    batch_import.add_argument('--confirm-bounded-import', action='store_true', required=True,
                              help='confirm importing only manifests matched to registered batch scene snapshots')
    ls = sub.add_parser('list')
    ls.add_argument('--state')
    get = sub.add_parser('show')
    get.add_argument('asset')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            with connect() as db:
                init(db)
            print(json.dumps({'database': DATABASE.relative_to(ROOT).as_posix(), 'schema_version': SCHEMA_VERSION,
                              'state': 'ready'}, ensure_ascii=False))
        elif args.command == 'bootstrap':
            bootstrap(args)
        elif args.command == 'register':
            register(args)
        elif args.command == 'relate':
            relate(args)
        elif args.command == 'alias':
            alias(args)
        elif args.command == 'correct-relation':
            correct_relation(args)
        elif args.command == 'import-batch-manifests':
            import_batch_manifests(args)
        elif args.command == 'list':
            listing(args)
        else:
            show(args)
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print('Blender source registry:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
